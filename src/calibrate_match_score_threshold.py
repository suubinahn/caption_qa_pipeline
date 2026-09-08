"""
②(캡션-영상 일치도) 절대 컷오프 보정 - calibration/holdout 분리
=====================================================================

validate_match_score.py는 "참 문장이 거짓 문장보다 높은 점수를 받는가"라는
상대 비교(margin)만 확인했다. 여기서는 한 단계 더 나아가 "match_score가
X 이상이면 캡션이 맞다"고 판단할 수 있는 절대 임계값을 찾는다.

검색 precision@K 컷오프 보정 때와 달리, 여기서는 "거짓" 문장을 직접
합성해서 만들기 때문에 표본 크기가 희귀 카테고리에 구애받지 않는다 -
그래서 100->740클립처럼 데이터가 한정되는 문제 없이 원하는 만큼 늘릴 수
있다. 60클립(참 60개 + 거짓 60개 = 120개 라벨링된 점수)을 클립 단위로
calibration(30클립)/holdout(30클립) 분리해서 F1-최대화 임계값을 찾고
holdout에서 검증한다.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))
from search_and_verify import verify_caption_matches_video

FULL_DATASET_JSON = Path("outputs/search_full_dataset.json")
OUTPUT_CSV = Path("outputs/match_score_calibration.csv")
N_CLIPS = 200
N_FRAMES = 5  # 코드리뷰로 발견: 이전엔 3이었는데 search_and_verify.py 프로덕션 기본값(5)과
              # 달라서, match_score(=여러 프레임 중 최댓값)가 프로덕션에서 체계적으로 더
              # 후하게 나오는 문제가 있었다 - 실제로 15개 샘플 테스트에서 판정이 뒤집힌
              # 사례(전부 "불일치"->"일치" 방향)를 확인한 뒤 5로 맞춤.
SEED = 42

FIELD_TEMPLATES = {
    "motion": {
        "moving straight": "The vehicle is moving straight.",
        "turning right": "The vehicle is turning right.",
        "turning left": "The vehicle is turning left.",
        "stop": "The vehicle is stopped.",
        "curved lane driving": "The vehicle is driving through a curve.",
    },
    "weather": {
        "sunny": "The weather is sunny.",
        "rainy": "The weather is rainy.",
        "cloudy": "The weather is cloudy.",
        "snowy": "It is snowing.",
    },
    "time_of_day": {
        "daytime": "It is daytime.",
        "nighttime": "It is nighttime.",
    },
}


def best_f1_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    order = np.argsort(scores)
    sorted_scores = scores[order]
    candidates = [sorted_scores[0] - 1.0]
    candidates += [(sorted_scores[i] + sorted_scores[i + 1]) / 2 for i in range(len(sorted_scores) - 1)]
    candidates.append(sorted_scores[-1] + 1.0)

    best_thr, best_f1 = candidates[0], -1.0
    for thr in candidates:
        pred = scores >= thr
        tp = ((pred == 1) & (labels == 1)).sum()
        fp = ((pred == 1) & (labels == 0)).sum()
        fn = ((pred == 0) & (labels == 1)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_thr, best_f1


def evaluate(scores: np.ndarray, labels: np.ndarray, thr: float) -> dict:
    pred = scores >= thr
    acc = (pred == labels).mean()
    tp = ((pred == 1) & (labels == 1)).sum()
    fp = ((pred == 1) & (labels == 0)).sum()
    fn = ((pred == 0) & (labels == 1)).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    return {"accuracy": acc, "precision": prec, "recall": rec}


def main():
    with open(FULL_DATASET_JSON, encoding="utf-8") as f:
        all_clips = json.load(f)

    random.seed(SEED)
    sample = random.sample(all_clips, N_CLIPS)

    rows = []
    t0 = time.time()
    for i, c in enumerate(sample):
        video_path = Path(c["clip_dir"]) / "1_clip" / "5.mp4"
        if not video_path.exists():
            continue

        for field, templates in FIELD_TEMPLATES.items():
            true_val = c[field]
            if true_val not in templates:
                continue
            true_sentence = templates[true_val]
            wrong_candidates = [v for v in templates if v != true_val]
            wrong_val = random.choice(wrong_candidates)
            wrong_sentence = templates[wrong_val]

            score_true = verify_caption_matches_video(str(video_path), true_sentence, n_frames=N_FRAMES)
            score_wrong = verify_caption_matches_video(str(video_path), wrong_sentence, n_frames=N_FRAMES)

            rows.append({"clip_dir": c["clip_dir"], "field": field, "sentence": true_sentence,
                         "score": score_true, "label": 1})
            rows.append({"clip_dir": c["clip_dir"], "field": field, "sentence": wrong_sentence,
                         "score": score_wrong, "label": 0})

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = elapsed / (i + 1)
            remaining = rate * (len(sample) - i - 1)
            print(f"  진행: {i+1}/{len(sample)}  (경과 {elapsed/60:.1f}분, 예상 잔여 {remaining/60:.1f}분)")

    df = pd.DataFrame(rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    # 클립 단위로 calibration/holdout 분리 (짝/홀 인덱스)
    clip_dirs = sorted(df["clip_dir"].unique())
    calib_clips = set(clip_dirs[i] for i in range(len(clip_dirs)) if i % 2 == 0)
    holdout_clips = set(clip_dirs[i] for i in range(len(clip_dirs)) if i % 2 == 1)

    calib_df = df[df["clip_dir"].isin(calib_clips)]
    holdout_df = df[df["clip_dir"].isin(holdout_clips)]
    print(f"\ncalibration: {len(calib_clips)}클립/{len(calib_df)}점  holdout: {len(holdout_clips)}클립/{len(holdout_df)}점\n")

    # 1) 필드 통합 단일 임계값
    thr, calib_f1 = best_f1_threshold(calib_df["score"].to_numpy(), calib_df["label"].to_numpy())
    hold_metrics = evaluate(holdout_df["score"].to_numpy(), holdout_df["label"].to_numpy(), thr)
    print(f"[통합 임계값] thr={thr:.4f}  calib F1={calib_f1*100:.1f}%")
    print(f"  holdout: accuracy={hold_metrics['accuracy']*100:.1f}%  precision={hold_metrics['precision']*100:.1f}%  recall={hold_metrics['recall']*100:.1f}%")

    # 2) 필드별 임계값
    print(f"\n[필드별 임계값]")
    field_thresholds = {}
    for field in FIELD_TEMPLATES:
        c_sub = calib_df[calib_df["field"] == field]
        h_sub = holdout_df[holdout_df["field"] == field]
        if len(c_sub) < 4 or len(h_sub) < 4:
            continue
        f_thr, f_calib_f1 = best_f1_threshold(c_sub["score"].to_numpy(), c_sub["label"].to_numpy())
        f_hold = evaluate(h_sub["score"].to_numpy(), h_sub["label"].to_numpy(), f_thr)
        field_thresholds[field] = {"threshold": float(f_thr), "holdout": f_hold}
        print(f"  {field:12s} thr={f_thr:.4f}  calib F1={f_calib_f1*100:.1f}%  "
              f"holdout acc={f_hold['accuracy']*100:.1f}% prec={f_hold['precision']*100:.1f}% recall={f_hold['recall']*100:.1f}%")

    result = {
        "unified_threshold": float(thr),
        "unified_holdout": hold_metrics,
        "field_thresholds": field_thresholds,
    }
    with open("outputs/match_score_threshold_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")
    print(f"결과 저장: {OUTPUT_CSV}, outputs/match_score_threshold_result.json")


if __name__ == "__main__":
    main()
