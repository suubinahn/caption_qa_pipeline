"""
road_context, road_type 필드 검증 + 컷오프 보정
====================================================

motion/weather/time_of_day 3개 필드에서 썼던 것과 동일한 절차를
road_context(터널/다리/교차로 등), road_type(차선 수)에도 적용한다.
- 실제 캡션 문장의 67%가 이 5개 필드(기존3+신규2) 중 아무것도 안 걸린다는
  걸 확인한 뒤, 가장 저렴하게 커버리지를 늘릴 수 있는 다음 2개 필드로 선택함.
- "거짓" 문장을 직접 합성하므로(참: 실제 info.txt 값, 거짓: 다른 값)
  희귀 카테고리(bridge=8, roundabout=4 등)에 구애받지 않고 표본을 늘릴 수
  있다 - calibrate_match_score_threshold.py와 동일한 이유.

주의: road_context 필드는 이미 검색(①) 단계에서 "캡션에 거의 그대로
주입되는 필드"(85~97%)라고 확인됐다 - 그래서 이 검증도 "진짜 시각적 이해"
보다는 "주입된 문구가 실제로 존재하는지 확인"에 가까울 수 있다는 한계가
있다. road_type은 injection이 덜해서(19~66%) 상대적으로 더 의미 있는
테스트다.
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
OUTPUT_CSV = Path("outputs/road_fields_calibration.csv")
N_CLIPS = 60
N_FRAMES = 3
SEED = 21

FIELD_TEMPLATES = {
    "road_context": {
        "city street": "The road is a city street.",
        "intersection": "The vehicle is at an intersection.",
        "tunnel": "The car drives through a tunnel.",
        "underpass": "The vehicle passes through an underpass.",
        "overpass": "The vehicle drives under an overpass.",
        "bridge": "The car crosses a bridge.",
        "roundabout": "The vehicle drives through a roundabout.",
    },
    "road_type": {
        "three-lane road": "The road is a three-lane road.",
        "multi-lane highway": "The vehicle drives on a multi-lane highway.",
        "two-lane road": "The road is a two-lane road.",
        "single-lane road": "The road is a single-lane road.",
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
            wrong_val = random.choice([v for v in templates if v != true_val])
            wrong_sentence = templates[wrong_val]

            score_true = verify_caption_matches_video(str(video_path), true_sentence, n_frames=N_FRAMES)
            score_wrong = verify_caption_matches_video(str(video_path), wrong_sentence, n_frames=N_FRAMES)

            rows.append({"clip_dir": c["clip_dir"], "field": field, "sentence": true_sentence,
                         "score": score_true, "label": 1, "margin": score_true - score_wrong})
            rows.append({"clip_dir": c["clip_dir"], "field": field, "sentence": wrong_sentence,
                         "score": score_wrong, "label": 0, "margin": score_true - score_wrong})

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = elapsed / (i + 1)
            remaining = rate * (len(sample) - i - 1)
            print(f"  진행: {i+1}/{len(sample)}  (경과 {elapsed/60:.1f}분, 예상 잔여 {remaining/60:.1f}분)")

    df = pd.DataFrame(rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    # --- 1) margin 검증 (참/거짓 구분력이 노이즈가 아닌지) ---
    print(f"\n{'='*80}\n[1] 참/거짓 margin 검증\n{'='*80}")
    from scipy.stats import ttest_1samp
    for field in FIELD_TEMPLATES:
        pair_margins = df[df["field"] == field].drop_duplicates(subset=["clip_dir", "field"])["margin"].to_numpy()
        pos_rate = (pair_margins > 0).mean()
        t_stat, p = ttest_1samp(pair_margins, 0)
        print(f"{field:15s} n={len(pair_margins):3d}  평균margin={pair_margins.mean():+.4f}  참>거짓비율={pos_rate*100:.1f}%  p={p:.4f}")

    # --- 2) calibration/holdout 컷오프 (클립 단위 짝/홀 분리) ---
    print(f"\n{'='*80}\n[2] calibration/holdout 컷오프 보정\n{'='*80}")
    clip_dirs = sorted(df["clip_dir"].unique())
    calib_clips = set(clip_dirs[i] for i in range(len(clip_dirs)) if i % 2 == 0)
    holdout_clips = set(clip_dirs[i] for i in range(len(clip_dirs)) if i % 2 == 1)
    calib_df = df[df["clip_dir"].isin(calib_clips)]
    holdout_df = df[df["clip_dir"].isin(holdout_clips)]

    result = {}
    for field in FIELD_TEMPLATES:
        c_sub = calib_df[calib_df["field"] == field]
        h_sub = holdout_df[holdout_df["field"] == field]
        if len(c_sub) < 4 or len(h_sub) < 4:
            print(f"{field}: 표본 부족, 건너뜀")
            continue
        thr, calib_f1 = best_f1_threshold(c_sub["score"].to_numpy(), c_sub["label"].to_numpy())
        h_metrics = evaluate(h_sub["score"].to_numpy(), h_sub["label"].to_numpy(), thr)
        result[field] = {"threshold": float(thr), "calib_f1": float(calib_f1), **h_metrics}
        print(f"{field:15s} thr={thr:.4f}  calib F1={calib_f1*100:.1f}%  "
              f"holdout acc={h_metrics['accuracy']*100:.1f}% prec={h_metrics['precision']*100:.1f}% recall={h_metrics['recall']*100:.1f}%")

    with open("outputs/road_fields_threshold_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")
    print(f"결과 저장: {OUTPUT_CSV}, outputs/road_fields_threshold_result.json")


if __name__ == "__main__":
    main()
