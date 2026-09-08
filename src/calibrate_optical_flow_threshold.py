"""
옵티컬 플로우 기반 "정지" 판별 신호 - calibration/holdout 검증
====================================================================

experiment_optical_flow_motion.py의 초기 실험(10개 표본, 정지/이동 완전
분리)이 유망해서, 정식으로 calibration/holdout 분리 검증한다.

배경(프레임 상단 35%) 영역의 옵티컬 플로우 평균 크기가 작을수록 "정지"
(카메라가 안 움직임), 클수록 "이동"(카메라가 움직임)이라는 가설을
F1-최대화 임계값으로 검증한다. motion="stop"을 양성(1), 그 외 전부
(moving straight/turning left/turning right/curved lane driving)를
음성(0)으로 라벨링한다 - 이번 세션 감사에서 발견한 오류가 "정지"
관련이었으므로 우선 이 이진 구분에 집중한다.
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
from experiment_optical_flow_motion import background_flow_magnitude

FULL_DATASET_JSON = Path("outputs/search_full_dataset.json")
OUTPUT_CSV = Path("outputs/optical_flow_calibration.csv")
N_PER_CLASS = 100
SEED = 7


def best_f1_threshold_low_is_positive(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """값이 "작을수록" 양성(정지)이라고 예측하는 방향의 F1-최대화 임계값
    (score <= thr 이면 양성으로 예측)."""
    order = np.argsort(scores)
    sorted_scores = scores[order]
    candidates = [sorted_scores[0] - 1.0]
    candidates += [(sorted_scores[i] + sorted_scores[i + 1]) / 2 for i in range(len(sorted_scores) - 1)]
    candidates.append(sorted_scores[-1] + 1.0)

    best_thr, best_f1 = candidates[0], -1.0
    for thr in candidates:
        pred = scores <= thr
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
    pred = scores <= thr
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

    stop_clips = [c for c in all_clips if c["motion"] == "stop"]
    other_clips = [c for c in all_clips if c["motion"] != "stop"]

    random.seed(SEED)
    sample_stop = random.sample(stop_clips, min(N_PER_CLASS, len(stop_clips)))
    sample_other = random.sample(other_clips, min(N_PER_CLASS, len(other_clips)))
    sample = [(c, 1) for c in sample_stop] + [(c, 0) for c in sample_other]
    random.shuffle(sample)

    rows = []
    t0 = time.time()
    for i, (c, label) in enumerate(sample):
        video_path = Path(c["clip_dir"]) / "1_clip" / "5.mp4"
        if not video_path.exists():
            continue
        mag = background_flow_magnitude(str(video_path))
        rows.append({"clip_dir": c["clip_dir"], "motion": c["motion"], "label": label,
                      "time_of_day": c["time_of_day"], "flow_magnitude": mag})
        if (i + 1) % 15 == 0:
            print(f"  진행: {i+1}/{len(sample)}  (경과 {(time.time()-t0)/60:.1f}분)")

    df = pd.DataFrame(rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    # 클립 단위 짝/홀 calibration/holdout 분리
    clip_dirs = sorted(df["clip_dir"].unique())
    calib_set = set(clip_dirs[i] for i in range(len(clip_dirs)) if i % 2 == 0)
    calib_df = df[df["clip_dir"].isin(calib_set)]
    holdout_df = df[~df["clip_dir"].isin(calib_set)]

    print(f"\ncalibration: {len(calib_df)}개 (정지 {calib_df['label'].sum()}개)  "
          f"holdout: {len(holdout_df)}개 (정지 {holdout_df['label'].sum()}개)")

    thr, calib_f1 = best_f1_threshold_low_is_positive(calib_df["flow_magnitude"].to_numpy(), calib_df["label"].to_numpy())
    hold_metrics = evaluate(holdout_df["flow_magnitude"].to_numpy(), holdout_df["label"].to_numpy(), thr)

    print(f"\n임계값(이하=정지로 예측): {thr:.4f}")
    print(f"calibration F1: {calib_f1*100:.1f}%")
    print(f"holdout: accuracy={hold_metrics['accuracy']*100:.1f}%  precision={hold_metrics['precision']*100:.1f}%  recall={hold_metrics['recall']*100:.1f}%")

    result = {"threshold": float(thr), "calib_f1": float(calib_f1), **hold_metrics}
    with open("outputs/optical_flow_threshold_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")
    print(f"결과 저장: {OUTPUT_CSV}, outputs/optical_flow_threshold_result.json")


if __name__ == "__main__":
    main()
