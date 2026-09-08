"""
옵티컬 플로우 방향 신호로 좌회전/우회전 판별 - calibration/holdout 검증
============================================================================

experiment_optical_flow_motion.py의 "정지" 성공(holdout 83.3%)을 방향
정보로 확장할 수 있는지 확인한다. 배경 영역의 평균 "수평" 흐름 성분의
부호/크기로 좌회전(카메라가 왼쪽으로 회전 -> 배경이 오른쪽으로 흐름,
양수 예상)과 우회전(반대, 음수 예상)을 구분한다.

주의: turning_left/turning_right는 회사 mount 전체(740클립)에도 각각
8개/15개뿐이라 - 검색 컷오프 때 겪었던 것과 동일한 데이터 희소성 문제가
여기도 그대로 있다. calibration/holdout으로 나누면 한쪽이 4~8개짜리
아주 작은 표본이 되므로, 여기서 나오는 holdout 수치는 "정지" 때(60클립)
보다 훨씬 불안정할 수 있다는 걸 감안해야 한다.

각 방향을 "moving straight"(가장 깨끗한 대조군, 수평 흐름이 0 근처일
것으로 예상) 대비 이진 분류로 검증한다.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))
from frame_extract import extract_frames

FULL_DATASET_JSON = Path("outputs/search_full_dataset.json")
OUTPUT_CSV = Path("outputs/turn_direction_calibration.csv")
SEED = 11


def mean_horizontal_flow(video_path: str, n_frames: int = 5, top_fraction: float = 0.35) -> float:
    frames = extract_frames(video_path, n_frames=n_frames)
    grays = [np.asarray(f.convert("L").resize((160, 160))) for f in frames]
    h = int(160 * top_fraction)
    xs = []
    for i in range(len(grays) - 1):
        flow = cv2.calcOpticalFlowFarneback(grays[i], grays[i + 1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        xs.append(flow[:h, :, 0].mean())
    return float(np.mean(xs))


def best_threshold(scores: np.ndarray, labels: np.ndarray, direction: str) -> tuple[float, float]:
    """direction='low'면 작을수록(더 음수) 양성, 'high'면 클수록(더 양수) 양성으로 예측."""
    order = np.argsort(scores)
    sorted_scores = scores[order]
    candidates = [sorted_scores[0] - 1.0]
    candidates += [(sorted_scores[i] + sorted_scores[i + 1]) / 2 for i in range(len(sorted_scores) - 1)]
    candidates.append(sorted_scores[-1] + 1.0)

    best_thr, best_f1 = candidates[0], -1.0
    for thr in candidates:
        pred = (scores <= thr) if direction == "low" else (scores >= thr)
        tp = ((pred == 1) & (labels == 1)).sum()
        fp = ((pred == 1) & (labels == 0)).sum()
        fn = ((pred == 0) & (labels == 1)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_thr, best_f1


def evaluate(scores: np.ndarray, labels: np.ndarray, thr: float, direction: str) -> dict:
    pred = (scores <= thr) if direction == "low" else (scores >= thr)
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

    by_motion = {}
    for c in all_clips:
        by_motion.setdefault(c["motion"], []).append(c)

    random.seed(SEED)
    straight_sample = random.sample(by_motion["moving straight"], 20)

    rows = []
    t0 = time.time()
    all_needed = by_motion["turning left"] + by_motion["turning right"] + straight_sample
    for i, c in enumerate(all_needed):
        video_path = Path(c["clip_dir"]) / "1_clip" / "5.mp4"
        if not video_path.exists():
            continue
        x = mean_horizontal_flow(str(video_path))
        rows.append({"clip_dir": c["clip_dir"], "motion": c["motion"], "flow_x": x})
        if (i + 1) % 10 == 0:
            print(f"  진행: {i+1}/{len(all_needed)}  (경과 {(time.time()-t0)/60:.1f}분)")

    df = pd.DataFrame(rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    results = {}
    for turn, direction in [("turning right", "low"), ("turning left", "high")]:
        sub = df[df["motion"].isin([turn, "moving straight"])].copy()
        sub["label"] = (sub["motion"] == turn).astype(int)

        clip_dirs = sorted(sub["clip_dir"].unique())
        calib_set = set(clip_dirs[i] for i in range(len(clip_dirs)) if i % 2 == 0)
        calib_df = sub[sub["clip_dir"].isin(calib_set)]
        holdout_df = sub[~sub["clip_dir"].isin(calib_set)]

        thr, calib_f1 = best_threshold(calib_df["flow_x"].to_numpy(), calib_df["label"].to_numpy(), direction)
        hold_metrics = evaluate(holdout_df["flow_x"].to_numpy(), holdout_df["label"].to_numpy(), thr, direction)

        n_pos_calib = int(calib_df["label"].sum())
        n_pos_hold = int(holdout_df["label"].sum())
        print(f"\n[{turn}] calib={len(calib_df)}개(양성{n_pos_calib}) holdout={len(holdout_df)}개(양성{n_pos_hold})")
        print(f"  임계값={thr:.4f}({direction})  calib F1={calib_f1*100:.1f}%")
        print(f"  holdout: accuracy={hold_metrics['accuracy']*100:.1f}% precision={hold_metrics['precision']*100:.1f}% recall={hold_metrics['recall']*100:.1f}%")

        results[turn] = {"threshold": float(thr), "direction": direction, "calib_f1": float(calib_f1),
                          "n_pos_calib": n_pos_calib, "n_pos_holdout": n_pos_hold, **hold_metrics}

    with open("outputs/turn_direction_threshold_result.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")
    print(f"결과 저장: {OUTPUT_CSV}, outputs/turn_direction_threshold_result.json")


if __name__ == "__main__":
    main()
