"""
객체/행동 단위 임계값 재보정 - 4~5단계: 분포 비교 + 새 임계값 계산 + 비교
=============================================================================

score_component_dataset.py로 채점한 124개 (영상, 단어, 진짜/가짜, 점수)를
가지고:
  1) 진짜 그룹과 가짜 그룹의 점수 분포(평균/최소/최대 + 히스토그램)를 비교한다.
  2) 두 분포를 가장 잘 가르는 임계값을 다시 계산한다 (calibrate_thresholds.py와
     동일한 "정확도 최대화 전수탐색" 방식).
  3) 기존 임계값(SUSPECT<0.05, BORDERLINE<0.15)과 새 임계값을 각각 적용했을 때
     정밀도(precision)/재현율(recall)이 어떻게 달라지는지 비교한다.
     - 여기서 "양성(positive)" = 가짜(FALSE, 즉 SUSPECT로 잡아내고 싶은 대상)
     - Precision = SUSPECT로 찍힌 것 중 실제로 가짜인 비율
                   (낮으면 = 진짜인데 SUSPECT로 잘못 찍는 오탐이 많다는 뜻 =
                    11단계에서 겪은 "정답이 REVIEW로 잘못 떨어지는" 문제)
     - Recall = 실제 가짜 중 SUSPECT로 잡아낸 비율
                (낮으면 = 가짜를 놓친다는 뜻)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SCORED_CSV = ROOT / "outputs" / "component_dataset_scored.csv"
FIG_PATH = ROOT / "results" / "images" / "fig5_component_threshold_histogram.png"

# 기존(11단계에서 2개 샘플만 보고 정한) 임계값
OLD_SUSPECT = 0.05
OLD_BORDERLINE = 0.15


def best_single_threshold(true_scores: np.ndarray, false_scores: np.ndarray) -> tuple[float, float]:
    """가짜(FALSE)를 SUSPECT(<t)로, 진짜(TRUE)를 OK(>=t)로 분류하는 정확도를
    최대화하는 임계값을 전수탐색으로 찾는다. (calibrate_thresholds.py와 동일한 방식)
    """
    candidates = np.unique(np.concatenate([true_scores, false_scores]))
    midpoints = (candidates[:-1] + candidates[1:]) / 2
    best_t, best_acc = 0.1, -1
    for t in midpoints:
        # t 미만이면 SUSPECT(가짜로 판정), t 이상이면 OK(진짜로 판정)
        acc = ((false_scores < t).sum() + (true_scores >= t).sum()) / (len(true_scores) + len(false_scores))
        if acc > best_acc:
            best_acc = acc
            best_t = t
    return float(best_t), float(best_acc)


def precision_recall(true_scores: np.ndarray, false_scores: np.ndarray, threshold: float) -> tuple[float, float, dict]:
    """threshold 미만이면 SUSPECT(=가짜로 예측)로 분류했을 때의 precision/recall.
    양성 클래스 = FALSE(가짜).
    """
    tp = int((false_scores < threshold).sum())   # 가짜를 가짜로 맞춤
    fn = int((false_scores >= threshold).sum())  # 가짜를 놓침 (진짜라고 오판)
    fp = int((true_scores < threshold).sum())    # 진짜를 가짜로 오판 (11단계 문제)
    tn = int((true_scores >= threshold).sum())   # 진짜를 진짜로 맞춤

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    return precision, recall, {"TP": tp, "FN": fn, "FP": fp, "TN": tn}


def plot_histogram(true_scores, false_scores, old_t, new_t, out_path: Path):
    noto_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    fm.fontManager.addfont(noto_path)
    font_name = fm.FontProperties(fname=noto_path).get_name()
    plt.rcParams.update({"font.family": font_name, "axes.unicode_minus": False, "font.size": 11})

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bins = np.linspace(0, 1, 26)
    ax.hist(false_scores, bins=bins, alpha=0.6, label=f"가짜(FALSE) 객체/행동 (n={len(false_scores)})", color="#C0504D")
    ax.hist(true_scores, bins=bins, alpha=0.6, label=f"진짜(TRUE) 객체/행동 (n={len(true_scores)})", color="#4E9A6A")
    ax.axvline(old_t, color="gray", linestyle="--", linewidth=1.5, label=f"기존 임계값 ({old_t:.2f})")
    ax.axvline(new_t, color="black", linestyle="-", linewidth=1.5, label=f"새 임계값 ({new_t:.3f})")
    ax.set_xlabel("객체/행동 단위 점수 (BLIP-ITM)")
    ax.set_ylabel("개수")
    ax.set_title("객체/행동 단위 점수 분포: 진짜 vs 가짜 (124개 표본)", fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    df = pd.read_csv(SCORED_CSV)
    true_scores = df[df["is_true"]]["score"].values
    false_scores = df[~df["is_true"]]["score"].values

    print("=" * 90)
    print("1) 분포 비교")
    print("=" * 90)
    print(f"진짜(TRUE)  n={len(true_scores):3d}  min={true_scores.min():.4f}  max={true_scores.max():.4f}  "
          f"mean={true_scores.mean():.4f}  median={np.median(true_scores):.4f}")
    print(f"가짜(FALSE) n={len(false_scores):3d}  min={false_scores.min():.4f}  max={false_scores.max():.4f}  "
          f"mean={false_scores.mean():.4f}  median={np.median(false_scores):.4f}")

    # object/action 타입별로도 따로 확인
    for t in ["object", "action"]:
        sub = df[df["type"] == t]
        ts = sub[sub["is_true"]]["score"].values
        fs = sub[~sub["is_true"]]["score"].values
        print(f"\n  [{t}] 진짜 n={len(ts)} mean={ts.mean():.4f}  |  가짜 n={len(fs)} mean={fs.mean():.4f}")

    print("\n" + "=" * 90)
    print("2) 새 임계값 계산")
    print("=" * 90)
    new_t, acc = best_single_threshold(true_scores, false_scores)
    print(f"정확도 최대화 임계값: {new_t:.4f}  (정확도 {acc*100:.1f}%, 전체 {len(true_scores)+len(false_scores)}개 중)")

    print("\n" + "=" * 90)
    print("3) 기존 임계값 vs 새 임계값 - Precision/Recall 비교")
    print("=" * 90)
    for label, t in [("기존 (0.05)", OLD_SUSPECT), (f"새 임계값 ({new_t:.4f})", new_t)]:
        p, r, counts = precision_recall(true_scores, false_scores, t)
        print(f"\n{label}:")
        print(f"  Precision = {p*100:5.1f}%   Recall = {r*100:5.1f}%")
        print(f"  TP(가짜를 잡음)={counts['TP']}  FP(진짜를 오판)={counts['FP']}  "
              f"FN(가짜를 놓침)={counts['FN']}  TN(진짜를 맞춤)={counts['TN']}")

    plot_histogram(true_scores, false_scores, OLD_SUSPECT, new_t, FIG_PATH)
    print(f"\n히스토그램 저장 위치: {FIG_PATH}")

    return new_t


if __name__ == "__main__":
    main()
