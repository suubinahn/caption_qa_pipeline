"""
진단 스크립트: "대표 프레임 1장"이 진짜 원인인지 검증
======================================================

이 스크립트는 정식 파이프라인 단계가 아니라, horse_herd 샘플에서 발견한
이상 현상("달린다"보다 "가만히 서서 풀을 뜯는다"가 더 높은 점수를 받음)의
원인을 파헤치기 위한 일회성 진단 도구다.

가설: 원인이 "영상에서 중간 프레임 딱 1장만 뽑는 방식" 자체에 있다면,
      영상 안의 다른 시점 프레임들은 점수가 크게 다르게 나와야 한다
      (즉 우연히 "정지 동작처럼 보이는 프레임"을 뽑았을 뿐, 영상 전체를
      보면 "달리는 중"이라는 게 더 명확할 것이다).

검증 방법: 영상을 시간 축으로 5등분(10/30/50/70/90%)해서 프레임을 각각
뽑고, 그때마다 "정답(달린다)"과 "행동 오답(가만히 서서 풀을 뜯는다)"의
BLIP-ITM 점수를 계산해 프레임마다 어떻게 요동치는지 관찰한다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blip_itm import compute_itm_score

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "investigation_horse_herd"

FRACTIONS = [0.10, 0.30, 0.50, 0.70, 0.90]

CORRECT = "A small group of horses is running together across a green pasture near a tree line."
WRONG_ACTION = "A small group of horses is standing still and grazing quietly in a green pasture near a tree line."


def main():
    video_path = ROOT / "data" / "videos" / "horse_herd.webm"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"영상 전체 프레임을 디코딩하는 중... ({video_path.name})")
    frames = list(iio.imiter(str(video_path), plugin="pyav"))
    n_frames = len(frames)
    print(f"총 프레임 수: {n_frames}\n")

    rows = []
    for frac in FRACTIONS:
        idx = int(n_frames * frac)
        frame_img = Image.fromarray(np.asarray(frames[idx]))
        out_path = OUT_DIR / f"horse_herd_{int(frac*100):02d}pct.jpg"
        frame_img.save(out_path)

        score_correct = compute_itm_score(frame_img, CORRECT)
        score_wrong = compute_itm_score(frame_img, WRONG_ACTION)
        diff = score_correct - score_wrong

        rows.append((frac, idx, score_correct, score_wrong, diff))
        print(f"[{int(frac*100):3d}%] frame_idx={idx:4d}  정답(달린다)={score_correct:.4f}  "
              f"오답(풀뜯는다)={score_wrong:.4f}  차이={diff:+.4f}  -> {out_path.name}")

    print("\n" + "=" * 90)
    print("요약")
    print("=" * 90)
    correct_scores = [r[2] for r in rows]
    wrong_scores = [r[3] for r in rows]
    print(f"정답 점수 범위   : {min(correct_scores):.4f} ~ {max(correct_scores):.4f} (표준편차 {np.std(correct_scores):.4f})")
    print(f"오답 점수 범위   : {min(wrong_scores):.4f} ~ {max(wrong_scores):.4f} (표준편차 {np.std(wrong_scores):.4f})")
    n_correct_wins = sum(1 for r in rows if r[2] > r[3])
    print(f"\n5개 프레임 중 정답이 오답보다 높았던 경우: {n_correct_wins}/5")
    print(f"프레임 이미지 저장 위치: {OUT_DIR}")


if __name__ == "__main__":
    main()
