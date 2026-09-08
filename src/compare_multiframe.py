"""
5단계 파이프라인: 중간 프레임 1장 vs 다중 프레임(N장) Pooling (두 가지 방식 비교)
====================================================================================

지금까지 발견한 문제를 정리하면:
  - dog_park, horse_herd 두 샘플에서 "행동이 바뀐 오답"이 정답보다
    높은 점수를 받는 역전 현상이 있었다 -> 원인은 "중간 프레임이 하필
    애매한 순간이었다"는 프레임 선택 문제였다.
  - 이를 고치려고 "캡션마다 N장 중 최댓값을 독립적으로 취하는" 방식
    (독립 max pooling)을 처음 적용했더니 dog_park/horse_herd는 고쳐졌지만,
    엉뚱하게 people_dancing 샘플이 나빠지는 부작용이 나타났다.
  - 원인을 파보니: 독립 max pooling은 "정답의 최댓값이 나온 프레임"과
    "오답의 최댓값이 나온 프레임"이 서로 다른 프레임이어도 그냥 비교해
    버린다. people_dancing의 70% 프레임은 아웃포커스된 머리가 화면을
    가려 정보량이 낮은 프레임이었는데, 하필 거기서 오답(속성 변경)이
    우연히 튀는 점수를 받았고, 그게 정답의 (다른 프레임에서 나온) 최댓값과
    짝지어져 margin을 깎아먹었다. 정작 70% 프레임 "안에서는" 정답이
    오답보다 여전히 훨씬 높았는데도 말이다.

그래서 이번에는 pooling 방식을 두 가지로 나눠서 비교한다:

  [방식 A] 독립 max pooling (기존, "캡션별 최댓값")
      correct_final = max_f( score(f, 정답) )
      wrong_final   = max_f( score(f, 오답) )   <- 정답과 다른 프레임이어도 무관
      margin = correct_final - mean(wrong_final들)

  [방식 B] 프레임 단위 margin 기반 pooling (신규, "같은 프레임 안에서 비교")
      먼저 각 프레임 f마다 "그 프레임 하나만 놓고" margin을 계산한다:
          margin(f) = score(f, 정답) - mean(score(f, 오답들))
      그 다음 margin(f)이 가장 큰 프레임을 "가장 확신 있게 정답과 오답을
      가르는 프레임"으로 보고, 그 프레임의 margin을 최종 값으로 쓴다:
          margin_final = max_f( margin(f) )
      이 방식은 항상 "같은 프레임에서 나온 정답 점수와 오답 점수"끼리만
      비교하므로, 서로 다른 프레임의 최댓값을 섞어 비교하는 방식 A의
      함정을 원천적으로 피한다.

두 방식 모두 CLIP(원본 코사인)과 BLIP-ITM 양쪽에서 계산해서, "중간
프레임 1장" 기준선과 함께 나란히 비교한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from frame_extract import extract_frames, save_frame
from clip_score import compute_similarity_matrix
from blip_itm import compute_itm_score

ROOT = Path(__file__).resolve().parent.parent
CAPTIONS_JSON = ROOT / "data" / "captions.json"
VIDEO_DIR = ROOT / "data" / "videos"
FRAME_DIR = ROOT / "data" / "frames"
OUTPUT_CSV = ROOT / "outputs" / "stage5_multiframe_results.csv"

N_FRAMES = 5
HIGHLIGHT_IDS = {"dog_park", "horse_herd", "people_dancing"}


def margin_independent_pool(score_matrix: np.ndarray) -> float:
    """방식 A: 캡션(열)마다 프레임(행) 축에서 독립적으로 최댓값을 취한 뒤 margin 계산.
    score_matrix: shape (n_frames, 4) - 열 순서는 [정답, 오답1, 오답2, 오답3]
    """
    pooled = score_matrix.max(axis=0)  # (4,) - 캡션별 최댓값, 서로 다른 프레임에서 나올 수 있음
    return float(pooled[0] - pooled[1:].mean())


def margin_byframe_pool(score_matrix: np.ndarray) -> float:
    """방식 B: 프레임(행)마다 먼저 margin을 계산하고, 그중 최댓값을 취함.
    -> 정답과 오답 점수가 항상 "같은 프레임"에서 나온 값끼리 비교된다.
    """
    per_frame_margin = score_matrix[:, 0] - score_matrix[:, 1:].mean(axis=1)  # (n_frames,)
    return float(per_frame_margin.max())


def main():
    with open(CAPTIONS_JSON, "r", encoding="utf-8") as f:
        samples = json.load(f)["samples"]

    rows = []
    for s in samples:
        sid = s["id"]
        video_path = VIDEO_DIR / s["video_file"]
        wrong_texts = [s["wrong_caption"], s["wrong_caption_action"], s["wrong_caption_attribute"]]
        all_texts = [s["correct_caption"]] + wrong_texts  # 열 순서: [정답, 오답1, 오답2, 오답3]

        # --- 기존 "중간 프레임 1장" ---
        single_frame_path = FRAME_DIR / f"{sid}.jpg"
        if not single_frame_path.exists():
            save_frame(video_path, single_frame_path)

        # --- "N장 균등 샘플링" ---
        print(f"[{sid}] {N_FRAMES}장 프레임 추출 중...")
        multi_frames = extract_frames(video_path, n_frames=N_FRAMES)

        # === CLIP: (n_frames, 4) 유사도 행렬을 한 번에 계산 ===
        sim_single = compute_similarity_matrix([single_frame_path], all_texts)  # (1, 4)
        clip_single_margin = margin_independent_pool(sim_single)  # 프레임이 1개뿐이라 두 방식 결과 동일

        sim_multi = compute_similarity_matrix(multi_frames, all_texts)  # (N, 4)
        clip_indep_margin = margin_independent_pool(sim_multi)
        clip_byframe_margin = margin_byframe_pool(sim_multi)

        # === BLIP-ITM: (n_frames, 4) 점수 행렬을 직접 forward pass로 채움 ===
        blip_single_matrix = np.array([[compute_itm_score(single_frame_path, t) for t in all_texts]])  # (1,4)
        blip_single_margin = margin_independent_pool(blip_single_matrix)

        blip_multi_matrix = np.array([
            [compute_itm_score(f, t) for t in all_texts] for f in multi_frames
        ])  # (N, 4)
        blip_indep_margin = margin_independent_pool(blip_multi_matrix)
        blip_byframe_margin = margin_byframe_pool(blip_multi_matrix)

        rows.append({
            "id": sid,
            "clip_single": round(clip_single_margin, 4),
            "clip_indep": round(clip_indep_margin, 4),
            "clip_byframe": round(clip_byframe_margin, 4),
            "blip_single": round(blip_single_margin, 4),
            "blip_indep": round(blip_indep_margin, 4),
            "blip_byframe": round(blip_byframe_margin, 4),
        })
        mark = "  <-- 관심 샘플" if sid in HIGHLIGHT_IDS else ""
        print(f"  clip: single={clip_single_margin:+.4f} indep={clip_indep_margin:+.4f} byframe={clip_byframe_margin:+.4f}   "
              f"blip: single={blip_single_margin:+.4f} indep={blip_indep_margin:+.4f} byframe={blip_byframe_margin:+.4f}{mark}")

    df = pd.DataFrame(rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 110)
    print(f"최종 비교: 중간 프레임 1장 vs {N_FRAMES}장(독립 max pooling) vs {N_FRAMES}장(프레임 단위 margin pooling)")
    print("=" * 110)
    print(df.to_string(index=False))

    print("\n" + "-" * 110)
    print("관심 샘플")
    print("-" * 110)
    for hid in HIGHLIGHT_IDS:
        row = df[df["id"] == hid].iloc[0]
        print(f"{hid}:")
        print(f"  CLIP  single={row['clip_single']:+.4f}  indep={row['clip_indep']:+.4f}  byframe={row['clip_byframe']:+.4f}")
        print(f"  BLIP  single={row['blip_single']:+.4f}  indep={row['blip_indep']:+.4f}  byframe={row['blip_byframe']:+.4f}")

    print(f"\n평균 margin (CLIP) : single={df['clip_single'].mean():.4f}  indep={df['clip_indep'].mean():.4f}  byframe={df['clip_byframe'].mean():.4f}")
    print(f"평균 margin (BLIP) : single={df['blip_single'].mean():.4f}  indep={df['blip_indep'].mean():.4f}  byframe={df['blip_byframe'].mean():.4f}")

    n_indep_worse = (df["blip_indep"] < df["blip_single"]).sum()
    n_byframe_worse = (df["blip_byframe"] < df["blip_single"]).sum()
    print(f"\n[BLIP] 단일 프레임보다 나빠진 샘플 수: 독립 max pooling={n_indep_worse}개, 프레임단위 margin pooling={n_byframe_worse}개")

    print(f"\n결과 CSV 저장 위치: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
