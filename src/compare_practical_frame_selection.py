"""
6단계: 오답 캡션 없이 좋은 프레임을 고르는 실전 방법 비교
============================================================

[왜 이 실험이 필요한가]
5단계에서 찾은 "프레임 단위 margin pooling"(방식 B)은 사실 실전에서 쓸 수
없는 방법이다. margin(f) = 정답 점수 - 오답 점수 를 프레임마다 계산해서
가장 큰 프레임을 고르는데, 이건 "오답 캡션이 뭔지 미리 알고 있어야"
계산할 수 있다. 그런데 실제 검수 파이프라인에서는 VLM이 만든 캡션 1개
(맞을 수도 틀릴 수도 있는 "검수 대상")만 있을 뿐, 그 캡션이 정확히
어떻게 틀렸을지 미리 알 수 없다 - 오답 캡션은 우리가 실험을 설계하려고
인위적으로 만든 것이지, 실전 입력이 아니다. 즉 방식 B는 "이 프레임
선택 전략이 이론적으로 얼마나 좋아질 수 있는가"를 보여주는 상한선
(upper bound)이지, 그대로 배포할 수 있는 방법이 아니다.

그래서 이번 6단계에서는 "오답 캡션 없이" 좋은 프레임을 고르는 3가지
실전 가능한 방법을 시도하고, 방식 B(오답을 알고 고른 이론적 최선)에
얼마나 가까이 다가가는지 비교한다.

[방법 1] 선명도(sharpness) 기반 필터링
    캡션 내용과 무관하게, "이미지 자체가 얼마나 선명한가"만 본다.
    people_dancing의 70% 프레임처럼 아웃포커스로 흐릿한 프레임은 애초에
    정보량이 적어서 문제를 일으켰다. 이런 프레임을 캡션을 보기도 전에
    걸러낼 수 있다면 좋다.
    측정 방법: "라플라시안 분산(variance of Laplacian)" - 고전적인 블러
    탐지 기법이다. 이미지에 라플라시안 필터(각 픽셀을 주변 픽셀과 비교해
    "얼마나 급격하게 밝기가 변하는지" = 엣지/경계를 강조하는 필터)를
    적용한 뒤, 그 결과의 분산(값이 얼마나 들쭉날쭉한지)을 구한다.
    - 사진이 선명하면: 경계(edge)가 뚜렷해서 라플라시안 값이 크게
      들쭉날쭉함 -> 분산이 큼
    - 사진이 흐릿하면: 경계가 뭉개져서 라플라시안 값이 대체로 밋밋함
      -> 분산이 작음
    5장 중 이 분산이 가장 큰(=가장 선명한) 프레임 1장을 대표로 뽑는다.

[방법 2] 검수 대상 캡션만으로 최댓값 프레임 선택
    "오답을 모르니 오답 점수는 아예 안 본다." 검수하려는 캡션(여기서는
    실험을 위해 correct_caption을 검수 대상이라고 가정) 하나만 5개
    프레임에 채점해보고, 그 캡션과 가장 잘 맞는 프레임(점수가 가장 높은
    프레임)을 대표로 선택한다. 그 다음 그 프레임에서 오답들 점수도 같이
    읽어서 margin을 계산한다 (margin은 우리가 실험을 평가하려고 계산하는
    것일 뿐, 실제 프레임 "선택" 과정에는 오답이 전혀 관여하지 않는다).

[방법 3] 전체 프레임 평균(mean pooling)
    프레임을 아예 "선택"하지 않는다. 5개 프레임 전부에 대해 각 캡션의
    점수를 구하고 평균을 낸다. 가장 단순하고, 사전 정보가 전혀 필요 없다.

[비교 기준]
방식 B(5단계, 오답을 알고 고른 이론적 최선)를 상한선으로 두고, 위 3가지
방법의 margin이 그 상한선 대비 몇 %나 되는지, 그리고 단일 중간 프레임
방식보다는 나은지를 확인한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageOps

from frame_extract import extract_frames
from clip_score import compute_similarity_matrix
from blip_itm import compute_itm_score
from compare_multiframe import margin_independent_pool, margin_byframe_pool

ROOT = Path(__file__).resolve().parent.parent
CAPTIONS_JSON = ROOT / "data" / "captions.json"
VIDEO_DIR = ROOT / "data" / "videos"
OUTPUT_CSV = ROOT / "outputs" / "stage6_practical_frame_selection.csv"

N_FRAMES = 5
HIGHLIGHT_IDS = {"dog_park", "horse_herd", "people_dancing"}


def sharpness_score(img: Image.Image) -> float:
    """라플라시안 분산으로 이미지 선명도를 측정한다. 값이 클수록 선명함.

    ImageFilter.FIND_EDGES는 PIL에 내장된 3x3 커널로, 정확히 고전적인
    라플라시안 커널([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])과 같다 - 중앙
    픽셀과 주변 8픽셀의 차이를 강조해서 "경계"를 부각시킨다.
    """
    gray = ImageOps.grayscale(img)
    edges = gray.filter(ImageFilter.FIND_EDGES)
    return float(np.asarray(edges, dtype=np.float64).var())


def margin_at_frame(score_matrix: np.ndarray, frame_idx: int) -> float:
    """특정 프레임 인덱스에서의 margin(정답 - 오답평균)을 계산한다."""
    row = score_matrix[frame_idx]
    return float(row[0] - row[1:].mean())


def margin_mean_pool(score_matrix: np.ndarray) -> float:
    """방법 3: 프레임 축으로 전체 평균을 낸 뒤 margin 계산."""
    pooled = score_matrix.mean(axis=0)
    return float(pooled[0] - pooled[1:].mean())


def main():
    with open(CAPTIONS_JSON, "r", encoding="utf-8") as f:
        samples = json.load(f)["samples"]

    rows = []
    for s in samples:
        sid = s["id"]
        video_path = VIDEO_DIR / s["video_file"]
        wrong_texts = [s["wrong_caption"], s["wrong_caption_action"], s["wrong_caption_attribute"]]
        all_texts = [s["correct_caption"]] + wrong_texts

        print(f"[{sid}] {N_FRAMES}장 프레임 추출 및 채점 중...")
        multi_frames = extract_frames(video_path, n_frames=N_FRAMES)

        # BLIP-ITM 점수 행렬 (N_FRAMES, 4) - 이번 실험의 주 대상
        blip_matrix = np.array([
            [compute_itm_score(f, t) for t in all_texts] for f in multi_frames
        ])

        # --- 방식 B (5단계, 오답을 알고 고른 이론적 최선) : 비교 기준선 ---
        margin_oracle = margin_byframe_pool(blip_matrix)

        # --- 방법 1: 선명도 기반 프레임 선택 (캡션을 아예 안 봄) ---
        sharpness_scores = [sharpness_score(f) for f in multi_frames]
        sharpest_idx = int(np.argmax(sharpness_scores))
        margin_sharpness = margin_at_frame(blip_matrix, sharpest_idx)

        # --- 방법 2: 검수 대상(정답) 캡션 점수만으로 프레임 선택 ---
        correct_scores_per_frame = blip_matrix[:, 0]
        best_by_caption_idx = int(np.argmax(correct_scores_per_frame))
        margin_caption_only = margin_at_frame(blip_matrix, best_by_caption_idx)

        # --- 방법 3: 전체 프레임 평균 ---
        margin_mean = margin_mean_pool(blip_matrix)

        # --- 참고용: 단일 중간 프레임 (기존 baseline) ---
        mid_idx = N_FRAMES // 2  # extract_frames는 10~90%를 균등 분할하므로 가운데 인덱스가 50% 지점
        margin_single = margin_at_frame(blip_matrix, mid_idx)

        rows.append({
            "id": sid,
            "single(중간1장)": round(margin_single, 4),
            "방법1_선명도": round(margin_sharpness, 4),
            "방법2_정답점수만": round(margin_caption_only, 4),
            "방법3_평균pooling": round(margin_mean, 4),
            "방식B_오답활용(이론적최선)": round(margin_oracle, 4),
            "선명한 프레임 idx": sharpest_idx,
            "정답점수 최고 idx": best_by_caption_idx,
        })
        mark = "  <-- 관심 샘플" if sid in HIGHLIGHT_IDS else ""
        print(f"  single={margin_single:+.4f}  sharp={margin_sharpness:+.4f}(f{sharpest_idx})  "
              f"caption_only={margin_caption_only:+.4f}(f{best_by_caption_idx})  "
              f"mean={margin_mean:+.4f}  oracle_B={margin_oracle:+.4f}{mark}")

    df = pd.DataFrame(rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 120)
    print("최종 비교표 (BLIP-ITM margin)")
    print("=" * 120)
    print(df.to_string(index=False))

    print("\n" + "=" * 120)
    print("평균 및 오라클(방식B) 대비 도달률")
    print("=" * 120)
    for col in ["single(중간1장)", "방법1_선명도", "방법2_정답점수만", "방법3_평균pooling", "방식B_오답활용(이론적최선)"]:
        mean_val = df[col].mean()
        pct_of_oracle = mean_val / df["방식B_오답활용(이론적최선)"].mean() * 100
        print(f"{col:28s}: 평균 margin = {mean_val:.4f}   (오라클 대비 {pct_of_oracle:5.1f}%)")

    print(f"\n결과 CSV 저장 위치: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
