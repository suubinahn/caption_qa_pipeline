"""
2단계 파이프라인: CSLS로 코사인 유사도 보정하기
================================================

1단계(compare_captions.py)에서는 "이미지 1장 vs 정답 캡션 1개",
"이미지 1장 vs 오답 캡션 1개"를 각각 독립적으로 비교했다. 이 방식의
한계는, 코사인 유사도만 보면 "이 텍스트가 원래 여러 이미지와 두루
비슷하게 나오는 허브인지" 알 수 없다는 것이다 (허브 문제는 비교 대상이
여러 개 모여서 "행렬"을 이뤄야 드러난다).

그래서 2단계에서는:
  1) 8개 이미지 x (정답 8개 + 오답 24개 = 32개 텍스트, 중복 제거 후엔 조금
     더 적음) 로 이루어진 "이미지 x 텍스트" 유사도 행렬을 통째로 만든다.
     오답은 샘플당 3개: (a) 1단계부터 쓰던, 다른 샘플의 정답을 재활용한
     오답, (b) 행동만 바꾼 오답, (c) 객체/속성만 바꾼 오답.
  2) 이 행렬에 CSLS 보정을 적용한다.
  3) 각 이미지에 대해 "정답 캡션 점수 - 오답 캡션들 평균 점수" 를
     원본 코사인 유사도 기준(margin_raw)과 CSLS 보정 기준(margin_csls)
     으로 각각 계산해서 나란히 비교한다.

오답 캡션을 늘린 이유: CSLS의 r_T(x), r_S(y)는 "이웃 K개의 평균"으로
계산되는데, 후보군(텍스트 개수)이 8개뿐이면 K를 크게 잡을 수 없어
통계적으로 불안정하다. 텍스트 축을 24개 안팎으로 늘리면 r_T(x)
(이미지 입장에서 보는 텍스트 이웃 평균)는 훨씬 안정적으로 계산된다.
다만 이미지는 여전히 8개뿐이라 r_S(y)(텍스트 입장에서 보는 이미지
이웃 평균)의 K는 크게 못 키운다 - 이 한계는 실행 결과 뒤에서 다시 짚는다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from frame_extract import save_frame
from clip_score import compute_similarity_matrix
from csls import csls

ROOT = Path(__file__).resolve().parent.parent
CAPTIONS_JSON = ROOT / "data" / "captions.json"
VIDEO_DIR = ROOT / "data" / "videos"
FRAME_DIR = ROOT / "data" / "frames"
OUTPUT_CSV = ROOT / "outputs" / "stage2_csls_results.csv"

# CSLS의 이웃 개수(K). 원 논문 기본값은 10이지만, 우리 데이터는 이미지가
# 8개뿐이라 텍스트->이미지 방향 이웃 평균(r_S)의 K는 8을 넘길 수 없다.
# 두 방향 모두 같은 K를 쓰는 게 CSLS의 표준 정의라서, 여기서는 두 축
# 크기(8, 32) 중 작은 쪽을 넘지 않는 K=5로 설정했다.
K_NEIGHBORS = 5


def load_samples():
    with open(CAPTIONS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["samples"]


def build_text_pool(samples):
    """모든 샘플의 정답/오답 캡션을 모아 '중복 없는 텍스트 후보 풀'을 만든다.

    왜 중복 제거가 필요한가?
    1단계에서 각 샘플의 wrong_caption은 사실 "다른 샘플의 correct_caption을
    그대로 재활용"한 것이었다(예: cat_string_toy의 오답 = horse_herd의 정답).
    그래서 그냥 다 이어붙이면 같은 문장이 여러 번 들어가 버린다. 텍스트
    풀에 같은 문장이 중복되면 유사도 행렬에서 같은 열(column)이 여러 개
    생기는 셈이라 CSLS의 이웃 평균 계산이 왜곡된다. 그래서 문자열 기준으로
    중복을 제거하고, "이 문장이 텍스트 풀의 몇 번째 인덱스인지"만 기억해서
    각 샘플이 그 인덱스를 가리키게 한다.

    반환값:
      text_pool: 중복 없는 캡션 문자열 리스트
      text_to_idx: {문장: 인덱스} 딕셔너리 (조회용)
    """
    text_pool: list[str] = []
    text_to_idx: dict[str, int] = {}

    def add(text: str) -> int:
        if text not in text_to_idx:
            text_to_idx[text] = len(text_pool)
            text_pool.append(text)
        return text_to_idx[text]

    for s in samples:
        add(s["correct_caption"])
        add(s["wrong_caption"])
        add(s["wrong_caption_action"])
        add(s["wrong_caption_attribute"])

    return text_pool, text_to_idx


def main():
    samples = load_samples()

    # --- 1) 이미지(프레임) 준비 ---
    frame_paths = []
    for s in samples:
        frame_path = FRAME_DIR / f"{s['id']}.jpg"
        if not frame_path.exists():
            save_frame(VIDEO_DIR / s["video_file"], frame_path)
        frame_paths.append(frame_path)

    # --- 2) 텍스트 후보 풀 구성 (중복 제거) ---
    text_pool, text_to_idx = build_text_pool(samples)
    n_images, n_texts = len(frame_paths), len(text_pool)
    print(f"이미지 {n_images}개 x 텍스트(중복 제거 후) {n_texts}개 유사도 행렬을 계산합니다...")

    # --- 3) 원본 코사인 유사도 행렬 계산 (배치 처리로 한 번에) ---
    sim_raw = compute_similarity_matrix(frame_paths, text_pool)  # shape: (n_images, n_texts)

    # --- 4) CSLS 보정 적용 ---
    sim_csls = csls(sim_raw, k=K_NEIGHBORS)

    # --- 5) 각 이미지(샘플)별로 "정답 점수 vs 오답들 평균 점수" margin 계산 ---
    rows = []
    for i, s in enumerate(samples):
        correct_idx = text_to_idx[s["correct_caption"]]
        wrong_texts = [s["wrong_caption"], s["wrong_caption_action"], s["wrong_caption_attribute"]]
        wrong_idxs = [text_to_idx[t] for t in wrong_texts]

        raw_correct = sim_raw[i, correct_idx]
        raw_wrong_mean = sim_raw[i, wrong_idxs].mean()
        margin_raw = raw_correct - raw_wrong_mean

        csls_correct = sim_csls[i, correct_idx]
        csls_wrong_mean = sim_csls[i, wrong_idxs].mean()
        margin_csls = csls_correct - csls_wrong_mean

        rows.append(
            {
                "id": s["id"],
                "raw_correct": round(raw_correct, 4),
                "raw_wrong_mean": round(raw_wrong_mean, 4),
                "margin_raw": round(margin_raw, 4),
                "csls_correct": round(csls_correct, 4),
                "csls_wrong_mean": round(csls_wrong_mean, 4),
                "margin_csls": round(margin_csls, 4),
                "margin 변화": round(margin_csls - margin_raw, 4),
            }
        )

    df = pd.DataFrame(rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 100)
    print(f"원본 코사인 margin vs CSLS 보정 margin 비교 (K={K_NEIGHBORS})")
    print("=" * 100)
    print(df[["id", "margin_raw", "margin_csls", "margin 변화"]].to_string(index=False))

    improved = (df["margin 변화"] > 0).sum()
    print(f"\n{len(df)}개 샘플 중 {improved}개에서 CSLS 적용 후 margin이 커졌음(=허브 보정 효과로 정답/오답 구분이 더 뚜렷해짐).")
    print(f"margin_raw 평균: {df['margin_raw'].mean():.4f}  ->  margin_csls 평균: {df['margin_csls'].mean():.4f}")

    bicycle_row = df[df["id"] == "bicycle_ride"]
    if not bicycle_row.empty:
        r = bicycle_row.iloc[0]
        print(f"\n[관심 샘플] bicycle_ride: margin_raw={r['margin_raw']:.4f} -> margin_csls={r['margin_csls']:.4f} (변화 {r['margin 변화']:+.4f})")

    print(f"\n결과 CSV 저장 위치: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
