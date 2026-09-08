"""
3단계 분석: "행동 변경" vs "객체/속성 변경" 오답 중 CLIP이 어느 쪽을 더 못 구분하는가
====================================================================================

[배경]
2단계에서 오답을 여러 종류로 늘리는 과정에서 흥미로운 패턴을 발견했다:
"자전거를 탄다 -> 옆에 서있다"(행동 변경) 같은 오답은 "빨간 셔츠 -> 파란
셔츠"(속성 변경) 오답보다 CLIP이 구분하기 더 어려워 보였다. 이 스크립트는
그 관찰을 8개 샘플 전체에 대해 체계적으로 검증한다.

[가설]
CLIP은 이미지 안의 "정적인 요소"(누가/무엇이 있는지, 색깔, 개수 등)에는
민감하지만, "시간에 따라 변하는 동작"(무엇을 하고 있는지)에는 상대적으로
둔감할 것이다. 왜 이런 가설이 나오는가?
  - CLIP은 정지 이미지 1장과 문장을 비교한다. 정지 이미지는 동작의 결과나
    한 순간의 자세는 보여주지만, "달리는 중"과 "서있는 중"이 시각적으로
    아주 비슷한 자세로 보일 때가 많다(특히 모션 블러가 없는 프레임에서는
    더더욱). 반면 "고양이 vs 개", "빨강 vs 파랑"처럼 정적 속성은 픽셀
    수준에서 뚜렷한 차이를 만든다.
  - CLIP의 사전학습 데이터(인터넷의 이미지-alt텍스트 쌍)도 "이 장면에 뭐가
    있는지"를 나열하는 캡션이 많고, 정교한 동작 묘사가 상대적으로 적다는
    지적이 여러 CLIP 관련 연구에서 있어왔다.

[측정 방법]
data/captions.json의 각 샘플에는 다음 3가지 오답이 있다:
  - wrong_caption          : 다른 샘플의 정답을 재활용 (장면 전체가 다름, 비교 X)
  - wrong_caption_action   : 행동만 바꾸고 나머지는 동일 (예: 논다 -> 잔다)
  - wrong_caption_attribute: 객체/속성만 바꾸고 나머지는 동일 (예: 고양이 -> 개)

각 샘플, 각 오답 유형에 대해:
    margin_action    = 정답 점수 - wrong_caption_action 점수
    margin_attribute = 정답 점수 - wrong_caption_attribute 점수

margin이 작을수록 "정답과 오답을 구분하기 어렵다"는 뜻이다. 그래서
"margin_action의 평균이 margin_attribute의 평균보다 작다"는 결과가 나오면
가설(CLIP이 행동 변경에 더 취약하다)이 지지되는 것이다.

원본 코사인 유사도와 CSLS 보정 유사도 양쪽 모두에서 계산해서, 이 패턴이
CSLS 보정 여부와 상관없이 일관되게 나타나는지도 함께 확인한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from clip_score import compute_similarity_matrix
from csls import csls
from compare_captions_csls import build_text_pool, K_NEIGHBORS

ROOT = Path(__file__).resolve().parent.parent
CAPTIONS_JSON = ROOT / "data" / "captions.json"
FRAME_DIR = ROOT / "data" / "frames"
OUTPUT_CSV = ROOT / "outputs" / "stage3_action_vs_attribute.csv"


def main():
    with open(CAPTIONS_JSON, "r", encoding="utf-8") as f:
        samples = json.load(f)["samples"]

    frame_paths = [FRAME_DIR / f"{s['id']}.jpg" for s in samples]

    # 2단계와 동일한 방식으로 텍스트 풀(중복 제거)과 유사도 행렬을 만든다.
    # (compare_captions_csls.py의 build_text_pool을 그대로 재사용 -> 로직 중복 방지)
    text_pool, text_to_idx = build_text_pool(samples)
    sim_raw = compute_similarity_matrix(frame_paths, text_pool)
    sim_csls = csls(sim_raw, k=K_NEIGHBORS)

    rows = []
    for i, s in enumerate(samples):
        correct_idx = text_to_idx[s["correct_caption"]]
        action_idx = text_to_idx[s["wrong_caption_action"]]
        attribute_idx = text_to_idx[s["wrong_caption_attribute"]]

        rows.append({
            "id": s["id"],
            "margin_action_raw": round(sim_raw[i, correct_idx] - sim_raw[i, action_idx], 4),
            "margin_attribute_raw": round(sim_raw[i, correct_idx] - sim_raw[i, attribute_idx], 4),
            "margin_action_csls": round(sim_csls[i, correct_idx] - sim_csls[i, action_idx], 4),
            "margin_attribute_csls": round(sim_csls[i, correct_idx] - sim_csls[i, attribute_idx], 4),
        })

    df = pd.DataFrame(rows)
    # 샘플별로 "행동 오답 margin < 속성 오답 margin"인지(=행동 변경 구분이 더 어려운지) 표시
    df["action이 더 어려움? (raw)"] = df["margin_action_raw"] < df["margin_attribute_raw"]
    df["action이 더 어려움? (csls)"] = df["margin_action_csls"] < df["margin_attribute_csls"]

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("=" * 100)
    print("샘플별 margin: 행동 변경 오답 vs 객체/속성 변경 오답")
    print("=" * 100)
    print(df.to_string(index=False))

    # --- 전체 평균 비교 ---
    mean_action_raw = df["margin_action_raw"].mean()
    mean_attr_raw = df["margin_attribute_raw"].mean()
    mean_action_csls = df["margin_action_csls"].mean()
    mean_attr_csls = df["margin_attribute_csls"].mean()

    n_action_harder_raw = df["action이 더 어려움? (raw)"].sum()
    n_action_harder_csls = df["action이 더 어려움? (csls)"].sum()

    print("\n" + "=" * 100)
    print("전체 평균 비교")
    print("=" * 100)
    print(f"{'':20s} {'행동 변경 margin 평균':>22s} {'속성 변경 margin 평균':>22s} {'차이(속성-행동)':>16s}")
    print(f"{'원본 코사인':20s} {mean_action_raw:>22.4f} {mean_attr_raw:>22.4f} {mean_attr_raw - mean_action_raw:>+16.4f}")
    print(f"{'CSLS 보정':20s} {mean_action_csls:>22.4f} {mean_attr_csls:>22.4f} {mean_attr_csls - mean_action_csls:>+16.4f}")

    print(f"\n8개 샘플 중 '행동 변경 오답이 더 헷갈림'에 해당하는 샘플 수:")
    print(f"  원본 코사인 기준: {n_action_harder_raw}/8")
    print(f"  CSLS 보정 기준  : {n_action_harder_csls}/8")

    print("\n" + "=" * 100)
    print("결론")
    print("=" * 100)
    if mean_action_raw < mean_attr_raw and mean_action_csls < mean_attr_csls:
        print("가설 지지: 두 기준(원본/CSLS) 모두에서 '행동 변경' margin 평균이 '속성 변경' margin 평균보다 작음.")
        print("-> CLIP은 객체/속성이 바뀐 오답보다 행동(동작)만 바뀐 오답을 구분하는 데 더 취약한 경향을 보인다.")
    elif mean_action_raw > mean_attr_raw and mean_action_csls > mean_attr_csls:
        print("가설 기각: 오히려 '속성 변경' margin 평균이 더 작음 (CLIP이 속성 변경을 더 못 구분함).")
    else:
        print("결과가 원본/CSLS 기준에서 엇갈림 -> 이 데이터셋(8개 샘플)만으로는 결론을 확정하기 어려움. 표본을 늘려 재검증 필요.")

    print(f"\n결과 CSV 저장 위치: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
