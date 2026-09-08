"""
12단계: 집계 규칙 튜닝 - "ANY(1개 이상)" vs "N개 이상"
==========================================================

11단계에서 확인한 문제: full_inspector.py의 PASS 분기는 "객체 단어 중
하나라도 SUSPECT면 REVIEW로 강등"(ANY 규칙)한다. 문장 하나에 명사가
5~8개씩 있으니, 단어 하나당 오탐률이 낮아도(12단계 이전 재보정으로
91.5%까지 개선했지만 여전히 100%는 아님) 누적되면서 문장 전체가 잘못
REVIEW로 빠지는 경우가 많았다 (8개 정답 중 3개만 PASS 유지).

이번 실험은 "SUSPECT 단어가 N개 이상일 때만 REVIEW"로 집계 규칙을
바꿔서, N을 1~5까지 바꿔가며:
  - 정답 8개가 얼마나 깨끗하게 PASS로 유지되는지 (정밀도 쪽)
  - 진짜 문제가 있는 오답들을 얼마나 잘 잡아내는지 (재현율 쪽)
를 비교한다. component_dataset_scored.csv에 이미 계산해둔 점수를
재사용하므로 새로 채점할 필요가 없다 - 각 캡션 문장에서 실제로 어떤
명사가 추출되는지만 다시 확인하고, 그 단어들의 점수를 조회하면 된다.

[중요 - 미리 알아야 할 구조적 한계]
full_inspector.py의 PASS 분기는 "객체만" 검증하고 "행동(동사)"은 검증
하지 않는다. 그래서 cat_string_toy(잔다, 행동 오류)와 van_driving(주차됨,
행동 오류)은 오답 문장이어도 명사 목록 자체가 정답과 완전히 동일하다.
즉 N을 아무리 조정해도 이 두 오답은 "객체 단위 검증"으로는 원리적으로
잡을 수 없다 - 이건 임계값/집계 규칙의 문제가 아니라 애초에 "행동 오류를
객체 검증으로 잡으려 한 것" 자체가 설계상 불가능한 시도였다는 뜻이다.
(11단계에서 이게 "잡혔던" 것처럼 보였던 건, 그 당시 임계값이 너무
헐거워서 무관한 다른 단어들이 우연히 SUSPECT로 찍혔기 때문이었다.)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from object_check import extract_nouns

ROOT = Path(__file__).resolve().parent.parent
CAPTIONS_JSON = ROOT / "data" / "captions.json"
SCORED_CSV = ROOT / "outputs" / "component_dataset_scored.csv"
OUTPUT_CSV = ROOT / "outputs" / "stage12_aggregation_tuning.csv"

OLD_THRESHOLD = 0.05
NEW_THRESHOLD = 0.0214

# PASS 분기(객체만 검증)로 실제 도달하는 오답 4개 - 나머지 2개(horse_herd,
# bicycle_ride의 행동오답)는 문장 단계에서 이미 FAIL이라 이 실험과 무관.
PROBLEM_WRONGS = [
    ("cat_string_toy", "wrong_caption_action", "행동 오류(잔다)"),
    ("cooking_thermometer", "wrong_caption_attribute", "객체 오류(치킨)"),
    ("van_driving", "wrong_caption_action", "행동 오류(주차됨)"),
    ("dog_park", "wrong_caption_attribute", "객체 오류(고양이)"),
]


def load_scores() -> dict[tuple[str, str], float]:
    df = pd.read_csv(SCORED_CSV)
    return {(row["video_file"], row["word"]): row["score"] for _, row in df.iterrows()}


def suspect_words(caption: str, video_file: str, score_lookup: dict, threshold: float) -> list[str]:
    """이 특정 캡션 문장에서 추출되는 명사들 중, threshold 미만인 것들을 반환."""
    nouns = extract_nouns(caption)  # [(word, tag), ...]
    result = []
    for word, _tag in nouns:
        score = score_lookup.get((video_file, word))
        if score is not None and score < threshold:
            result.append(f"{word}({score:.3f})")
    return result


def main():
    import json
    with open(CAPTIONS_JSON, "r", encoding="utf-8") as f:
        samples = {s["id"]: s for s in json.load(f)["samples"]}

    score_lookup = load_scores()

    N_VALUES = [1, 2, 3, 4, 5]
    THRESHOLDS = [("기존", OLD_THRESHOLD), ("재보정", NEW_THRESHOLD)]

    summary_rows = []
    detail_rows = []

    for t_label, threshold in THRESHOLDS:
        for n in N_VALUES:
            # --- 정답 8개: SUSPECT 단어 개수가 n 미만이면 PASS 유지 ---
            n_correct_pass = 0
            for sid, s in samples.items():
                sw = suspect_words(s["correct_caption"], s["video_file"], score_lookup, threshold)
                stays_pass = len(sw) < n
                n_correct_pass += int(stays_pass)
                detail_rows.append({
                    "threshold": t_label, "N": n, "id": sid, "구분": "정답",
                    "suspect_count": len(sw), "suspect_words": ", ".join(sw),
                    "결과": "PASS 유지" if stays_pass else "REVIEW로 강등",
                })

            # --- 문제 오답 4개: SUSPECT 단어 개수가 n 이상이면 (제대로) 잡힘 ---
            n_wrong_caught = 0
            for sid, field, desc in PROBLEM_WRONGS:
                s = samples[sid]
                sw = suspect_words(s[field], s["video_file"], score_lookup, threshold)
                caught = len(sw) >= n
                n_wrong_caught += int(caught)
                detail_rows.append({
                    "threshold": t_label, "N": n, "id": f"{sid}/{desc}", "구분": "오답",
                    "suspect_count": len(sw), "suspect_words": ", ".join(sw),
                    "결과": "REVIEW로 잡힘" if caught else "PASS로 놓침",
                })

            summary_rows.append({
                "임계값": t_label,
                "N (이 개수 이상이면 REVIEW)": n,
                "정답 PASS 유지": f"{n_correct_pass}/8",
                "정답 유지율": round(n_correct_pass / 8 * 100, 1),
                "오답 검출": f"{n_wrong_caught}/4",
                "오답 검출율(재현율)": round(n_wrong_caught / 4 * 100, 1),
            })

    summary_df = pd.DataFrame(summary_rows)
    detail_df = pd.DataFrame(detail_rows)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    detail_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("=" * 100)
    print("요약: 임계값 x N 조합별 정답 유지율 / 오답 검출율")
    print("=" * 100)
    print(summary_df.to_string(index=False))

    print("\n" + "=" * 100)
    print("상세: 재보정 임계값(0.0214) 기준, N=1일 때 문제 오답 4개 각각의 상태")
    print("=" * 100)
    detail_n1_new = detail_df[(detail_df["threshold"] == "재보정") & (detail_df["N"] == 1) & (detail_df["구분"] == "오답")]
    print(detail_n1_new[["id", "suspect_count", "suspect_words", "결과"]].to_string(index=False))

    print(f"\n상세 결과 CSV 저장 위치: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
