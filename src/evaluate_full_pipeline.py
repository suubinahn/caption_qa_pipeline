"""
11단계: 통합 파이프라인(inspect_caption_full)을 전체 테스트 세트에 적용
=========================================================================

테스트 세트 구성 (14개):
  A) 8개 샘플의 "정답 캡션" 전부 - 회귀 테스트. 지금까지 잘 맞던 것들이
     이 새 로직으로도 여전히 잘 맞는지(불필요하게 REVIEW로 떨어지지
     않는지) 확인한다.
  B) 7단계에서 "문장 점수만으로는 잘못 PASS됐던" 4개 오답 - 이번 파이프라인이
     이걸 잡아내는지 확인한다.
       - cat_string_toy / wrong_caption_action ("잔다", 행동 오류)
       - cooking_thermometer / wrong_caption_attribute ("치킨", 객체 오류)
       - van_driving / wrong_caption_action ("주차됨", 행동 오류)
       - dog_park / wrong_caption_attribute ("고양이", 객체 오류)
  C) 10단계에서 다룬 행동 오류 2개 - 문장 단계에서 이미 REVIEW/FAIL로
     걸러지는지, 그리고 진단이 올바르게 "행동"을 지목하는지 확인한다.
       - horse_herd / wrong_caption_action ("서서 풀을 뜯는다")
       - bicycle_ride / wrong_caption_action ("옆에 서있다")
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from full_inspector import inspect_caption_full

ROOT = Path(__file__).resolve().parent.parent
CAPTIONS_JSON = ROOT / "data" / "captions.json"
VIDEO_DIR = ROOT / "data" / "videos"
OUTPUT_CSV = ROOT / "outputs" / "stage11_full_pipeline.csv"


def main():
    with open(CAPTIONS_JSON, "r", encoding="utf-8") as f:
        samples = {s["id"]: s for s in json.load(f)["samples"]}

    test_cases = []
    # A) 8개 정답 전부
    for sid, s in samples.items():
        test_cases.append((sid, "정답", s["correct_caption"]))

    # B) 7단계에서 놓쳤던 4개 오답
    test_cases.append(("cat_string_toy", "오답(행동, 7단계 놓침)", samples["cat_string_toy"]["wrong_caption_action"]))
    test_cases.append(("cooking_thermometer", "오답(속성, 7단계 놓침)", samples["cooking_thermometer"]["wrong_caption_attribute"]))
    test_cases.append(("van_driving", "오답(행동, 7단계 놓침)", samples["van_driving"]["wrong_caption_action"]))
    test_cases.append(("dog_park", "오답(속성, 7단계 놓침)", samples["dog_park"]["wrong_caption_attribute"]))

    # C) 10단계 행동 오류 2개
    test_cases.append(("horse_herd", "오답(행동)", samples["horse_herd"]["wrong_caption_action"]))
    test_cases.append(("bicycle_ride", "오답(행동)", samples["bicycle_ride"]["wrong_caption_action"]))

    rows = []
    for sid, ctype, caption in test_cases:
        video = VIDEO_DIR / samples[sid]["video_file"]
        r = inspect_caption_full(video, caption)

        expected = "PASS" if ctype == "정답" else ("REVIEW/FAIL" if "놓침" in ctype or ctype.startswith("오답") else "?")
        caught = (r["final_verdict"] != "PASS") if ctype != "정답" else (r["final_verdict"] == "PASS")

        rows.append({
            "id": sid,
            "구분": ctype,
            "문장점수": round(r["sentence_score"], 4),
            "문장판정": r["sentence_verdict"],
            "최종판정": r["final_verdict"],
            "스타일": r["style"],
            "객체검증": "O" if r["objects_checked"] else "-",
            "행동검증": "O" if r["actions_checked"] else "-",
            "의심 구성요소": "; ".join(r["suspect_components"]) if r["suspect_components"] else "-",
            "결과": "O" if caught else "X",
            "caption": caption[:50],
        })
        print(f"[{sid:20s}][{ctype:20s}] 문장={r['sentence_score']:.4f}({r['sentence_verdict']:6s}) "
              f"-> 최종={r['final_verdict']:6s}  의심={r['suspect_components']}")

    df = pd.DataFrame(rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 130)
    print("전체 결과표")
    print("=" * 130)
    print(df[["id", "구분", "문장점수", "문장판정", "최종판정", "스타일", "의심 구성요소", "결과"]].to_string(index=False))

    n_correct = (df["구분"] == "정답").sum()
    n_correct_pass = ((df["구분"] == "정답") & (df["최종판정"] == "PASS")).sum()
    n_wrong = (df["구분"] != "정답").sum()
    n_wrong_caught = ((df["구분"] != "정답") & (df["최종판정"] != "PASS")).sum()

    print(f"\n정답 {n_correct}개 중 최종 PASS 유지: {n_correct_pass}/{n_correct} (회귀 없음 확인)")
    print(f"오답 {n_wrong}개 중 최종적으로 PASS를 면한(REVIEW/FAIL) 개수: {n_wrong_caught}/{n_wrong}")

    print(f"\n결과 CSV 저장 위치: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
