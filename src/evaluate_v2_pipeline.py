"""
13단계: PASS 분기에 행동 검증을 추가한 새 파이프라인을 8개 샘플에 적용
=========================================================================

11단계와 동일한 테스트 세트(정답 8 + 문제 오답 4 + 이미 걸러지던 행동오답 2)에
새로 수정한 inspect_caption_full()을 적용해서, 11단계 대비 무엇이 달라졌는지
비교한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from full_inspector import inspect_caption_full

ROOT = Path(__file__).resolve().parent.parent
CAPTIONS_JSON = ROOT / "data" / "captions.json"
VIDEO_DIR = ROOT / "data" / "videos"
OUTPUT_CSV = ROOT / "outputs" / "stage13_v2_pipeline.csv"


def main():
    with open(CAPTIONS_JSON, "r", encoding="utf-8") as f:
        samples = {s["id"]: s for s in json.load(f)["samples"]}

    test_cases = []
    for sid, s in samples.items():
        test_cases.append((sid, "정답", s["correct_caption"]))

    test_cases.append(("cat_string_toy", "오답(행동, 잔다)", samples["cat_string_toy"]["wrong_caption_action"]))
    test_cases.append(("cooking_thermometer", "오답(속성, 치킨)", samples["cooking_thermometer"]["wrong_caption_attribute"]))
    test_cases.append(("van_driving", "오답(행동, 주차됨)", samples["van_driving"]["wrong_caption_action"]))
    test_cases.append(("dog_park", "오답(속성, 고양이)", samples["dog_park"]["wrong_caption_attribute"]))
    test_cases.append(("horse_herd", "오답(행동)", samples["horse_herd"]["wrong_caption_action"]))
    test_cases.append(("bicycle_ride", "오답(행동)", samples["bicycle_ride"]["wrong_caption_action"]))

    rows = []
    for sid, ctype, caption in test_cases:
        video = VIDEO_DIR / samples[sid]["video_file"]
        r = inspect_caption_full(video, caption)

        is_correct_type = (ctype == "정답")
        ok = (r["final_verdict"] == "PASS") if is_correct_type else (r["final_verdict"] != "PASS")

        rows.append({
            "id": sid,
            "구분": ctype,
            "문장점수": round(r["sentence_score"], 4),
            "문장판정": r["sentence_verdict"],
            "최종판정": r["final_verdict"],
            "의심 구성요소": "; ".join(r["suspect_components"]) if r["suspect_components"] else "-",
            "결과": "O" if ok else "X",
        })
        print(f"[{sid:20s}][{ctype:16s}] 문장={r['sentence_score']:.4f}({r['sentence_verdict']:6s}) "
              f"-> 최종={r['final_verdict']:6s}  {'✅' if ok else '❌'}   의심={r['suspect_components']}")

    df = pd.DataFrame(rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 120)
    print("전체 결과표")
    print("=" * 120)
    print(df.to_string(index=False))

    n_correct = (df["구분"] == "정답").sum()
    n_correct_pass = ((df["구분"] == "정답") & (df["최종판정"] == "PASS")).sum()
    n_wrong = (df["구분"] != "정답").sum()
    n_wrong_caught = ((df["구분"] != "정답") & (df["최종판정"] != "PASS")).sum()

    print(f"\n[13단계 결과] 정답 유지율: {n_correct_pass}/{n_correct} ({n_correct_pass/n_correct*100:.1f}%)")
    print(f"[13단계 결과] 오답 검출율(재현율): {n_wrong_caught}/{n_wrong} ({n_wrong_caught/n_wrong*100:.1f}%)")
    print(f"\n[11단계 결과 참고] 정답 유지율: 3/8 (37.5%)  오답 검출율: 6/6 (100%, 단 그중 2개는 우연히 걸린 것으로 확인됨)")

    print(f"\n결과 CSV 저장 위치: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
