"""
16단계: 20개 샘플(기존 8 + 신규 12) 전체 재검증
==================================================

13단계에서 확정한 inspect_caption_full()을 파라미터 변경 없이 그대로
20개 샘플(정답 20개 + 오답 3종 x 20개 = 60개, 총 80개 항목)에 적용한다.

비교 대상:
  - 기존 8개만의 결과 (13단계 결과 재사용/재현)
  - 신규 12개만의 결과
  - 20개 전체 결과
이 세 가지를 나란히 비교해서, 표본이 늘어나도 13단계의 결론(정답
유지율 62.5%, 오답 검출율 83.3%)이 유지되는지 확인한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from full_inspector import inspect_caption_full

ROOT = Path(__file__).resolve().parent.parent
CAPTIONS_JSON = ROOT / "data" / "captions.json"
VIDEO_DIR = ROOT / "data" / "videos"
OUTPUT_CSV = ROOT / "outputs" / "stage16_largescale_results.csv"

OLD_IDS = {"cat_string_toy", "cooking_thermometer", "van_driving", "dog_park",
           "horse_herd", "guitar_play", "people_dancing", "bicycle_ride"}


def main():
    with open(CAPTIONS_JSON, "r", encoding="utf-8") as f:
        samples = {s["id"]: s for s in json.load(f)["samples"]}

    test_cases = []
    for sid, s in samples.items():
        group = "기존8" if sid in OLD_IDS else "신규12"
        test_cases.append((sid, group, "정답", s["correct_caption"]))
        test_cases.append((sid, group, "오답(재활용)", s["wrong_caption"]))
        test_cases.append((sid, group, "오답(행동)", s["wrong_caption_action"]))
        test_cases.append((sid, group, "오답(속성)", s["wrong_caption_attribute"]))

    print(f"총 {len(test_cases)}개 항목 검증 시작 (20 샘플 x 4 캡션)...\n")

    rows = []
    for i, (sid, group, ctype, caption) in enumerate(test_cases):
        video = VIDEO_DIR / samples[sid]["video_file"]
        r = inspect_caption_full(video, caption)

        is_correct_type = (ctype == "정답")
        ok = (r["final_verdict"] == "PASS") if is_correct_type else (r["final_verdict"] != "PASS")

        rows.append({
            "id": sid, "group": group, "구분": ctype,
            "문장점수": round(r["sentence_score"], 4), "문장판정": r["sentence_verdict"],
            "최종판정": r["final_verdict"],
            "의심 구성요소": "; ".join(r["suspect_components"]) if r["suspect_components"] else "-",
            "결과": "O" if ok else "X",
        })
        print(f"[{i+1:3d}/{len(test_cases)}] [{sid:20s}][{group}][{ctype:10s}] "
              f"문장={r['sentence_score']:.4f}({r['sentence_verdict']:6s}) -> 최종={r['final_verdict']:6s}  "
              f"{'OK' if ok else 'FAIL'}")

    df = pd.DataFrame(rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 100)
    print("그룹별 요약")
    print("=" * 100)
    for group_name, gdf in [("기존 8개", df[df["group"] == "기존8"]),
                              ("신규 12개", df[df["group"] == "신규12"]),
                              ("전체 20개", df)]:
        correct = gdf[gdf["구분"] == "정답"]
        wrong = gdf[gdf["구분"] != "정답"]
        n_correct_pass = (correct["최종판정"] == "PASS").sum()
        n_wrong_caught = (wrong["최종판정"] != "PASS").sum()
        print(f"\n[{group_name}]")
        print(f"  정답 유지율: {n_correct_pass}/{len(correct)} ({n_correct_pass/len(correct)*100:.1f}%)")
        print(f"  오답 검출율(재현율): {n_wrong_caught}/{len(wrong)} ({n_wrong_caught/len(wrong)*100:.1f}%)")

    print(f"\n결과 CSV 저장 위치: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
