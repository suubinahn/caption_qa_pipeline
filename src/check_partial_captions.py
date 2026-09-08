"""
14단계: "부분 정답" 캡션 탐지 - 정답 캡션이 5개 프레임 중 몇 개에서 뒷받침되는가
====================================================================================

cat_string_toy에서 발견한 것: 영상 안에 "쉬는 순간"과 "노는 순간"이 섞여
있는데, 정답 캡션은 "논다"라고만 되어 있었다. 즉 캡션이 영상 전체가
아니라 일부 순간만 정확히 반영하는 "부분 정답(partially-correct caption)"
이었을 수 있다.

이걸 정량적으로 확인하는 방법: inspect_caption()이 하듯 5개 프레임 중
"최댓값"만 보는 게 아니라, **5개 프레임 각각을 정답 캡션과 개별 대조**해서
점수를 낸다. 그리고 이미 검증된 문장 단위 임계값(LOW=0.9101, HIGH=0.9560,
inspector.py 참고)을 프레임 하나하나에 그대로 적용해 PASS/REVIEW/FAIL로
분류한다.

  - 5개 프레임 전부 PASS       -> 캡션이 영상 전체에 걸쳐 일관되게 맞음
  - 일부만 PASS, 나머지는 REVIEW/FAIL -> "부분 정답" 캡션 (cat_string_toy 패턴)
  - 대부분 REVIEW/FAIL        -> 애초에 대표 프레임 선정 자체가 어려운 영상
    (dog_park, horse_herd처럼 핵심 순간이 짧게 스쳐 지나가는 경우)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from frame_extract import extract_frames
from blip_itm import compute_itm_score
from inspector import classify_score

ROOT = Path(__file__).resolve().parent.parent
CAPTIONS_JSON = ROOT / "data" / "captions.json"
VIDEO_DIR = ROOT / "data" / "videos"
OUTPUT_CSV = ROOT / "outputs" / "stage14_partial_captions.csv"

FRACTIONS = [10, 30, 50, 70, 90]


def main():
    with open(CAPTIONS_JSON, "r", encoding="utf-8") as f:
        samples = json.load(f)["samples"]

    rows = []
    for s in samples:
        sid = s["id"]
        video = VIDEO_DIR / s["video_file"]
        caption = s["correct_caption"]

        frames = extract_frames(video, n_frames=5)
        frame_scores = [compute_itm_score(f, caption) for f in frames]
        frame_verdicts = [classify_score(sc) for sc in frame_scores]

        n_pass = frame_verdicts.count("PASS")
        n_review = frame_verdicts.count("REVIEW")
        n_fail = frame_verdicts.count("FAIL")

        rows.append({
            "id": sid,
            "correct_caption": caption[:60],
            **{f"{f}%": round(sc, 4) for f, sc in zip(FRACTIONS, frame_scores)},
            "PASS수": n_pass,
            "REVIEW수": n_review,
            "FAIL수": n_fail,
            "패턴": "일관됨(5/5 PASS)" if n_pass == 5 else (
                "부분 정답" if n_pass >= 1 and (n_review + n_fail) >= 1 else "전체가 애매/실패"
            ),
        })

        print(f"[{sid:20s}] " + "  ".join(f"{f}%={sc:.3f}({v[:1]})" for f, sc, v in zip(FRACTIONS, frame_scores, frame_verdicts))
              + f"   PASS {n_pass}/5")

    df = pd.DataFrame(rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 130)
    print("전체 결과표")
    print("=" * 130)
    print(df[["id", "10%", "30%", "50%", "70%", "90%", "PASS수", "REVIEW수", "FAIL수", "패턴"]].to_string(index=False))

    print("\n" + "=" * 130)
    print("요약")
    print("=" * 130)
    print(df["패턴"].value_counts())

    partial = df[df["패턴"] == "부분 정답"]
    if not partial.empty:
        print(f"\n'부분 정답' 패턴을 보인 샘플 ({len(partial)}개):")
        print(partial[["id", "PASS수", "REVIEW수", "FAIL수"]].to_string(index=False))

    print(f"\n결과 CSV 저장 위치: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
