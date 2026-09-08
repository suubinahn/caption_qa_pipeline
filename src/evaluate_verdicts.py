"""
7단계: PASS/REVIEW/FAIL 판정을 8개 샘플 전체(정답+오답)에 적용
================================================================

inspect_caption()에 새로 추가된 verdict 판정이 실제로 정답은 PASS 쪽으로,
오답은 FAIL 쪽으로 잘 갈리는지 32개 항목(정답 8 + 오답 24) 전체에 대해
확인한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from inspector import inspect_caption

ROOT = Path(__file__).resolve().parent.parent
CAPTIONS_JSON = ROOT / "data" / "captions.json"
VIDEO_DIR = ROOT / "data" / "videos"
OUTPUT_CSV = ROOT / "outputs" / "stage7_verdicts.csv"

TYPE_LABELS = {
    "correct": "정답",
    "wrong_caption": "오답(재활용)",
    "wrong_caption_action": "오답(행동변경)",
    "wrong_caption_attribute": "오답(속성변경)",
}


def main():
    with open(CAPTIONS_JSON, "r", encoding="utf-8") as f:
        samples = json.load(f)["samples"]

    rows = []
    for s in samples:
        video = VIDEO_DIR / s["video_file"]

        captions_to_check = [("correct", s["correct_caption"])]
        for key in ["wrong_caption", "wrong_caption_action", "wrong_caption_attribute"]:
            captions_to_check.append((key, s[key]))

        for ctype, caption in captions_to_check:
            result = inspect_caption(video, caption)
            expected = "PASS" if ctype == "correct" else "FAIL"
            correct_judgement = (
                (ctype == "correct" and result["verdict"] == "PASS")
                or (ctype != "correct" and result["verdict"] == "FAIL")
            )
            rows.append({
                "id": s["id"],
                "구분": TYPE_LABELS[ctype],
                "score": round(result["score"], 4),
                "verdict": result["verdict"],
                "기대값": expected,
                "일치?": "O" if correct_judgement else ("REVIEW" if result["verdict"] == "REVIEW" else "X"),
                "caption": caption[:55],
            })

    df = pd.DataFrame(rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("=" * 120)
    print("전체 32개 항목 (정답 8 + 오답 24) 판정 결과")
    print("=" * 120)
    print(df.to_string(index=False))

    print("\n" + "=" * 120)
    print("요약")
    print("=" * 120)
    verdict_by_type = df.groupby(["구분", "verdict"]).size().unstack(fill_value=0)
    print(verdict_by_type)

    n_correct_ok = ((df["구분"] == "정답") & (df["verdict"] == "PASS")).sum()
    n_correct_total = (df["구분"] == "정답").sum()
    n_wrong_ok = ((df["구분"] != "정답") & (df["verdict"] == "FAIL")).sum()
    n_wrong_review = ((df["구분"] != "정답") & (df["verdict"] == "REVIEW")).sum()
    n_wrong_total = (df["구분"] != "정답").sum()
    n_wrong_bad = ((df["구분"] != "정답") & (df["verdict"] == "PASS")).sum()

    print(f"\n정답 8개 중 PASS: {n_correct_ok}/{n_correct_total}")
    print(f"오답 {n_wrong_total}개 중 FAIL: {n_wrong_ok}/{n_wrong_total}, REVIEW: {n_wrong_review}/{n_wrong_total}, "
          f"잘못 PASS된 개수: {n_wrong_bad}/{n_wrong_total}")

    if n_wrong_bad > 0:
        print("\n[주의] 오답인데 PASS로 잘못 판정된 항목:")
        print(df[(df["구분"] != "정답") & (df["verdict"] == "PASS")][["id", "구분", "score", "caption"]].to_string(index=False))

    print(f"\n결과 CSV 저장 위치: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
