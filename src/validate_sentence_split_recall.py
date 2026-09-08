"""
문장 단위 분할 검증 - 재현율 재시도
======================================

긴 캡션(문단)을 명사 단위(40개)가 아니라 문장 단위(6~8개)로 쪼갠 뒤,
이미 검증된 "문장 전체 채점"(inspector.classify_score와 동일한 임계값)을
각 문장에 그대로 적용한다.

품사 태깅(NLTK)에 전혀 의존하지 않는다는 게 핵심 차이 - 그래서 지난번
행동(동사) 검증이 실패했던 이유(VBZ 시제라 동사 추출 자체가 안 됨)가
이 방식에는 아예 해당되지 않는다. "문장"은 시제와 무관하게 그냥 문장이다.

validate_ratio_rule_recall.py에서 썼던 것과 같은 5개 오답 편집(객체 5개
+ 행동 5개, 단 행동 5개 중 4개는 지난번 동사 추출 자체가 안 됐던 것들 -
이번엔 문장 단위라 상관없이 전부 유효한 테스트가 된다)을 재사용한다.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from frame_extract import extract_frames
from blip_itm import compute_itm_score
from inspector import classify_score

ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = ROOT / "data" / "pilot_videos"
OUTPUT_CSV = ROOT / "outputs" / "sentence_split_validation.csv"

# validate_ratio_rule_recall.py와 동일한 편집 목록
EDITS = [
    ("example_01", "1.mp4", "a yellow van is parked", "a yellow bus is parked", "object"),
    ("example_01", "1.mp4", "is turning right along", "is reversing along", "action"),
    ("example_02", "2.mp4", "a white van and a silver sedan", "a white van and a silver hatchback", "object"),
    ("example_02", "2.mp4", "proceeds straight along", "makes a sharp U-turn along", "action"),
    ("example_03", "3.mp4", "a silver SUV directly in front", "a silver sedan directly in front", "object"),
    ("example_03", "3.mp4", "proceeds straight along", "comes to a complete stop along", "action"),
    ("example_04", "4.mp4", "a green bus travels ahead", "a green truck travels ahead", "object"),
    ("example_04", "4.mp4", "proceeds straight along", "reverses slowly along", "action"),
    ("example_05", "5.mp4", "a silver sedan and a truck", "a silver sedan and a motorcycle", "object"),
    ("example_05", "5.mp4", "proceeds straight along", "swerves sharply along", "action"),
    ("example_06", "6.mp4", "a silver sedan directly ahead", "a silver hatchback directly ahead", "object"),
    ("example_06", "6.mp4", "proceeds straight along", "makes a quick lane change along", "action"),
    ("example_07", "7.mp4", "a white SUV and a white pickup truck", "a white SUV and a white delivery van", "object"),
    ("example_07", "7.mp4", "proceeds straight along", "brakes suddenly along", "action"),
    ("example_08", "8.mp4", "a black SUV parked on the right", "a black sedan parked on the right", "object"),
    ("example_08", "8.mp4", "proceeds straight along", "accelerates rapidly along", "action"),
    ("example_09", "9.mp4", "a white sedan is parked near a bus stop shelter", "a white van is parked near a bus stop shelter", "object"),
    ("example_09", "9.mp4", "proceeds straight along", "veers into the left lane along", "action"),
    ("example_10", "10.mp4", "a white sedan and a silver sedan", "a white sedan and a silver SUV", "object"),
    ("example_10", "10.mp4", "proceeds straight along", "turns sharply along", "action"),
]

import csv
PILOT_ROWS = {r["id"]: r for r in csv.DictReader(open(ROOT / "results" / "pilot_result.csv", encoding="utf-8-sig"))}


def split_sentences(text: str) -> list[str]:
    """아주 단순한 문장 분리 (마침표 뒤 공백 기준). 캡션들이 전부 표준적인
    영어 문장부호를 쓰고 있어서 이 정도로 충분하다."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def find_target_sentence(sentences: list[str], phrase: str) -> tuple[int, str]:
    for i, s in enumerate(sentences):
        if phrase in s:
            return i, s
    raise ValueError(f"'{phrase}' 를 포함하는 문장을 찾지 못함")


def main():
    frame_cache = {}
    results = []

    for vid_id, video_file, old_phrase, new_phrase, edit_type in EDITS:
        caption = PILOT_ROWS[vid_id]["caption"]
        sentences = split_sentences(caption)
        idx, target_sentence = find_target_sentence(sentences, old_phrase)
        wrong_sentence = target_sentence.replace(old_phrase, new_phrase)

        if video_file not in frame_cache:
            frame_cache[video_file] = extract_frames(VIDEO_DIR / video_file, n_frames=5)
        frames = frame_cache[video_file]

        correct_score = max(compute_itm_score(f, target_sentence) for f in frames)
        wrong_score = max(compute_itm_score(f, wrong_sentence) for f in frames)

        correct_verdict = classify_score(correct_score)
        wrong_verdict = classify_score(wrong_score)
        margin = correct_score - wrong_score
        # 절대 임계값(문단 전체용, 문장 하나엔 안 맞음)이 아니라, 정답이 오답보다
        # "진짜로 유의미하게" 높은지(margin > 0.05)로 판정한다 (지난 라운드에서
        # 절대 임계값 기준이 오답이 더 높은 경우까지 "잡힘"으로 잘못 셌던 걸 수정).
        caught = margin > 0.05

        print(f"[{vid_id}][{edit_type}] 문장 {idx+1}/{len(sentences)}")
        print(f"  정답: \"{target_sentence[:80]}...\" -> {correct_score:.4f} ({correct_verdict})")
        print(f"  오답: \"{wrong_sentence[:80]}...\" -> {wrong_score:.4f} ({wrong_verdict})  margin={margin:+.4f}  "
              f"{'✅ 잡힘' if caught else '❌ 놓침'}")
        print()

        results.append({
            "video_id": vid_id, "type": edit_type, "sentence_index": idx, "total_sentences": len(sentences),
            "target_sentence": target_sentence, "wrong_sentence": wrong_sentence,
            "correct_score": round(correct_score, 4), "correct_verdict": correct_verdict,
            "wrong_score": round(wrong_score, 4), "wrong_verdict": wrong_verdict,
            "margin": round(margin, 4), "caught": caught,
        })

    df = pd.DataFrame(results)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("=" * 90)
    print("요약 (margin > 0.05 기준)")
    print("=" * 90)
    print(df.groupby("type").agg(진짜구분=("caught", "sum"), 표본수=("caught", "count"), 평균margin=("margin", "mean")))
    print(f"\n전체 재현율: {df['caught'].sum()}/{len(df)} ({df['caught'].mean()*100:.1f}%)")
    print(f"평균 문장 개수: {df['total_sentences'].mean():.1f}개 (참고: 명사는 34~59개였음)")
    print(f"\n결과 CSV: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
