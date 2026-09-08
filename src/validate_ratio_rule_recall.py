"""
20% 비율 규칙의 재현율 검증 - "단어 1개만 바꾼" 주행영상 오답 캡션
======================================================================

MSVD/Wikimedia에서 쓰던 것과 같은 방식(단일 명사/동사 하나만 실제와
다른 것으로 교체)으로 주행영상 캡션 5개에 오답 버전을 만들어서, 지난번
제안한 "SUSPECT 비율 > 20%면 REVIEW" 규칙이 이 단일 오류를 잡아내는지
검증한다.

효율성을 위해 이미 계산해둔 outputs/driving_domain_object_scores.csv의
447개 점수를 최대한 재사용한다 - 캡션 하나에서 딱 1단어만 바뀌므로,
바뀌지 않은 나머지 단어 점수는 다시 계산할 필요가 없다. 새로 채점해야
하는 건 "바뀐 그 단어"뿐이다.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from frame_extract import extract_frames
from blip_itm import compute_itm_score
from object_check import extract_nouns, noun_to_query
from action_check import extract_actions, action_to_query

ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = ROOT / "data" / "pilot_videos"
SCORED_CSV = ROOT / "outputs" / "driving_domain_object_scores.csv"
OUTPUT_CSV = ROOT / "outputs" / "driving_wrong_caption_validation.csv"

PER_WORD_THRESHOLD = 0.01   # 지난번 도메인 재보정 제안값
RATIO_THRESHOLD = 0.20      # 지난번 제안한 20% 규칙

# (video_id, video_file, 원래 문구, 바뀐 문구, 종류) - 전부 "화면에 없는 것"으로 교체
WRONG_EDITS = [
    ("example_01", "1.mp4", "a yellow van is parked", "a yellow bus is parked", "object"),
    ("example_01", "1.mp4", "is turning right along", "is turning left along", "action"),
    ("example_02", "2.mp4", "a white van and a silver sedan", "a white van and a silver hatchback", "object"),
    ("example_02", "2.mp4", "proceeds straight along", "makes a sharp U-turn along", "action"),
    ("example_03", "3.mp4", "a silver SUV directly in front", "a silver sedan directly in front", "object"),
    ("example_03", "3.mp4", "proceeds straight along", "comes to a complete stop along", "action"),
    ("example_04", "4.mp4", "a green bus travels ahead", "a green truck travels ahead", "object"),
    ("example_04", "4.mp4", "proceeds straight along", "reverses slowly along", "action"),
    ("example_05", "5.mp4", "a silver sedan and a truck", "a silver sedan and a motorcycle", "object"),
    ("example_05", "5.mp4", "proceeds straight along", "swerves sharply along", "action"),
]


def main():
    scored = pd.read_csv(SCORED_CSV)

    import csv
    pilot_rows = {r["id"]: r for r in csv.DictReader(open(ROOT / "results" / "pilot_result.csv", encoding="utf-8-sig"))}

    frame_cache = {}
    results = []

    for vid_id, video_file, old_phrase, new_phrase, edit_type in WRONG_EDITS:
        original_caption = pilot_rows[vid_id]["caption"]
        assert old_phrase in original_caption, f"'{old_phrase}' not found in {vid_id}"
        wrong_caption = original_caption.replace(old_phrase, new_phrase)

        if video_file not in frame_cache:
            frame_cache[video_file] = extract_frames(VIDEO_DIR / video_file, n_frames=5)
        frames = frame_cache[video_file]

        if edit_type == "object":
            orig_words = {w for w, _ in extract_nouns(original_caption)}
            new_words_tagged = extract_nouns(wrong_caption)
        else:
            orig_words = {w for w, _ in extract_actions(original_caption)}
            new_words_tagged = extract_actions(wrong_caption)

        new_words = {w for w, _ in new_words_tagged}
        changed_words = new_words - orig_words
        removed_words = orig_words - new_words

        # 바뀐(새로 생긴) 단어만 새로 채점. 나머지는 캐시에서 재사용.
        changed_scores = {}
        for w, tag in new_words_tagged:
            if w in changed_words:
                query = noun_to_query(w, tag, style="photo") if edit_type == "object" else action_to_query(w, style="photo")
                score = max(compute_itm_score(f, query) for f in frames)
                changed_scores[w] = score

        if edit_type == "object":
            cached = scored[scored["video_id"] == vid_id]
            base_scores = {row["word"]: row["score"] for _, row in cached.iterrows() if row["word"] in orig_words}
            # 원래 단어 집합에서 제거된 단어를 빼고, 바뀐 단어를 새 점수로 추가
            final_scores = {w: s for w, s in base_scores.items() if w not in removed_words}
            final_scores.update(changed_scores)

            total = len(final_scores)
            suspect = sum(1 for s in final_scores.values() if s < PER_WORD_THRESHOLD)
            ratio = suspect / total
            caught = ratio > RATIO_THRESHOLD

            print(f"[{vid_id}] object 교체: '{old_phrase}' -> '{new_phrase}'")
            print(f"  바뀐 단어: {changed_words} (점수: {changed_scores})")
            print(f"  전체 {total}개 중 SUSPECT {suspect}개 -> 비율 {ratio*100:.1f}%  "
                  f"{'REVIEW (잡힘)' if caught else 'PASS 그대로 (놓침)'}")

            results.append({
                "video_id": vid_id, "type": "object", "old": old_phrase, "new": new_phrase,
                "changed_word": list(changed_words)[0] if changed_words else None,
                "changed_score": round(list(changed_scores.values())[0], 4) if changed_scores else None,
                "total_words": total, "suspect_count": suspect,
                "suspect_ratio": round(ratio, 4), "caught_by_20pct_rule": caught,
            })
        else:
            # 행동은 원래 규칙(비율이 아니라 "하나라도 SUSPECT면") 그대로 검증
            print(f"[{vid_id}] action 교체: '{old_phrase}' -> '{new_phrase}'")
            for w, s in changed_scores.items():
                caught = s < PER_WORD_THRESHOLD
                print(f"  바뀐 행동 단어: {w} = {s:.4f}  {'SUSPECT (잡힘)' if caught else '통과 (놓침)'}")
                results.append({
                    "video_id": vid_id, "type": "action", "old": old_phrase, "new": new_phrase,
                    "changed_word": w, "changed_score": round(s, 4),
                    "total_words": None, "suspect_count": None,
                    "suspect_ratio": None, "caught_by_20pct_rule": caught,
                })
        print()

    df = pd.DataFrame(results)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("=" * 90)
    print("요약")
    print("=" * 90)
    n_obj = df[df["type"] == "object"]
    n_act = df[df["type"] == "action"]
    print(f"객체 교체(20% 비율규칙 적용): {n_obj['caught_by_20pct_rule'].sum()}/{len(n_obj)} 잡음")
    print(f"행동 교체(단일 SUSPECT 판정): {n_act['caught_by_20pct_rule'].sum()}/{len(n_act)} 잡음")
    print(f"\n결과 CSV: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
