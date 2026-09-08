"""
객체 표본을 10 -> 20으로 늘려 문장 단위 임계값의 정확도 한계(70%)가
진짜인지 재확인한다. validate_sentence_split_recall.py와 동일한 로직,
동일한 10개 영상에 대해 "두 번째 객체 치환"을 하나씩 더 추가했다.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from frame_extract import extract_frames
from blip_itm import compute_itm_score
from inspector import classify_score
from validate_sentence_split_recall import split_sentences, find_target_sentence, PILOT_ROWS

ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = ROOT / "data" / "pilot_videos"
PREV_CSV = ROOT / "outputs" / "sentence_split_validation.csv"
OUTPUT_CSV = ROOT / "outputs" / "sentence_split_validation_round2.csv"

ROUND2_EDITS = [
    ("example_01", "1.mp4", "a few pedestrians crossing the street near a traffic light", "a few cyclists crossing the street near a traffic light"),
    ("example_02", "2.mp4", "The road is flanked by modern buildings and trees", "The road is flanked by modern warehouses and trees"),
    ("example_03", "3.mp4", "pedestrians visible on the sidewalks holding umbrellas", "cyclists visible on the sidewalks holding umbrellas"),
    ("example_04", "4.mp4", "Pedestrians with umbrellas are visible on the right sidewalk", "Cyclists with umbrellas are visible on the right sidewalk"),
    ("example_05", "5.mp4", "a concrete barrier on the left", "a concrete wall on the left"),
    ("example_06", "6.mp4", "a black SUV also in the adjacent lane", "a black van also in the adjacent lane"),
    ("example_07", "7.mp4", "a multi-story structure with glass panels on the right", "a multi-story structure with brick walls on the right"),
    ("example_08", "8.mp4", "a few pedestrians visible on sidewalks", "a few cyclists visible on sidewalks"),
    ("example_09", "9.mp4", "multi-story buildings featuring commercial storefronts", "multi-story buildings featuring parking garages"),
    ("example_10", "10.mp4", "a landscaped green area visible to the right", "a landscaped parking lot visible to the right"),
]


def main():
    frame_cache = {}
    results = []

    for vid_id, video_file, old_phrase, new_phrase in ROUND2_EDITS:
        caption = PILOT_ROWS[vid_id]["caption"]
        sentences = split_sentences(caption)
        idx, target_sentence = find_target_sentence(sentences, old_phrase)
        wrong_sentence = target_sentence.replace(old_phrase, new_phrase)

        if video_file not in frame_cache:
            frame_cache[video_file] = extract_frames(VIDEO_DIR / video_file, n_frames=5)
        frames = frame_cache[video_file]

        correct_score = max(compute_itm_score(f, target_sentence) for f in frames)
        wrong_score = max(compute_itm_score(f, wrong_sentence) for f in frames)
        margin = correct_score - wrong_score

        print(f"[{vid_id}] 문장 {idx+1}/{len(sentences)}")
        print(f"  정답: \"{target_sentence[:70]}...\" -> {correct_score:.4f}")
        print(f"  오답: \"{wrong_sentence[:70]}...\" -> {wrong_score:.4f}  margin={margin:+.4f}")
        print()

        results.append({
            "video_id": vid_id, "type": "object", "round": 2,
            "target_sentence": target_sentence, "wrong_sentence": wrong_sentence,
            "correct_score": round(correct_score, 4), "wrong_score": round(wrong_score, 4),
            "margin": round(margin, 4),
        })

    df_new = pd.DataFrame(results)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_new.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    # 1라운드(10개) + 2라운드(10개) 합쳐서 20개로 재계산
    df_prev = pd.read_csv(PREV_CSV)
    df_prev_obj = df_prev[df_prev["type"] == "object"][["video_id", "correct_score", "wrong_score"]].copy()
    df_prev_obj["round"] = 1
    df_all = pd.concat([df_prev_obj, df_new[["video_id", "correct_score", "wrong_score", "round"]]], ignore_index=True)

    import numpy as np
    correct = df_all["correct_score"].values
    wrong = df_all["wrong_score"].values

    print("=" * 90)
    print(f"객체 표본 합산 (n={len(df_all)} = 1라운드 10 + 2라운드 10)")
    print("=" * 90)
    print(f"정답: min={correct.min():.4f} max={correct.max():.4f} mean={correct.mean():.4f} median={np.median(correct):.4f}")
    print(f"오답: min={wrong.min():.4f} max={wrong.max():.4f} mean={wrong.mean():.4f} median={np.median(wrong):.4f}")

    candidates = np.unique(np.concatenate([correct, wrong]))
    midpoints = (candidates[:-1] + candidates[1:]) / 2
    best_t, best_acc = 0.5, -1
    for t in midpoints:
        acc = ((correct >= t).sum() + (wrong < t).sum()) / (len(correct) + len(wrong))
        if acc > best_acc:
            best_acc = acc
            best_t = t
    print(f"\n정확도 최대화 임계값: {best_t:.4f} (정확도 {best_acc*100:.1f}%, {len(df_all)*2}개 중)")

    tp = (correct >= best_t).sum()
    fp = (wrong >= best_t).sum()
    print(f"정답 중 PASS 유지: {tp}/{len(correct)}   오답 중 PASS로 오판: {fp}/{len(wrong)}")

    combined_out = ROOT / "outputs" / "sentence_split_object_combined_20.csv"
    df_all.to_csv(combined_out, index=False, encoding="utf-8-sig")
    print(f"\n합산 결과 CSV: {combined_out}")


if __name__ == "__main__":
    main()
