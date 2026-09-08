"""
9단계: 스타일 자동판별 + 객체검증 통합 테스트 (cooking_thermometer)
=====================================================================

8단계에서는 "a photo of a X"가 안 맞아서 사람이 직접 "a cartoon of a X"로
바꿔봐야 했다. 이번엔 CLIP 제로샷 스타일 분류로 그 과정을 자동화한 뒤,
동일한 결과(혹은 더 나은 결과)가 나오는지 확인한다.

비교 대상 3가지:
  1) 고정 템플릿 "photo" (8단계 최초 시도 - 실패했던 방식)
  2) 사람이 수동으로 고른 "cartoon" (8단계에서 직접 시도해서 성공을 확인한 값)
  3) CLIP 자동판별 스타일 (이번 단계 - 사람 개입 없이 자동으로 나온 값)
"""

from __future__ import annotations

import json
from pathlib import Path

from inspector import inspect_caption
from object_check import check_objects_auto_style, noun_to_query
from style_detect import detect_video_style

ROOT = Path(__file__).resolve().parent.parent
CAPTIONS_JSON = ROOT / "data" / "captions.json"
VIDEO_DIR = ROOT / "data" / "videos"

SID = "cooking_thermometer"
SWAP = {"correct": ("burger", "NN"), "wrong": ("chicken", "NN")}


def main():
    with open(CAPTIONS_JSON, "r", encoding="utf-8") as f:
        samples = {s["id"]: s for s in json.load(f)["samples"]}
    s = samples[SID]
    video = VIDEO_DIR / s["video_file"]

    print(f"샘플: {SID}")
    print(f"정답 문장: {s['correct_caption']}")
    print(f"오답 문장(속성변경): {s['wrong_caption_attribute']}\n")

    # --- 0) 문장 전체 점수 (기준선, 7단계에서 이미 확인한 값) ---
    sent_correct = inspect_caption(video, s["correct_caption"])
    sent_wrong = inspect_caption(video, s["wrong_caption_attribute"])
    print(f"[문장 전체 점수] 정답={sent_correct['score']:.4f}  오답={sent_wrong['score']:.4f}  "
          f"margin={sent_correct['score']-sent_wrong['score']:+.4f}\n")

    correct_word, correct_tag = SWAP["correct"]
    wrong_word, wrong_tag = SWAP["wrong"]

    # --- 1) 고정 템플릿 "photo" (8단계 최초 실패 사례) ---
    q_correct_photo = noun_to_query(correct_word, correct_tag, style="photo")
    q_wrong_photo = noun_to_query(wrong_word, wrong_tag, style="photo")
    r1c = inspect_caption(video, q_correct_photo)
    r1w = inspect_caption(video, q_wrong_photo)
    margin1 = r1c["score"] - r1w["score"]
    print(f"[1. 고정 템플릿 'photo']")
    print(f"  \"{q_correct_photo}\" -> {r1c['score']:.4f}")
    print(f"  \"{q_wrong_photo}\" -> {r1w['score']:.4f}")
    print(f"  margin = {margin1:+.4f}\n")

    # --- 2) 사람이 수동으로 고른 "cartoon" ---
    q_correct_cartoon = noun_to_query(correct_word, correct_tag, style="cartoon")
    q_wrong_cartoon = noun_to_query(wrong_word, wrong_tag, style="cartoon")
    r2c = inspect_caption(video, q_correct_cartoon)
    r2w = inspect_caption(video, q_wrong_cartoon)
    margin2 = r2c["score"] - r2w["score"]
    print(f"[2. 수동 지정 템플릿 'cartoon']")
    print(f"  \"{q_correct_cartoon}\" -> {r2c['score']:.4f}")
    print(f"  \"{q_wrong_cartoon}\" -> {r2w['score']:.4f}")
    print(f"  margin = {margin2:+.4f}\n")

    # --- 3) CLIP 자동판별 스타일 (사람 개입 없음) ---
    style_info = detect_video_style(video)
    print(f"[3. CLIP 자동 스타일 판별]")
    print(f"  판별 결과: {style_info['style']}  (프레임별 투표: {style_info['vote_counts']})")

    result_correct = check_objects_auto_style(video, s["correct_caption"])
    result_wrong = check_objects_auto_style(video, s["wrong_caption_attribute"])

    obj_correct = next(o for o in result_correct["objects"] if o["word"] == correct_word)
    obj_wrong = next(o for o in result_wrong["objects"] if o["word"] == wrong_word)
    margin3 = obj_correct["score"] - obj_wrong["score"]

    print(f"  \"{obj_correct['query']}\" -> {obj_correct['score']:.4f}")
    print(f"  \"{obj_wrong['query']}\" -> {obj_wrong['score']:.4f}")
    print(f"  margin = {margin3:+.4f}\n")

    print("=" * 80)
    print("요약 비교")
    print("=" * 80)
    print(f"{'방식':30s} {'margin':>10s}")
    print(f"{'문장 전체 (기준선)':30s} {sent_correct['score']-sent_wrong['score']:>+10.4f}")
    print(f"{'객체 단위, 고정 photo':30s} {margin1:>+10.4f}")
    print(f"{'객체 단위, 수동 cartoon':30s} {margin2:>+10.4f}")
    print(f"{'객체 단위, 자동판별(' + style_info['style'] + ')':30s} {margin3:>+10.4f}")

    if abs(margin3 - margin2) < 0.02:
        print("\n=> 자동 판별 결과가 사람이 수동으로 고른 값과 거의 동일! 자동화 성공.")
    else:
        print("\n=> 자동 판별 결과가 수동 값과 차이가 남 - 원인 확인 필요.")


if __name__ == "__main__":
    main()
