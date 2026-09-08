"""
8단계: 객체 단위 검증이 dog_park / cooking_thermometer의 놓친 오류를 잡아내는가
================================================================================

7단계에서 문장 전체 점수(BLIP-ITM)로는 놓쳤던 두 케이스:
  - dog_park: "개 세 마리" -> "고양이 세 마리" (속성변경 오답, 문장 점수 0.9933으로 PASS 오판)
  - cooking_thermometer: "버거" -> "치킨" (속성변경 오답, 문장 점수 0.9997으로 PASS 오판)

이 두 샘플에 대해, 문제가 된 핵심 명사(정답 명사 vs 오답으로 바뀐 명사)만
따로 뽑아 "a photo of a {명사}" 형태로 object_check을 돌려서, 문장 전체
점수는 놓쳤던 차이를 객체 단위 점수는 잡아내는지 확인한다.
"""

from __future__ import annotations

import json
from pathlib import Path

from inspector import inspect_caption
from object_check import check_objects

ROOT = Path(__file__).resolve().parent.parent
CAPTIONS_JSON = ROOT / "data" / "captions.json"
VIDEO_DIR = ROOT / "data" / "videos"

TARGET_IDS = ["dog_park", "cooking_thermometer"]

# 두 샘플에서 정답<->오답 사이에 실제로 바뀐 핵심 명사 (수동으로 지정 -
# 문장 비교를 통해 무엇이 바뀌었는지는 이미 3단계에서 확인한 내용이다)
SWAPPED_NOUNS = {
    "dog_park": {"correct": ("dogs", "NNS"), "wrong": ("cats", "NNS")},
    "cooking_thermometer": {"correct": ("burger", "NN"), "wrong": ("chicken", "NN")},
}


def main():
    with open(CAPTIONS_JSON, "r", encoding="utf-8") as f:
        samples = {s["id"]: s for s in json.load(f)["samples"]}

    for sid in TARGET_IDS:
        s = samples[sid]
        video = VIDEO_DIR / s["video_file"]

        print("\n" + "=" * 100)
        print(f"샘플: {sid}")
        print("=" * 100)
        print(f"정답 문장: {s['correct_caption']}")
        print(f"오답 문장(속성변경): {s['wrong_caption_attribute']}")

        # --- 기존 방식: 문장 전체 점수 (7단계 결과 재확인) ---
        sent_correct = inspect_caption(video, s["correct_caption"])
        sent_wrong = inspect_caption(video, s["wrong_caption_attribute"])
        print(f"\n[문장 전체 점수]")
        print(f"  정답 문장: {sent_correct['score']:.4f}  ({sent_correct['verdict']})")
        print(f"  오답 문장: {sent_wrong['score']:.4f}  ({sent_wrong['verdict']})  <- 7단계에서 PASS로 오판했던 값")

        # --- 새 방식: 캡션 전체에서 명사를 자동 추출해 전부 채점 ---
        print(f"\n[정답 문장에서 추출된 명사별 객체 점수]")
        for r in check_objects(video, s["correct_caption"]):
            print(f"  {r['word']:12s} ({r['tag']:4s}) score={r['score']:.4f}")

        print(f"\n[오답 문장에서 추출된 명사별 객체 점수]")
        for r in check_objects(video, s["wrong_caption_attribute"]):
            print(f"  {r['word']:12s} ({r['tag']:4s}) score={r['score']:.4f}")

        # --- 핵심 비교: 실제로 바뀐 명사 하나만 정답 vs 오답으로 직접 대조 ---
        swap = SWAPPED_NOUNS[sid]
        correct_word, correct_tag = swap["correct"]
        wrong_word, wrong_tag = swap["wrong"]

        from object_check import noun_to_query
        correct_query = noun_to_query(correct_word, correct_tag)
        wrong_query = noun_to_query(wrong_word, wrong_tag)

        obj_correct = inspect_caption(video, correct_query)
        obj_wrong = inspect_caption(video, wrong_query)

        print(f"\n[핵심 비교: 실제 바뀐 명사 하나만 대조]")
        print(f"  \"{correct_query}\" (정답 객체) -> {obj_correct['score']:.4f}")
        print(f"  \"{wrong_query}\" (오답 객체)   -> {obj_wrong['score']:.4f}")
        object_margin = obj_correct["score"] - obj_wrong["score"]
        sentence_margin = sent_correct["score"] - sent_wrong["score"]
        print(f"  객체 단위 margin(정답-오답): {object_margin:+.4f}   (문장 단위 margin: {sentence_margin:+.4f})")

        if object_margin > 0.3 and sentence_margin < 0.1:
            print(f"  => 문장 단위로는 거의 못 잡던 차이를 객체 단위 검증이 뚜렷하게 잡아냄!")
        elif object_margin > sentence_margin:
            print(f"  => 객체 단위 검증이 문장 단위보다 더 크게 차이를 벌림.")
        else:
            print(f"  => 객체 단위 검증도 뚜렷한 개선을 보이지 않음.")


if __name__ == "__main__":
    main()
