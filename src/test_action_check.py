"""
10단계: 행동 단위 검증이 horse_herd / bicycle_ride의 행동 오류를 잡아내는가
============================================================================

지금까지 문장 전체 점수(BLIP-ITM)로 봤던 이 두 샘플의 행동 오답 margin
(7단계 결과):
  - horse_herd: 정답(달린다) vs 오답(서서 풀을 뜯는다) -> 문장 margin이 작음
  - bicycle_ride: 정답(탄다) vs 오답(옆에 서있다) -> 문장 margin이 작음
    (심지어 4단계 최초 BLIP 단일 프레임 테스트에서는 margin이 음수에
    가까웠던 원조 사례)

이번엔 캡션에서 동사만 뽑아 "a {style} of {동사}" 형태로 독립 검증해서,
문장 단위로는 흐릿했던 차이가 행동 단위로는 뚜렷해지는지 확인한다.
스타일은 CLIP으로 자동 판별한 값을 그대로 쓴다 (9단계와 동일한 방식).
"""

from __future__ import annotations

import json
from pathlib import Path

from inspector import inspect_caption
from action_check import check_actions_auto_style, action_to_query
from style_detect import detect_video_style

ROOT = Path(__file__).resolve().parent.parent
CAPTIONS_JSON = ROOT / "data" / "captions.json"
VIDEO_DIR = ROOT / "data" / "videos"

TARGET_IDS = ["horse_herd", "bicycle_ride"]

# 정답/오답 문장에서 실제로 바뀐 핵심 동사 (3단계에서 이미 파악한 내용)
SWAPPED_VERBS = {
    "horse_herd": {"correct": "running", "wrong": ["standing", "grazing"]},
    "bicycle_ride": {"correct": "riding", "wrong": ["standing"]},
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
        print(f"오답 문장(행동변경): {s['wrong_caption_action']}")

        # --- 0) 문장 전체 점수 (기준선) ---
        sent_correct = inspect_caption(video, s["correct_caption"])
        sent_wrong = inspect_caption(video, s["wrong_caption_action"])
        sentence_margin = sent_correct["score"] - sent_wrong["score"]
        print(f"\n[문장 전체 점수]")
        print(f"  정답: {sent_correct['score']:.4f}  오답: {sent_wrong['score']:.4f}  margin={sentence_margin:+.4f}")

        # --- 1) 스타일 자동 판별 ---
        style_info = detect_video_style(video)
        print(f"\n[스타일 자동 판별] {style_info['style']}  (투표: {style_info['vote_counts']})")

        # --- 2) 정답/오답 문장에서 동사 자동 추출 + 행동 단위 검증 ---
        result_correct = check_actions_auto_style(video, s["correct_caption"])
        result_wrong = check_actions_auto_style(video, s["wrong_caption_action"])

        print(f"\n[정답 문장에서 추출된 동사 검증]")
        for r in result_correct["actions"]:
            print(f"  {r['word']:12s} -> \"{r['query']}\"  score={r['score']:.4f}")

        print(f"\n[오답 문장에서 추출된 동사 검증]")
        for r in result_wrong["actions"]:
            print(f"  {r['word']:12s} -> \"{r['query']}\"  score={r['score']:.4f}")

        # --- 3) 핵심 비교: 실제 바뀐 동사만 정답 vs 오답으로 직접 대조 ---
        swap = SWAPPED_VERBS[sid]
        style = style_info["style"]
        correct_verb = swap["correct"]
        wrong_verbs = swap["wrong"]

        q_correct = action_to_query(correct_verb, style=style)
        r_correct = inspect_caption(video, q_correct)

        print(f"\n[핵심 비교: 실제 바뀐 동사만 대조]")
        print(f"  \"{q_correct}\" (정답 행동) -> {r_correct['score']:.4f}")

        wrong_scores = []
        for wv in wrong_verbs:
            q_wrong = action_to_query(wv, style=style)
            r_wrong = inspect_caption(video, q_wrong)
            wrong_scores.append(r_wrong["score"])
            print(f"  \"{q_wrong}\" (오답 행동)   -> {r_wrong['score']:.4f}")

        wrong_mean = sum(wrong_scores) / len(wrong_scores)
        action_margin = r_correct["score"] - wrong_mean

        print(f"\n  행동 단위 margin(정답-오답평균): {action_margin:+.4f}   (문장 단위 margin: {sentence_margin:+.4f})")

        if action_margin > sentence_margin + 0.1:
            print(f"  => 문장 단위로 흐릿했던 행동 차이를 동사 단위 검증이 뚜렷하게 잡아냄!")
        elif action_margin > sentence_margin:
            print(f"  => 동사 단위 검증이 문장 단위보다 다소 개선됨.")
        else:
            print(f"  => 동사 단위 검증도 뚜렷한 개선을 보이지 않음 (예상과 다름 - 원인 확인 필요).")


if __name__ == "__main__":
    main()
