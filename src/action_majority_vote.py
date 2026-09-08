"""
15단계: "지배적 행동" 다수결 판정
====================================

14단계에서 확인한 문제: cat_string_toy처럼 영상 안에 "쉼"과 "놀이"가
섞여 있으면, 5개 프레임 중 최댓값 하나만 보는 기존 방식은 순간적으로
스친 상태에도 낚일 수 있다(예: "sleeping"이 우연히 30% 프레임에서
어느 정도 점수를 받음). 하지만 8개 샘플을 다 뜯어보니 이런 "진짜 상태
전환"은 cat_string_toy 하나뿐이었다.

이번 방식: "이 프레임 하나만 보면 후보 행동들 중 뭐가 제일 그럴듯한가"를
5개 프레임 각각에서 물어(투표), 5표 중 다수를 차지한 쪽을 "영상 전체의
지배적 행동"으로 판정한다. 최댓값(가장 그럴듯했던 한 순간)이 아니라
"대부분의 순간에 어느 쪽이 더 그럴듯한가"를 보는 것이라, 순간적으로
스친 상태 하나에 전체 판정이 휘둘리지 않는다.

방법: 정답 캡션과 오답(행동변경) 캡션에서 서로 다른 동사만 추출한다
(swapped_out=정답에만 있는 동사, swapped_in=오답에만 있는 동사). 5개
프레임 각각에서 두 그룹의 최고점수를 비교해 그 프레임의 승자를 정하고,
5표를 다수결로 집계한다.

cooking_thermometer, van_driving은 오답 문장의 바뀐 동사가 VBG(현재분사)
형태가 아니라서("is parked"는 VBN) 애초에 비교할 가짜 행동 후보가 없다 -
이 두 샘플은 이번 실험에서 원천적으로 테스트 불가능하다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from frame_extract import extract_frames
from blip_itm import compute_itm_score
from action_check import extract_actions, action_to_query
from style_detect import detect_video_style

ROOT = Path(__file__).resolve().parent.parent
CAPTIONS_JSON = ROOT / "data" / "captions.json"
VIDEO_DIR = ROOT / "data" / "videos"
OUTPUT_CSV = ROOT / "outputs" / "stage15_action_majority_vote.csv"

FRACTIONS = [10, 30, 50, 70, 90]
FOCUS_SAMPLES = ["cat_string_toy", "horse_herd"]  # 이번에 집중적으로 볼 샘플
ALL_TEST_SAMPLES = ["cat_string_toy", "dog_park", "horse_herd", "guitar_play", "people_dancing", "bicycle_ride"]


def get_swap_verbs(correct_caption: str, wrong_caption_action: str):
    correct_verbs = {w for w, _ in extract_actions(correct_caption)}
    wrong_verbs = {w for w, _ in extract_actions(wrong_caption_action)}
    return sorted(correct_verbs - wrong_verbs), sorted(wrong_verbs - correct_verbs)


def main():
    with open(CAPTIONS_JSON, "r", encoding="utf-8") as f:
        samples = {s["id"]: s for s in json.load(f)["samples"]}

    rows = []
    for sid in ALL_TEST_SAMPLES:
        s = samples[sid]
        video = VIDEO_DIR / s["video_file"]

        true_verbs, false_verbs = get_swap_verbs(s["correct_caption"], s["wrong_caption_action"])
        if not true_verbs or not false_verbs:
            print(f"[{sid}] 비교할 진짜/가짜 행동 동사가 없어 테스트 불가 (true={true_verbs}, false={false_verbs})")
            continue

        style = detect_video_style(video)["style"]
        frames = extract_frames(video, n_frames=5)

        is_focus = sid in FOCUS_SAMPLES
        print(f"\n{'='*90}\n[{sid}] 진짜후보={true_verbs}  가짜후보={false_verbs}  스타일={style}\n{'='*90}")

        votes_true = 0
        votes_false = 0
        for frac, frame in zip(FRACTIONS, frames):
            true_scores = {v: compute_itm_score(frame, action_to_query(v, style=style)) for v in true_verbs}
            false_scores = {v: compute_itm_score(frame, action_to_query(v, style=style)) for v in false_verbs}

            best_true_verb = max(true_scores, key=true_scores.get)
            best_false_verb = max(false_scores, key=false_scores.get)
            best_true_score = true_scores[best_true_verb]
            best_false_score = false_scores[best_false_verb]

            frame_winner = "TRUE" if best_true_score > best_false_score else "FALSE"
            votes_true += int(frame_winner == "TRUE")
            votes_false += int(frame_winner == "FALSE")

            if is_focus:
                print(f"  [{frac:3d}%] 진짜 최고: {best_true_verb}={best_true_score:.4f}  |  "
                      f"가짜 최고: {best_false_verb}={best_false_score:.4f}  -> 이 프레임 승자: {frame_winner}")

            rows.append({
                "id": sid, "frac": frac, "true_verb": best_true_verb, "true_score": round(best_true_score, 4),
                "false_verb": best_false_verb, "false_score": round(best_false_score, 4), "frame_winner": frame_winner,
            })

        majority_winner = "TRUE(정답)" if votes_true > votes_false else "FALSE(오답)"
        print(f"  => 5표 중 진짜 {votes_true}표 vs 가짜 {votes_false}표  ->  최종 판정: {majority_winner}")

    df = pd.DataFrame(rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 90)
    print("전체 요약: 샘플별 다수결 결과")
    print("=" * 90)
    summary = df.groupby("id")["frame_winner"].apply(lambda x: (x == "TRUE").sum()).reset_index(name="TRUE표")
    summary["FALSE표"] = 5 - summary["TRUE표"]
    summary["최종판정"] = summary["TRUE표"].apply(lambda t: "정답 승리 (O)" if t > 2 else ("오답 승리 (X, 문제)" if t < 2 else "동률(2:2? 불가능)"))
    print(summary.to_string(index=False))

    print(f"\n결과 CSV 저장 위치: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
