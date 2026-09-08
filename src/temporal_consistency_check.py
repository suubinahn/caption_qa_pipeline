"""
캡션의 "타임라인 일관성" 평가 - 새로운 평가축
==================================================

[문제의식]
VLM은 영상을 보고 캡션을 쓸 때, 실제 사건이 일어난 시간 순서를 무시하고
"장면 전체를 종합해서" 캡션을 쓰는 경향이 있다. 예를 들어 캡션 후반부에
쓴 내용이 실제로는 영상 초반에 일어난 일일 수도 있다. 지금까지 만든
검증 도구들은 "이 문장이 영상 어딘가에 있는 내용인가"만 확인했지,
"캡션에 적힌 순서와 영상 속 시간 순서가 맞는가"는 전혀 안 봤다.

[측정 방법]
1) 캡션을 문장 단위로 쪼갠다 (split_sentences - 이미 검증된 방식).
2) 영상에서 N개 프레임을 시간순으로 균등 추출한다 (extract_frames).
3) 각 문장이 N개 프레임 중 어떤 프레임과 가장 잘 맞는지(BLIP-ITM 최고점)
   찾는다 -> 그 문장의 "매칭된 시간대"로 삼는다.
4) "캡션에 문장이 등장하는 순서"와 "매칭된 시간대 순서"가 얼마나
   일치하는지 스피어만 상관계수(Spearman correlation)로 잰다.
     - +1에 가까움: 캡션 순서 = 영상 시간 순서 (타임라인을 잘 지킴)
     - 0 근처: 순서에 아무 관련 없음 (뒤죽박죽)
     - -1에 가까움: 캡션이 영상 시간 순서를 거�라로 서술함

[한계 - 미리 밝혀둘 것]
- 문장 하나가 여러 프레임에 걸친 사건을 설명할 수도 있고, 애초에 "장면을
  종합 묘사"하는 문장(예: "전체적으로 조용한 장면이다")은 특정 시간대에
  매칭되는 게 부자연스러울 수 있다 - 이런 문장은 노이즈를 만든다.
- 프레임을 N개만 보므로, 그보다 촘촘한 시간 단위의 사건 순서는 못 잡는다.
- 이건 "얼마나 자주 맞는지"를 검증한 표준 지표가 아니라 이번에 새로 설계한
  지표라, 실제로 유용한지는 여러 캡션에 적용해보며 계속 검증해야 한다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from frame_extract import extract_frames
from blip_itm import compute_itm_score
from validate_sentence_split_recall import split_sentences


def temporal_consistency_check(video_path, caption: str, n_frames: int = 10) -> dict:
    """캡션의 문장 순서와, 각 문장이 매칭되는 영상 내 시간 순서가
    얼마나 일치하는지 측정한다.
    """
    sentences = split_sentences(caption)
    frames = extract_frames(video_path, n_frames=n_frames)
    # extract_frames는 10%~90% 구간을 균등 분할한다 (frame_extract.py 참고)
    timestamps = np.linspace(0.10, 0.90, n_frames)

    per_sentence = []
    for i, sentence in enumerate(sentences):
        scores = [compute_itm_score(f, sentence) for f in frames]
        best_frame_idx = int(np.argmax(scores))
        per_sentence.append({
            "caption_order": i,
            "sentence": sentence,
            "best_frame_idx": best_frame_idx,
            "best_timestamp_pct": round(timestamps[best_frame_idx] * 100, 1),
            "best_score": round(scores[best_frame_idx], 4),
        })

    caption_order = [s["caption_order"] for s in per_sentence]
    matched_time_order = [s["best_frame_idx"] for s in per_sentence]

    if len(sentences) < 3:
        correlation, p_value = float("nan"), float("nan")
    else:
        correlation, p_value = spearmanr(caption_order, matched_time_order)

    return {
        "n_sentences": len(sentences),
        "per_sentence": per_sentence,
        "temporal_correlation": correlation,
        "p_value": p_value,
    }


if __name__ == "__main__":
    import csv
    ROOT = Path(__file__).resolve().parent.parent
    rows = {r["id"]: r for r in csv.DictReader(open(ROOT / "results" / "pilot_result.csv", encoding="utf-8-sig"))}

    for vid_id, video_file in [("example_01", "1.mp4"), ("example_06", "6.mp4")]:
        caption = rows[vid_id]["caption"]
        video_path = ROOT / "data" / "pilot_videos" / video_file

        print(f"\n{'='*90}\n{vid_id}\n{'='*90}")
        result = temporal_consistency_check(video_path, caption, n_frames=10)

        for s in result["per_sentence"]:
            print(f"  문장 {s['caption_order']+1}: (매칭 시점 {s['best_timestamp_pct']:>5.1f}%, score={s['best_score']:.3f}) "
                  f"\"{s['sentence'][:70]}...\"")

        print(f"\n타임라인 일관성(스피어만 상관계수): {result['temporal_correlation']:.3f}  (p={result['p_value']:.3f})")
        if result["temporal_correlation"] > 0.5:
            print("  -> 캡션 순서가 영상 시간 순서와 대체로 일치함")
        elif result["temporal_correlation"] < -0.5:
            print("  -> 캡션이 시간 순서를 거꾸로 서술하는 경향")
        else:
            print("  -> 캡션 순서와 영상 시간 순서 사이에 뚜렷한 관계 없음 (종합 묘사형일 수 있음)")
