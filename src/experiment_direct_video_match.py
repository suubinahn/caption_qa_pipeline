"""
캡션을 거치지 않는 직접 매칭 실험 (검색어 <-> 영상 프레임, CLIP)
======================================================================

지금 파이프라인은 검색어를 캡션 문장과 비교(①)하고, 그 문장이 실제
영상과 맞는지 별도로 검증(②)한다 - 캡션을 항상 중간에 거친다. NVIDIA
Cosmos Dataset Search 비교 분석 중 "캡션 없이 검색어를 영상에 직접
매칭하면 결과가 얼마나 달라지는가"가 궁금해져서 확인해본다.

1단계(캡션 사실검증) 때 만들어둔 clip_score.py의 compute_similarity()
(이미지 1장 + 문장 1개 -> CLIP 코사인 유사도)를 그대로 재사용한다 -
새로 만들 게 거의 없다. 클립당 프레임 5장(n_frames=5, 프로덕션과 동일)을
뽑아 검색어와 직접 비교하고, 최댓값을 그 클립의 "직접 매칭 점수"로
쓴다(지금 파이프라인 전체에서 쓰는 것과 같은 max-pooling 패턴).

이건 지금 목표(캡션 신뢰도 검증)에 직접 기여하지 않는 곁가지 실험이다 -
"캡션 경유 방식이 실제로 더 나은가"를 정량적으로 확인하는 참고용.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from clip_score import compute_similarity
from frame_extract import extract_frames
from search_module import CaptionSearchIndex, VALIDATED_ANCHOR_QUERIES

N_FRAMES = 5
EVAL_K = 10

TEST_CASES = [
    ("The vehicle is stopped.", "motion", "stop"),
    ("The scene is at night.", "time_of_day", "nighttime"),
    ("It is raining.", "weather", "rainy"),
    ("The car drives through a tunnel.", "road_context", "tunnel"),
]


def precision_at_k(ranked_clip_dirs: list[str], is_relevant, k: int) -> float:
    top = ranked_clip_dirs[:k]
    return sum(is_relevant(cd) for cd in top) / k


def main():
    with open("outputs/search_full_dataset.json", encoding="utf-8") as f:
        clips = json.load(f)
    clip_dirs = [c["clip_dir"] for c in clips]
    info_by_dir = {c["clip_dir"]: c for c in clips}

    print(f"①검색용 인덱스 빌드 중... ({len(clip_dirs)}개 클립)")
    index = CaptionSearchIndex(clip_dirs, anchor_queries=VALIDATED_ANCHOR_QUERIES)
    print("빌드 완료.\n")

    # 클립당 프레임을 한 번만 뽑아서 모든 쿼리가 재사용한다 (쿼리마다 다시 뽑지 않음)
    print(f"프레임 추출 중... ({len(clip_dirs)}개 클립 x {N_FRAMES}장)")
    t0 = time.time()
    frames_by_clip: dict[str, list] = {}
    for i, clip_dir in enumerate(clip_dirs):
        video_path = Path(clip_dir) / "1_clip" / "5.mp4"
        if video_path.exists():
            frames_by_clip[clip_dir] = extract_frames(str(video_path), n_frames=N_FRAMES)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(clip_dirs)}  (경과 {(time.time()-t0)/60:.1f}분)")
    print(f"프레임 추출 완료 ({(time.time()-t0)/60:.1f}분, {len(frames_by_clip)}개 클립에 영상 있음)\n")

    for query, field, value in TEST_CASES:
        def is_relevant(cd, field=field, value=value):
            return info_by_dir[cd].get(field) == value

        # ① 캡션 경유 방식 (기존 파이프라인, CSLS 적용)
        caption_results = index.search(query, top_k=None, use_csls=True)
        caption_ranked = [r.clip_dir for r in caption_results]
        caption_p = precision_at_k(caption_ranked, is_relevant, EVAL_K)

        # 직접 매칭 방식 (캡션 안 거침, CLIP으로 프레임과 검색어 바로 비교)
        t0 = time.time()
        direct_scores = []
        for cd, frames in frames_by_clip.items():
            score = max(compute_similarity(f, query) for f in frames)
            direct_scores.append((cd, score))
        direct_scores.sort(key=lambda x: x[1], reverse=True)
        direct_ranked = [cd for cd, _ in direct_scores]
        direct_p = precision_at_k(direct_ranked, is_relevant, EVAL_K)
        elapsed = time.time() - t0

        overlap = len(set(caption_ranked[:EVAL_K]) & set(direct_ranked[:EVAL_K]))

        print(f'검색어: "{query}"  (기준: {field}={value})')
        print(f"  ① 캡션 경유    precision@{EVAL_K}: {caption_p*100:5.1f}%")
        print(f"  직접 매칭(CLIP) precision@{EVAL_K}: {direct_p*100:5.1f}%  (계산 {elapsed:.1f}초)")
        print(f"  상위 {EVAL_K}개 중 두 방식이 겹치는 클립: {overlap}개\n")


if __name__ == "__main__":
    main()
