"""
관련성 피드백(Rocchio 스타일 쿼리 재조정) 효과 검증 실험
================================================================

NVIDIA Cosmos Dataset Search의 "linear probe query learner"(사용자가 결과에
관련있음/없음을 표시하면 쿼리 임베딩을 그 방향으로 조정)를 우리 파이프라인
임베딩 공간에도 적용할 수 있는지 확인한다. 정보검색의 고전 기법인 Rocchio
알고리즘과 사실상 동일한 아이디어다.

사람이 직접 라벨을 다는 대신, info.txt(실제 정답)로 라벨을 자동 생성해서
"이 기법이 우리 데이터에서 실제로 순위를 개선하는가"부터 저비용으로
검증한다 - 이 프로젝트에서 계속 써온 것과 동일한 자동 검증 방식이다.

방법:
  1) CSLS 없이(순수 코사인) 검색해서 전체 순위를 매긴다
     (Rocchio는 임베딩 벡터 자체를 조정하는 기법이라, 점수 공간에서
     동작하는 CSLS와 분리해서 이 기법 자체의 효과만 먼저 본다)
  2) 상위 N_SEEN개를 "사용자가 이미 본 결과"로 간주하고, info.txt로
     관련있음/없음을 자동 라벨링한다
  3) 라벨링된 클립들의 문장 임베딩으로 "관련있음 평균 - 관련없음 평균"
     방향을 구해서 쿼리 벡터에 더한다 (Rocchio 공식)
  4) 조정된 쿼리로 나머지(못 본) 클립들만 다시 순위를 매겨서, precision@10이
     원래보다 개선됐는지 비교한다 - "이미 본 것 중에서" 재현하는 게 아니라
     "아직 못 찾은 관련 클립을 더 찾아내는가"가 진짜 검증 포인트이므로,
     라벨링에 쓴 클립은 평가에서 제외한다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from search_module import CaptionSearchIndex, VALIDATED_ANCHOR_QUERIES

N_SEEN = 10  # "사용자가 본" 상위 결과 개수 (피드백 라벨링에 씀)
EVAL_K = 10  # 못 본 나머지 클립들 중 precision@K
ALPHA = 1.0  # Rocchio 조정 강도


def rank_clips(index: CaptionSearchIndex, q_feat: torch.Tensor, use_csls: bool = False, csls_k: int = 10) -> list[tuple[str, float, int]]:
    """(clip_dir, score, sentence_index) 리스트를 점수 내림차순으로 반환.
    클립별 최고 문장만 남기는 max-pooling은 search()와 동일한 방식.
    use_csls=True면 search()와 동일한 공식(2*cos - r_T - r_S)으로 보정한다 -
    r_S는 인덱스 빌드 시 이미 계산돼 있고, r_T는 이 쿼리 자신의 상위 k개
    유사도 평균이라 조정된 쿼리 벡터에 맞게 매번 다시 계산해야 한다."""
    sims = (index._embeds @ q_feat).cpu().numpy()
    if use_csls:
        k = min(csls_k, len(sims))
        r_T = float(sims[sims.argsort()[-k:]].mean())
        sims = 2 * sims - r_T - index._r_S
    best_per_clip: dict[str, tuple[float, int]] = {}
    for i, (sim, clip_dir) in enumerate(zip(sims, index._owner)):
        sim = float(sim)
        if clip_dir not in best_per_clip or sim > best_per_clip[clip_dir][0]:
            best_per_clip[clip_dir] = (sim, i)
    ranked = sorted(((cd, s, i) for cd, (s, i) in best_per_clip.items()), key=lambda x: x[1], reverse=True)
    return ranked


def precision_at_k(ranked: list[tuple[str, float, int]], is_relevant, k: int) -> float:
    top = ranked[:k]
    return sum(is_relevant(cd) for cd, _, _ in top) / k


def main():
    with open("outputs/search_full_dataset.json", encoding="utf-8") as f:
        clips = json.load(f)
    clip_dirs = [c["clip_dir"] for c in clips]
    info_by_dir = {c["clip_dir"]: c for c in clips}

    print(f"인덱스 빌드 중... ({len(clip_dirs)}개 클립)")
    index = CaptionSearchIndex(clip_dirs, anchor_queries=VALIDATED_ANCHOR_QUERIES)
    print("빌드 완료.\n")

    test_cases = [
        ("It is raining.", "weather", "rainy"),
        ("The scene is at night.", "time_of_day", "nighttime"),
        ("The vehicle is stopped.", "motion", "stop"),
        ("The vehicle is turning left.", "motion", "turning left"),
    ]

    for query, field, value in test_cases:
        def is_relevant(clip_dir, field=field, value=value):
            return info_by_dir[clip_dir].get(field) == value

        q_feat = index._embed_texts([query])[0]
        print(f'검색어: "{query}"  (기준: {field}={value})')

        for use_csls in (False, True):
            label = "CSLS 없음(순수 코사인)" if not use_csls else "CSLS 적용(프로덕션 조합)"
            ranked_before = rank_clips(index, q_feat, use_csls=use_csls)

            seen = ranked_before[:N_SEEN]
            seen_dirs = {cd for cd, _, _ in seen}
            unseen_before = [(cd, s, i) for cd, s, i in ranked_before if cd not in seen_dirs]
            baseline_p = precision_at_k(unseen_before, is_relevant, EVAL_K)

            pos_idxs = [i for cd, _, i in seen if is_relevant(cd)]
            neg_idxs = [i for cd, _, i in seen if not is_relevant(cd)]

            print(f"  [{label}] 상위 {N_SEEN}개 중 관련있음={len(pos_idxs)}개, 관련없음={len(neg_idxs)}개")

            if not pos_idxs or not neg_idxs:
                print("    -> 상위 결과가 전부 한쪽으로 쏠려서 피드백 방향을 계산할 수 없음 (건너뜀)")
                continue

            pos_mean = index._embeds[pos_idxs].mean(dim=0)
            neg_mean = index._embeds[neg_idxs].mean(dim=0)
            direction = pos_mean - neg_mean

            new_q = q_feat + ALPHA * direction
            ranked_after_full = rank_clips(index, new_q, use_csls=use_csls)
            unseen_after = [(cd, s, i) for cd, s, i in ranked_after_full if cd not in seen_dirs]
            after_p = precision_at_k(unseen_after, is_relevant, EVAL_K)

            delta = after_p - baseline_p
            print(f"    못 본 클립 중 precision@{EVAL_K}: 피드백 전 {baseline_p*100:.1f}% -> 피드백 후 {after_p*100:.1f}%  ({delta*100:+.1f}%p)")
        print()


if __name__ == "__main__":
    main()
