"""
관련성 피드백(Rocchio 쿼리 재조정) 정식 검증
====================================================

experiment_relevance_feedback.py의 예비 실험(쿼리 4개, ALPHA=1.0 고정,
1회 시행)에서 강한 개선 신호(+80~90%p)를 봤다. 이 스크립트는 그걸
더 정식으로 검증한다:

  1) 예비 실험 4개가 아니라, 9개 검증 앵커 쿼리 전부(각각 info.txt
     필드/값으로 자동 라벨링 가능)로 확장
  2) ALPHA(피드백 강도)를 여러 값으로 시험해서 민감도를 확인하고
     기본값을 정한다
  3) 희귀 카테고리(좌회전 8개, 다리 8개, 눈 8개처럼 전체 표본이 아주
     적은 경우)는 "못 본 나머지 중 진짜 관련 있는 게 몇 개나 남았는지"
     (ceiling)를 같이 보여줘서, precision@10이 낮게 나와도 그게 기법
     실패인지 단순히 찾을 게 별로 안 남아서인지 구분할 수 있게 한다

CSLS 적용(프로덕션 조합, use_csls=True)만 검증한다 - 이게 실제 배포될
조합이기 때문이다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from search_module import CaptionSearchIndex, VALIDATED_ANCHOR_QUERIES

N_SEEN = 10
EVAL_K = 10
ALPHAS = [0.5, 1.0, 1.5, 2.0]

# (검색어, info.txt 필드, 값, 전체 데이터셋 내 총 개수) - 총 개수는
# ceiling 계산·해석용으로 미리 적어둠 (search_full_dataset.json 실측)
TEST_CASES = [
    ("The vehicle is turning right.", "motion", "turning right", 15),
    ("The vehicle is turning left.", "motion", "turning left", 8),
    ("The vehicle is stopped.", "motion", "stop", 126),
    ("The car drives through a tunnel.", "road_context", "tunnel", 16),
    ("The car crosses a bridge.", "road_context", "bridge", 8),
    ("The scene is at night.", "time_of_day", "nighttime", 24),
    ("It is raining.", "weather", "rainy", 240),
    ("It is snowing.", "weather", "snowy", 8),
    ("The vehicle drives on a multi-lane highway.", "road_type", "multi-lane highway", 216),
]


def rank_clips(index: CaptionSearchIndex, q_feat: torch.Tensor, csls_k: int = 10) -> list[tuple[str, float, int]]:
    sims = (index._embeds @ q_feat).cpu().numpy()
    k = min(csls_k, len(sims))
    r_T = float(sims[sims.argsort()[-k:]].mean())
    sims = 2 * sims - r_T - index._r_S

    best_per_clip: dict[str, tuple[float, int]] = {}
    for i, (sim, clip_dir) in enumerate(zip(sims, index._owner)):
        sim = float(sim)
        if clip_dir not in best_per_clip or sim > best_per_clip[clip_dir][0]:
            best_per_clip[clip_dir] = (sim, i)
    return sorted(((cd, s, i) for cd, (s, i) in best_per_clip.items()), key=lambda x: x[1], reverse=True)


def precision_at_k(ranked, is_relevant, k: int) -> float:
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

    # alpha별 결과를 모아서 마지막에 요약(평균 개선폭)까지 낸다
    alpha_deltas: dict[float, list[float]] = {a: [] for a in ALPHAS}

    for query, field, value, total_count in TEST_CASES:
        def is_relevant(clip_dir, field=field, value=value):
            return info_by_dir[clip_dir].get(field) == value

        q_feat = index._embed_texts([query])[0]
        ranked_before = rank_clips(index, q_feat)

        seen = ranked_before[:N_SEEN]
        seen_dirs = {cd for cd, _, _ in seen}
        unseen_before = [(cd, s, i) for cd, s, i in ranked_before if cd not in seen_dirs]

        n_seen_pos = sum(is_relevant(cd) for cd, _, _ in seen)
        n_remaining_pos = total_count - n_seen_pos
        ceiling = min(1.0, n_remaining_pos / EVAL_K)
        baseline_p = precision_at_k(unseen_before, is_relevant, EVAL_K)

        pos_idxs = [i for cd, _, i in seen if is_relevant(cd)]
        neg_idxs = [i for cd, _, i in seen if not is_relevant(cd)]

        print(f'검색어: "{query}"  (기준: {field}={value}, 전체 {total_count}개)')
        print(f"  상위 {N_SEEN}개 중 관련있음={len(pos_idxs)}개 -> 못 본 나머지 중 관련있음 최대 {n_remaining_pos}개 "
              f"(ceiling precision@{EVAL_K}={ceiling*100:.0f}%)")
        print(f"  피드백 전 precision@{EVAL_K}: {baseline_p*100:.1f}%")

        if not pos_idxs or not neg_idxs:
            print("  -> 라벨이 한쪽으로 쏠려 피드백 방향 계산 불가 (건너뜀)\n")
            continue

        pos_mean = index._embeds[pos_idxs].mean(dim=0)
        neg_mean = index._embeds[neg_idxs].mean(dim=0)
        direction = pos_mean - neg_mean

        for alpha in ALPHAS:
            new_q = q_feat + alpha * direction
            ranked_after = rank_clips(index, new_q)
            unseen_after = [(cd, s, i) for cd, s, i in ranked_after if cd not in seen_dirs]
            after_p = precision_at_k(unseen_after, is_relevant, EVAL_K)
            delta = after_p - baseline_p
            alpha_deltas[alpha].append(delta)
            print(f"    alpha={alpha:.1f}: {after_p*100:5.1f}%  ({delta*100:+.1f}%p, ceiling {ceiling*100:.0f}%)")
        print()

    print("=" * 60)
    print("alpha별 평균 개선폭 (9개 쿼리 중 계산 가능했던 것만 평균)")
    for alpha, deltas in alpha_deltas.items():
        if deltas:
            avg = sum(deltas) / len(deltas)
            print(f"  alpha={alpha:.1f}: 평균 {avg*100:+.1f}%p  (n={len(deltas)})")


if __name__ == "__main__":
    main()
