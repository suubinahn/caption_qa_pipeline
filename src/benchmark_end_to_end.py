"""
통합 파이프라인(search_and_verify) 종단간(end-to-end) 벤치마크
====================================================================

지금까지 ①(검색 precision@K)과 ②(match_score 참/거짓 margin)를 각각
따로 검증했지만, "②로 필터링하면 실제 검색 결과 품질이 좋아지는가"라는
통합 효과는 측정한 적이 없었다.

방법: 9개 검증된 검색어 각각에 대해 top_k=n_positive(실험 전반에서 써온
precision@K와 동일 정의)로 search_and_verify()를 실행한 뒐, 결과를
is_video_verified 여부로 나눠서 각각의 precision(=info.txt 기준 실제
관련 있는 비율)을 비교한다.

가설: is_video_verified=True인 결과가 False/None인 결과보다 relevant
비율이 높다면, ②가 실제로 "캡션이 그럴듯해서 뽑혔지만 틀린 내용" 케이스를
걸러내는 데 기여하고 있다는 뜻. 반대로 차이가 없거나 낮다면, ②가 검색
품질 개선에는 별 도움이 안 된다는 뜻(단, ②는애초에 "관련성"이 아니라
"캡션 정확성"을 재는 것이므로 - night/daytime 사례처럼 - 큰 개선이
없더라도 그 자체로 ②가 무의미하다는 뜻은 아니다. 각각 재는 게 다르다).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))
from search_module import CaptionSearchIndex, VALIDATED_ANCHOR_QUERIES
from search_and_verify import search_and_verify

SAMPLE_JSON = Path("outputs/search_sample_100.json")
OUTPUT_CSV = Path("outputs/e2e_benchmark_results.csv")
OUTPUT_SUMMARY_CSV = Path("outputs/e2e_benchmark_summary.csv")

QUERIES = [
    ("The vehicle is turning right.", "motion", "turning right"),
    ("The vehicle is turning left.", "motion", "turning left"),
    ("The vehicle is stopped.", "motion", "stop"),
    ("The car drives through a tunnel.", "road_context", "tunnel"),
    ("The car crosses a bridge.", "road_context", "bridge"),
    ("The scene is at night.", "time_of_day", "nighttime"),
    ("It is raining.", "weather", "rainy"),
    ("It is snowing.", "weather", "snowy"),
    ("The vehicle drives on a multi-lane highway.", "road_type", "multi-lane highway"),
]


def main():
    with open(SAMPLE_JSON, encoding="utf-8") as f:
        clips = json.load(f)
    clip_map = {c["clip_dir"]: c for c in clips}
    clip_dirs = list(clip_map.keys())

    print(f"검색 인덱스 빌드 중... ({len(clip_dirs)}개 클립)")
    index = CaptionSearchIndex(clip_dirs, anchor_queries=VALIDATED_ANCHOR_QUERIES)
    print("빌드 완료.\n")

    all_rows = []
    t0 = time.time()
    for qi, (query, field, val) in enumerate(QUERIES):
        n_positive = sum(1 for c in clips if c[field] == val)
        print(f"[{qi+1}/9] \"{query}\" (top_k={n_positive})")
        results = search_and_verify(index, query, top_k=n_positive, use_csls=True)
        for r in results:
            is_relevant_gt = clip_map[r.clip_dir][field] == val
            all_rows.append({
                "query": query, "clip_dir": r.clip_dir,
                "search_score": r.search_score, "is_video_verified": r.is_video_verified,
                "match_score": r.match_score, "is_relevant_gt": is_relevant_gt,
            })
        elapsed = time.time() - t0
        print(f"  누적 소요: {elapsed/60:.1f}분")

    df = pd.DataFrame(all_rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"\n{'='*90}")
    print(f"{'검색어':45s} {'전체 precision':>14s} {'verified 중 precision':>20s} {'unverified 중 precision':>22s}")
    summary_rows = []
    for query, field, val in QUERIES:
        sub = df[df["query"] == query]
        overall_prec = sub["is_relevant_gt"].mean()

        verified = sub[sub["is_video_verified"] == True]
        unverified = sub[sub["is_video_verified"] == False]

        v_prec = verified["is_relevant_gt"].mean() if len(verified) > 0 else float("nan")
        u_prec = unverified["is_relevant_gt"].mean() if len(unverified) > 0 else float("nan")

        summary_rows.append({
            "query": query, "n_total": len(sub), "overall_precision": overall_prec,
            "n_verified": len(verified), "verified_precision": v_prec,
            "n_unverified": len(unverified), "unverified_precision": u_prec,
        })
        v_str = f"{v_prec*100:.1f}%(n={len(verified)})" if len(verified) > 0 else "n/a"
        u_str = f"{u_prec*100:.1f}%(n={len(unverified)})" if len(unverified) > 0 else "n/a"
        print(f"{query:45s} {overall_prec*100:13.1f}% {v_str:>20s} {u_str:>22s}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    print(f"\n결과 저장: {OUTPUT_CSV}, {OUTPUT_SUMMARY_CSV}")
    print(f"총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
