"""
검색 관련성 대규모 실험 (100클립 x 9검색어) - 회사 실데이터 기반
======================================================================

outputs/search_sample_100.json (희귀 카테고리 전량 + 무작위 채움, 100개)의
각 클립에 대해 9개 검색어를 프레임 직접비교(캡션 없이) 방식으로 채점한다.

info.txt의 구조화 태그를 "근사 정답"으로 삼아 정밀도/재현율을 계산한다.
단, info.txt/캡션도 완벽한 정답은 아닐 수 있다는 점을 감안해, 결과가
크게 어긋나는 경우는 별도로 표시해 나중에 직접 영상을 확인할 수 있게 한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from frame_extract import extract_frames
from blip_itm import compute_itm_score

SAMPLE_JSON = Path("outputs/search_sample_100.json")
OUTPUT_CSV = Path("outputs/search_largescale_results.csv")
OUTPUT_SUMMARY_CSV = Path("outputs/search_largescale_summary.csv")

# (검색어, info.txt 필드, 해당 값) - 이 필드=값인 클립이 "정답"
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

N_FRAMES = 5


def main():
    with open(SAMPLE_JSON, encoding="utf-8") as f:
        clips = json.load(f)

    print(f"클립 {len(clips)}개, 검색어 {len(QUERIES)}개 실험 시작\n")

    # --- 1) 프레임 추출 (클립당 1회, 이후 모든 검색어에 재사용) ---
    frame_cache = {}
    for i, c in enumerate(clips):
        video_path = Path(c["clip_dir"]) / "1_clip" / "5.mp4"
        if not video_path.exists():
            print(f"[{i+1}/{len(clips)}] 영상 없음: {video_path}")
            continue
        try:
            frame_cache[c["clip_dir"]] = extract_frames(str(video_path), n_frames=N_FRAMES)
        except Exception as e:
            print(f"[{i+1}/{len(clips)}] 프레임 추출 실패: {video_path} ({e})")
        if (i + 1) % 20 == 0:
            print(f"  프레임 추출 진행: {i+1}/{len(clips)}")

    print(f"\n프레임 추출 완료: {len(frame_cache)}/{len(clips)}개 성공\n")

    # --- 2) 검색어별 채점 ---
    all_rows = []
    for qi, (query, field, val) in enumerate(QUERIES):
        print(f"[검색어 {qi+1}/{len(QUERIES)}] \"{query}\" ({field}={val})")
        for c in clips:
            if c["clip_dir"] not in frame_cache:
                continue
            frames = frame_cache[c["clip_dir"]]
            score = max(compute_itm_score(f, query) for f in frames)
            is_positive = (c[field] == val)
            all_rows.append({
                "query": query, "clip_dir": c["clip_dir"], "score": score,
                "is_positive": is_positive, "motion": c["motion"],
                "road_context": c["road_context"], "weather": c["weather"],
                "time_of_day": c["time_of_day"],
            })

    df = pd.DataFrame(all_rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    # --- 3) 검색어별 정밀도/재현율 요약 ---
    summary_rows = []
    print("\n" + "=" * 100)
    print("검색어별 결과 요약")
    print("=" * 100)
    for query, field, val in QUERIES:
        sub = df[df["query"] == query].sort_values("score", ascending=False).reset_index(drop=True)
        n_positive = sub["is_positive"].sum()
        n_total = len(sub)
        if n_positive == 0:
            continue

        # precision@K: K = 정답 개수(n_positive)로 설정 -> "정답 개수만큼 뽑았을 때 몇 개가 진짜 정답인가"
        k = n_positive
        top_k = sub.head(k)
        precision_at_k = top_k["is_positive"].sum() / k

        # 상위 10위 안에 정답이 몇 개 있는지도 참고용으로
        top10_hits = sub.head(10)["is_positive"].sum()

        print(f"\n[{query}] (정답 {n_positive}/{n_total})")
        print(f"  Precision@{k}: {precision_at_k*100:.1f}%  (상위 {k}개 중 진짜 정답 {top_k['is_positive'].sum()}개)")
        print(f"  상위 10위 안의 정답 개수: {top10_hits}/{min(10,n_positive)}")
        print(f"  상위 5개: {sub.head(5)[['clip_dir','score','is_positive']].to_string(index=False)}")

        summary_rows.append({
            "query": query, "field": field, "value": val,
            "n_positive": n_positive, "n_total": n_total,
            "precision_at_k": round(precision_at_k, 3),
            "top10_hits": int(top10_hits),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    print(f"\n\n전체 결과: {OUTPUT_CSV}")
    print(f"요약: {OUTPUT_SUMMARY_CSV}")


if __name__ == "__main__":
    main()
