"""
검색 관련성(retrieval) 실험 - 진짜 목적에 맞춘 첫 테스트
============================================================

목적: "이 검색어를 넣으면, 실제로 관련 있는 영상이 상위에 뜨는가?"를
확인한다. 캡션을 거치지 않고 검색어를 영상 프레임과 직접 비교한다
(캡션 정확도와 무관하게, 순수하게 "이 영상 내용 자체가 이 검색어와
얼마나 관련 있는가"를 보기 위함).

방법: 영상마다 5프레임을 뽑아 캐싱해두고, 검색어 5개 각각에 대해
10개 영상 전부와 BLIP-ITM 최대값을 구해 순위를 매긴다.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from frame_extract import extract_frames
from blip_itm import compute_itm_score

ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = ROOT / "data" / "pilot_videos"
PILOT_CSV = ROOT / "results" / "pilot_result.csv"
OUTPUT_CSV = ROOT / "outputs" / "search_relevance_results.csv"

QUERIES = [
    "A pedestrian is crossing the street.",
    "The vehicle is driving in heavy rain.",
    "The car is turning right at an intersection.",
    "Bright sunny weather with clear skies.",
    "A bus is visible on the road.",
]


def main():
    with open(PILOT_CSV, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    video_ids = [r["id"] for r in rows]
    video_paths = {r["id"]: r["video_path"] for r in rows}

    print("프레임 추출 중 (10개 영상)...")
    frame_cache = {}
    for vid_id in video_ids:
        frame_cache[vid_id] = extract_frames(video_paths[vid_id], n_frames=5)
    print("완료.\n")

    all_scores = {q: {} for q in QUERIES}
    for q in QUERIES:
        print(f"[검색어] \"{q}\"")
        for vid_id in video_ids:
            frames = frame_cache[vid_id]
            score = max(compute_itm_score(f, q) for f in frames)
            all_scores[q][vid_id] = score
        ranked = sorted(all_scores[q].items(), key=lambda x: -x[1])
        for rank, (vid_id, score) in enumerate(ranked, 1):
            print(f"  {rank:2d}위. {vid_id}  {score:.4f}")
        print()

    df = pd.DataFrame(all_scores)
    df.index.name = "video_id"
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, encoding="utf-8-sig")
    print(f"결과 CSV: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
