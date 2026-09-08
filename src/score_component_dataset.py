"""
객체/행동 단위 임계값 재보정 - 3단계: 전체 채점 + 분포 확인
================================================================

build_component_dataset.py로 만든 124개 (영상, 단어, 진짜/가짜) 쌍을
전부 inspect_caption()으로 채점한다. 영상마다 스타일(photo/cartoon/
illustration)을 한 번만 판별해서 캐싱해두고, 같은 영상 안의 모든 단어
채점에 재사용한다 (9단계에서 만든 스타일 인식 템플릿 그대로 적용).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from build_component_dataset import build_dataset
from inspector import inspect_caption
from object_check import noun_to_query
from action_check import action_to_query
from style_detect import detect_video_style

ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = ROOT / "data" / "videos"
OUTPUT_CSV = ROOT / "outputs" / "component_dataset_scored.csv"


def main():
    df = build_dataset()

    # 영상별 스타일을 한 번만 판별해서 캐싱 (video_file -> style)
    style_cache = {}
    for video_file in df["video_file"].unique():
        style_info = detect_video_style(VIDEO_DIR / video_file)
        style_cache[video_file] = style_info["style"]
        print(f"[style] {video_file}: {style_info['style']}  (투표: {style_info['vote_counts']})")

    scores = []
    for i, row in df.iterrows():
        video = VIDEO_DIR / row["video_file"]
        style = style_cache[row["video_file"]]

        if row["type"] == "action":
            query = action_to_query(row["word"], style=style)
        else:
            query = noun_to_query(row["word"], row["tag"], style=style)

        result = inspect_caption(video, query)
        scores.append(result["score"])
        print(f"  [{row['id']:20s}][{row['type']:6s}] {row['word']:12s} ({'TRUE ' if row['is_true'] else 'FALSE'}) "
              f"-> \"{query}\" = {result['score']:.4f}")

    df["style"] = df["video_file"].map(style_cache)
    df["score"] = scores

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n저장 위치: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
