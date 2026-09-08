"""
주행영상 도메인 전용 객체 점수 분포 분석
============================================

기존 object_check.check_objects()는 명사 하나마다 inspect_caption()을
호출하고, 그 안에서 매번 extract_frames()로 영상을 다시 디코딩한다.
MSVD/Wikimedia의 짧은 클립(수 초~수십 초)에서는 이게 별문제 없었지만,
주행영상은 해상도가 높고(1920x1200) 캡션 하나에 명사가 30~40개씩
있어서, 그대로 돌리면 영상 1개당 디코딩만 수십~수백 번 반복돼 매우
느려진다(example_01 하나가 약 3~4분 소요 추정).

그래서 이 스크립트는 영상마다 프레임을 "딱 한 번만" 추출해 재사용하고,
그 프레임들에 대해 여러 단어를 채점하는 방식으로 다시 짰다 - 로직은
object_check.py와 완전히 동일하고(같은 noun_to_query, 같은 max-pooling),
단지 "영상 디코딩을 반복하지 않는다"는 효율성만 다르다.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from frame_extract import extract_frames
from blip_itm import compute_itm_score
from object_check import extract_nouns, noun_to_query
from style_detect import detect_video_style, classify_frame_style

ROOT = Path(__file__).resolve().parent.parent
PILOT_CSV = ROOT / "results" / "pilot_result.csv"
OUTPUT_CSV = ROOT / "outputs" / "driving_domain_object_scores.csv"


def main():
    with open(PILOT_CSV, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    all_results = []
    for row in rows:
        vid_id = row["id"]
        video_path = Path(row["video_path"])
        caption = row["caption"]
        if not video_path.exists() or not caption:
            print(f"[skip] {vid_id}: 영상 없음 또는 캡션 없음")
            continue

        print(f"\n[{vid_id}] 프레임 추출 중...")
        frames = extract_frames(video_path, n_frames=5)

        # 스타일은 프레임별로 다시 분류(이미 뽑은 프레임 재사용, detect_video_style처럼 다수결)
        from collections import Counter
        votes = Counter(classify_frame_style(f)[0] for f in frames)
        style = votes.most_common(1)[0][0]

        nouns = extract_nouns(caption)
        print(f"  스타일={style}, 명사 {len(nouns)}개 채점 중...")

        for word, tag in nouns:
            query = noun_to_query(word, tag, style=style)
            scores = [compute_itm_score(f, query) for f in frames]
            best_score = max(scores)
            all_results.append({
                "video_id": vid_id, "word": word, "tag": tag,
                "query": query, "score": round(best_score, 4),
            })

    df = pd.DataFrame(all_results)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    scores = df["score"].values
    print("\n" + "=" * 90)
    print(f"주행영상 전체 객체 점수 분포 (n={len(scores)}, {df['video_id'].nunique()}개 영상)")
    print("=" * 90)
    print(f"min={scores.min():.4f}  max={scores.max():.4f}  mean={scores.mean():.4f}  median={np.median(scores):.4f}")
    for p in [5, 10, 25, 50, 75, 90]:
        print(f"  {p}th percentile: {np.percentile(scores, p):.4f}")

    print(f"\n기존 threshold(0.0214) 미만 개수: {(scores < 0.0214).sum()} / {len(scores)} ({(scores < 0.0214).mean()*100:.1f}%)")
    print(f"OK 등급(>=0.15) 개수: {(scores >= 0.15).sum()} / {len(scores)} ({(scores>=0.15).mean()*100:.1f}%)")

    print(f"\n결과 CSV 저장 위치: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
