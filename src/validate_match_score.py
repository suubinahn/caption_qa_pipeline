"""
②(캡션-영상 일치도) 검증 - 실제 회사 영상으로 참/거짓 판별력 확인
========================================================================

search_and_verify.py의 verify_caption_matches_video()가 실제로 캡션의
사실 여부를 구분하는지, 아니면 이 도메인에서 그냥 노이즈인지 확인한다.

방법 (1단계 stage 1의 "정답 vs 오답 margin" 방법론을, 합성 캡션이 아니라
실제 회사 영상+info.txt에 그대로 적용):
  1) 각 클립의 실제 info.txt 값으로 "참" 문장을 만든다 (예: motion=stop이면
     "The vehicle is stopped.")
  2) 같은 필드의 다른 값으로 "거짓" 문장을 만든다 (예: "The vehicle is
     turning right.")
  3) 같은 영상 프레임에 대해 두 문장의 BLIP-ITM 매칭 점수를 각각 구하고,
     margin = 참 점수 - 거짓 점수를 계산한다.
  4) margin이 유의미하게 양수면 -> match_score가 실제로 캡션 사실 여부를
     구분하는 신호. 0 근처거나 음수면 -> 이 도메인/필드에서는 못 믿을 신호.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

from scipy.stats import ttest_1samp, wilcoxon

import sys
sys.path.insert(0, str(Path(__file__).parent))
from search_and_verify import verify_caption_matches_video

FULL_DATASET_JSON = Path("outputs/search_full_dataset.json")
OUTPUT_CSV = Path("outputs/match_score_validation.csv")
N_CLIPS = 20
N_FRAMES = 3
SEED = 42

FIELD_TEMPLATES = {
    "motion": {
        "moving straight": "The vehicle is moving straight.",
        "turning right": "The vehicle is turning right.",
        "turning left": "The vehicle is turning left.",
        "stop": "The vehicle is stopped.",
        "curved lane driving": "The vehicle is driving through a curve.",
    },
    "weather": {
        "sunny": "The weather is sunny.",
        "rainy": "The weather is rainy.",
        "cloudy": "The weather is cloudy.",
        "snowy": "It is snowing.",
    },
    "time_of_day": {
        "daytime": "It is daytime.",
        "nighttime": "It is nighttime.",
    },
}


def main():
    with open(FULL_DATASET_JSON, encoding="utf-8") as f:
        all_clips = json.load(f)

    random.seed(SEED)
    sample = random.sample(all_clips, N_CLIPS)

    rows = []
    t0 = time.time()
    for i, c in enumerate(sample):
        video_path = Path(c["clip_dir"]) / "1_clip" / "5.mp4"
        if not video_path.exists():
            continue

        for field, templates in FIELD_TEMPLATES.items():
            true_val = c[field]
            if true_val not in templates:
                continue
            true_sentence = templates[true_val]
            wrong_candidates = [v for v in templates if v != true_val]
            wrong_val = random.choice(wrong_candidates)
            wrong_sentence = templates[wrong_val]

            score_true = verify_caption_matches_video(str(video_path), true_sentence, n_frames=N_FRAMES)
            score_wrong = verify_caption_matches_video(str(video_path), wrong_sentence, n_frames=N_FRAMES)

            rows.append({
                "clip_dir": c["clip_dir"], "field": field,
                "true_value": true_val, "wrong_value": wrong_val,
                "true_sentence": true_sentence, "wrong_sentence": wrong_sentence,
                "score_true": score_true, "score_wrong": score_wrong,
                "margin": score_true - score_wrong,
            })

        if (i + 1) % 5 == 0:
            elapsed = time.time() - t0
            print(f"  진행: {i+1}/{len(sample)}  (경과 {elapsed/60:.1f}분)")

    import pandas as pd
    df = pd.DataFrame(rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"\n{'='*80}")
    print(f"총 {len(df)}개 (참,거짓) 쌍 비교 완료 (소요 {(time.time()-t0)/60:.1f}분)\n")

    print(f"{'필드':15s} {'개수':>5s} {'평균 margin':>12s} {'양수 비율':>10s} {'t-검정 p':>10s}")
    for field in FIELD_TEMPLATES:
        sub = df[df["field"] == field]
        if len(sub) < 2:
            continue
        margins = sub["margin"].to_numpy()
        t_stat, p = ttest_1samp(margins, 0)
        pos_rate = (margins > 0).mean()
        print(f"{field:15s} {len(sub):5d} {margins.mean():+11.4f} {pos_rate*100:9.1f}% {p:10.4f}")

    all_margins = df["margin"].to_numpy()
    t_stat, p_all = ttest_1samp(all_margins, 0)
    pos_rate_all = (all_margins > 0).mean()
    print(f"\n전체: 평균 margin={all_margins.mean():+.4f}, 양수 비율={pos_rate_all*100:.1f}%, p={p_all:.4f}")
    print(f"\n결과 저장: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
