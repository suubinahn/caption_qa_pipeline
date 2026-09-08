"""
타임라인 일관성 대규모 검증 (180클립, 실제 회사 데이터)
============================================================

temporal_consistency_check.py를 10개 파일럿 샘플로 돌렸을 때 평균 상관계수
-0.115가 나왔지만, 표본이 10개뿐이라 통계적으로 유의한지 확인할 수 없었다
(1표본 t-검정 p=0.512). 검정력 계산 결과 이 정도 효과크기(-0.115, sd=0.532)를
잡아내려면 약 150~170개 표본이 필요해서, 실제 회사 mount 전체(740클립)에서
180개를 무작위 추출해 재검증한다.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, wilcoxon

import sys
sys.path.insert(0, str(Path(__file__).parent))
from temporal_consistency_check import temporal_consistency_check

FULL_DATASET_JSON = Path("outputs/search_full_dataset.json")
OUTPUT_CSV = Path("outputs/temporal_consistency_largescale.csv")
N_SAMPLE = 180
SEED = 42


def main():
    with open(FULL_DATASET_JSON, encoding="utf-8") as f:
        all_clips = json.load(f)

    random.seed(SEED)
    sample = random.sample(all_clips, min(N_SAMPLE, len(all_clips)))
    print(f"전체 {len(all_clips)}개 중 {len(sample)}개 무작위 추출 (seed={SEED})\n")

    rows = []
    t_start = time.time()
    for i, c in enumerate(sample):
        video_path = Path(c["clip_dir"]) / "1_clip" / "5.mp4"
        caption_path = Path(c["clip_dir"]) / "caption.txt"
        if not video_path.exists() or not caption_path.exists():
            continue
        caption = caption_path.read_text(encoding="utf-8").strip()
        try:
            result = temporal_consistency_check(str(video_path), caption, n_frames=10)
        except Exception as e:
            print(f"  [{i+1}/{len(sample)}] 실패: {c['clip_dir']} ({e})")
            continue

        rows.append({
            "clip_dir": c["clip_dir"],
            "n_sentences": result["n_sentences"],
            "correlation": result["temporal_correlation"],
            "p_value": result["p_value"],
        })

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t_start
            rate = elapsed / (i + 1)
            remaining = rate * (len(sample) - i - 1)
            print(f"  진행: {i+1}/{len(sample)}  (경과 {elapsed/60:.1f}분, 예상 잔여 {remaining/60:.1f}분)")

    df = pd.DataFrame(rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    valid = df[df["n_sentences"] >= 3].dropna(subset=["correlation"])
    corrs = valid["correlation"].to_numpy()

    print(f"\n{'='*80}")
    print(f"유효 표본: {len(corrs)}개 (문장 3개 미만이라 상관계수 계산 불가한 클립 제외)")
    print(f"평균: {corrs.mean():+.3f}  중앙값: {np.median(corrs):+.3f}  표준편차: {corrs.std(ddof=1):.3f}")
    print(f"0.5 이상(순서 잘 지킴): {(corrs > 0.5).sum()}개 ({(corrs>0.5).mean()*100:.1f}%)")
    print(f"-0.5~0.5(뒤죽박죽): {((corrs >= -0.5) & (corrs <= 0.5)).sum()}개 ({((corrs>=-0.5)&(corrs<=0.5)).mean()*100:.1f}%)")
    print(f"-0.5 이하(역순): {(corrs < -0.5).sum()}개 ({(corrs<-0.5).mean()*100:.1f}%)")

    t_stat, p_ttest = ttest_1samp(corrs, 0)
    w_stat, p_wilcoxon = wilcoxon(corrs)
    print(f"\n1표본 t-검정 (평균이 0과 다른가): t={t_stat:.3f}, p={p_ttest:.4f}")
    print(f"Wilcoxon 부호순위 검정: W={w_stat:.3f}, p={p_wilcoxon:.4f}")
    verdict = "유의함 (평균이 0과 다르다고 말할 수 있음)" if p_ttest < 0.05 else "유의하지 않음"
    print(f"-> p<0.05 기준: {verdict}")

    print(f"\n결과 저장: {OUTPUT_CSV}")
    print(f"총 소요 시간: {(time.time()-t_start)/60:.1f}분")


if __name__ == "__main__":
    main()
