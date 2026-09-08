"""
타임라인 일관성 재검증 - 이벤트성 문장만 걸러서 재계산
============================================================

temporal_consistency_largescale.py(180클립)에서 "뒤죽박죽"(상관계수 -0.5~0.5)이
74%로 나왔는데, 이게 진짜 순서 혼동 때문인지, 아니면 애초에 특정 시점이 없는
"종합 묘사" 문장(예: "날씨가 흐리다", "도로가 젖어있다")이 섞여서 노이즈가 낀
것인지 구분되지 않았다 - temporal_consistency_check.py를 처음 만들 때부터
"이런 문장은 노이즈를 만든다"고 적어뒀던 한계인데 실제로 검증한 적은 없었다.

이번엔 문장을 "이벤트성"(구체적 동작/전환, 특정 시점에 묶임)과 "종합묘사"
(날씨/도로유형/색상 등, 영상 전체에 걸쳐 참인 정적 서술)로 나눠서, 이벤트성
문장만으로 상관계수를 다시 계산해 비교한다. 원본 180클립 표본(seed=42)을
그대로 재사용하고, 이번엔 문장 단위 원본 데이터도 CSV에 저장해서 나중에
다른 방식으로도 재분석할 수 있게 한다 (BLIP-ITM 재실행 없이).
"""

from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_1samp, wilcoxon

import sys
sys.path.insert(0, str(Path(__file__).parent))
from frame_extract import extract_frames
from blip_itm import compute_itm_score
from validate_sentence_split_recall import split_sentences

FULL_DATASET_JSON = Path("outputs/search_full_dataset.json")
OUTPUT_SENTENCES_CSV = Path("outputs/temporal_per_sentence.csv")
OUTPUT_SUMMARY_CSV = Path("outputs/temporal_event_filtered_summary.csv")
N_SAMPLE = 180
N_FRAMES = 10
SEED = 42

# 이벤트성 문장 키워드: 특정 시점에 묶이는 동작/전환을 나타내는 표현.
# 이것들이 안 걸리면 "종합묘사"(날씨/도로유형/색상/전반적 분위기 등)로 간주.
EVENT_KEYWORDS = [
    "turn", "turning", "turns",
    "approach", "approaching", "approaches",
    "enter", "entering", "enters",
    "exit", "exiting", "exits",
    "pass", "passing", "passes",
    "cross", "crossing", "crosses",
    "merge", "merging", "merges",
    "stop", "stopping", "stops", "stopped",
    "accelerat", "decelerat", "brake", "braking",
    "pulls over", "pulling over",
    "changes lane", "changing lane", "change lane",
    "navigates", "navigating",
    "proceeds through", "proceeding through",
    "arrives", "arriving",
]


def is_event_sentence(sentence: str) -> bool:
    s = sentence.lower()
    return any(kw in s for kw in EVENT_KEYWORDS)


def temporal_check_with_sentences(video_path: str, caption: str, n_frames: int = 10) -> list[dict]:
    sentences = split_sentences(caption)
    frames = extract_frames(video_path, n_frames=n_frames)
    timestamps = np.linspace(0.10, 0.90, n_frames)

    rows = []
    for i, sentence in enumerate(sentences):
        scores = [compute_itm_score(f, sentence) for f in frames]
        best_idx = int(np.argmax(scores))
        rows.append({
            "caption_order": i, "sentence": sentence, "best_frame_idx": best_idx,
            "best_timestamp_pct": round(timestamps[best_idx] * 100, 1),
            "best_score": round(scores[best_idx], 4),
            "is_event": is_event_sentence(sentence),
        })
    return rows


def correlation_of(rows: list[dict]) -> tuple[float, float]:
    if len(rows) < 3:
        return float("nan"), float("nan")
    orders = [r["caption_order"] for r in rows]
    frame_idx = [r["best_frame_idx"] for r in rows]
    return spearmanr(orders, frame_idx)


def main():
    with open(FULL_DATASET_JSON, encoding="utf-8") as f:
        all_clips = json.load(f)

    random.seed(SEED)
    sample = random.sample(all_clips, min(N_SAMPLE, len(all_clips)))
    print(f"전체 {len(all_clips)}개 중 {len(sample)}개 (이전과 동일한 seed={SEED} 표본)\n")

    all_sentence_rows = []
    clip_results = []
    t0 = time.time()
    for i, c in enumerate(sample):
        video_path = Path(c["clip_dir"]) / "1_clip" / "5.mp4"
        caption_path = Path(c["clip_dir"]) / "caption.txt"
        if not video_path.exists() or not caption_path.exists():
            continue
        caption = caption_path.read_text(encoding="utf-8").strip()
        try:
            rows = temporal_check_with_sentences(str(video_path), caption, n_frames=N_FRAMES)
        except Exception as e:
            print(f"  [{i+1}/{len(sample)}] 실패: {c['clip_dir']} ({e})")
            continue

        for r in rows:
            r["clip_dir"] = c["clip_dir"]
            all_sentence_rows.append(r)

        # 전체 문장 기준 상관계수
        corr_all, p_all = correlation_of(rows)
        # 이벤트성 문장만 걸러서 재계산
        event_rows = [r for r in rows if r["is_event"]]
        corr_event, p_event = correlation_of(event_rows)

        clip_results.append({
            "clip_dir": c["clip_dir"], "n_sentences": len(rows), "n_event_sentences": len(event_rows),
            "correlation_all": corr_all, "correlation_event_only": corr_event,
        })

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            rate = elapsed / (i + 1)
            remaining = rate * (len(sample) - i - 1)
            print(f"  진행: {i+1}/{len(sample)}  (경과 {elapsed/60:.1f}분, 예상 잔여 {remaining/60:.1f}분)")

    df_sent = pd.DataFrame(all_sentence_rows)
    OUTPUT_SENTENCES_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_sent.to_csv(OUTPUT_SENTENCES_CSV, index=False, encoding="utf-8-sig")

    df_clip = pd.DataFrame(clip_results)
    df_clip.to_csv(OUTPUT_SUMMARY_CSV, index=False, encoding="utf-8-sig")

    print(f"\n{'='*80}")
    print(f"문장 총 {len(df_sent)}개 중 이벤트성 문장: {df_sent['is_event'].sum()}개 ({df_sent['is_event'].mean()*100:.1f}%)")

    for label, col in [("전체 문장 기준", "correlation_all"), ("이벤트성 문장만", "correlation_event_only")]:
        valid = df_clip.dropna(subset=[col])
        corrs = valid[col].to_numpy()
        if len(corrs) < 3:
            print(f"\n[{label}] 유효 표본 부족 ({len(corrs)}개)")
            continue
        t_stat, p_ttest = ttest_1samp(corrs, 0)
        print(f"\n[{label}] 유효 클립 {len(corrs)}개")
        print(f"  평균={corrs.mean():+.3f}  중앙값={np.median(corrs):+.3f}  표준편차={corrs.std(ddof=1):.3f}")
        print(f"  0.5이상={100*(corrs>0.5).mean():.1f}%  뒤죽박죽={100*((corrs>=-0.5)&(corrs<=0.5)).mean():.1f}%  -0.5이하={100*(corrs<-0.5).mean():.1f}%")
        print(f"  1표본 t-검정: t={t_stat:.3f}, p={p_ttest:.4f}")

    print(f"\n결과 저장: {OUTPUT_SENTENCES_CSV}, {OUTPUT_SUMMARY_CSV}")
    print(f"총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
