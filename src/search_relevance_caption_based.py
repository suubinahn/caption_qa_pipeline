"""
검색 관련성 실험 (정정판) - 검색어를 프레임이 아니라 "캡션 텍스트"와 비교
==========================================================================

지난 실험은 검색어를 영상 프레임과 직접 비교해서 "영상 내용 자체를 얼마나
잘 이해하는가"를 봤는데, 실제 검색 시스템은 캡션(텍스트)을 인덱스로 써서
검색어와 "텍스트 vs 텍스트"로 비교한다. 그래서 캡션이 부정확하면 검색도
같이 나빠진다 - 이게 우리가 캡션 정확도를 검증해온 진짜 이유였다.

이번엔 CLIP의 텍스트 인코더로 "검색어 텍스트"와 "캡션 텍스트"를 직접
비교한다 (이미지 인코더는 아예 안 씀 - 순수 텍스트 대 텍스트 의미 유사도).

두 가지 방식을 같이 계산한다:
  1) 캡션 전체를 그대로 넣기 - 단, CLIP 텍스트 인코더는 77토큰까지만
     보므로, 우리 캡션(평균 180토큰)은 뒤 절반 이상이 잘려나간다. 실제
     시스템을 안일하게 구현하면 이렇게 될 수 있다는 걸 그대로 보여준다.
  2) 캡션을 문장 단위로 쪼개서, 검색어와 가장 잘 맞는 문장 하나를 골라
     그 점수를 쓴다 (max-pooling) - 잘림 없이 캡션 전체를 살펴보는 더
     공정한 방식.
"""

from __future__ import annotations

import csv
from pathlib import Path

import torch
import pandas as pd

from clip_score import load_model
from validate_sentence_split_recall import split_sentences

ROOT = Path(__file__).resolve().parent.parent
PILOT_CSV = ROOT / "results" / "pilot_result.csv"
FRAME_BASED_CSV = ROOT / "outputs" / "search_relevance_results.csv"
OUTPUT_CSV = ROOT / "outputs" / "search_relevance_caption_based.csv"

QUERIES = [
    "A pedestrian is crossing the street.",
    "The vehicle is driving in heavy rain.",
    "The car is turning right at an intersection.",
    "Bright sunny weather with clear skies.",
    "A bus is visible on the road.",
]


def text_similarity(text_a: str, text_b: str) -> float:
    """CLIP 텍스트 인코더로 두 문장의 코사인 유사도를 구한다 (이미지 없이,
    순수 텍스트 대 텍스트). truncation=True로 77토큰 넘으면 자동으로 자른다."""
    model, processor, device = load_model()
    inputs = processor(text=[text_a, text_b], return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        feats = model.get_text_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        sim = (feats[0] @ feats[1]).item()
    return sim


def main():
    with open(PILOT_CSV, "r", encoding="utf-8-sig", newline="") as f:
        rows = {r["id"]: r for r in csv.DictReader(f)}

    video_ids = list(rows.keys())

    # 캡션을 미리 문장 단위로 쪼개둔다
    sentences_by_video = {vid: split_sentences(rows[vid]["caption"]) for vid in video_ids}

    results_full = {q: {} for q in QUERIES}   # 방식1: 캡션 전체(잘림)
    results_sent = {q: {} for q in QUERIES}   # 방식2: 문장 단위 max

    for q in QUERIES:
        print(f"\n[검색어] \"{q}\"")
        for vid in video_ids:
            full_caption = rows[vid]["caption"]
            score_full = text_similarity(q, full_caption)
            results_full[q][vid] = score_full

            sent_scores = [text_similarity(q, s) for s in sentences_by_video[vid]]
            score_sent = max(sent_scores)
            results_sent[q][vid] = score_sent

        print("  [방식1: 캡션 전체(77토큰 잘림)]")
        for rank, (vid, sc) in enumerate(sorted(results_full[q].items(), key=lambda x: -x[1]), 1):
            print(f"    {rank:2d}위. {vid}  {sc:.4f}")
        print("  [방식2: 문장 단위 max (잘림 없음)]")
        for rank, (vid, sc) in enumerate(sorted(results_sent[q].items(), key=lambda x: -x[1]), 1):
            print(f"    {rank:2d}위. {vid}  {sc:.4f}")

    df_full = pd.DataFrame(results_full)
    df_sent = pd.DataFrame(results_sent)
    df_full.index.name = "video_id"
    df_sent.index.name = "video_id"

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_full.to_csv(str(OUTPUT_CSV).replace(".csv", "_full.csv"), encoding="utf-8-sig")
    df_sent.to_csv(str(OUTPUT_CSV).replace(".csv", "_sentmax.csv"), encoding="utf-8-sig")

    # --- 프레임 기반(지난 실험) 결과와 나란히 비교 ---
    if FRAME_BASED_CSV.exists():
        df_frame = pd.read_csv(FRAME_BASED_CSV, index_col="video_id")
        print("\n" + "=" * 100)
        print("검색어별 1~3위 비교: 프레임 직접비교 vs 캡션 전체(잘림) vs 캡션 문장단위(잘림없음)")
        print("=" * 100)
        for q in QUERIES:
            top_frame = df_frame[q].sort_values(ascending=False).index[:3].tolist()
            top_full = df_full[q].sort_values(ascending=False).index[:3].tolist()
            top_sent = df_sent[q].sort_values(ascending=False).index[:3].tolist()
            print(f"\n[{q}]")
            print(f"  프레임 직접비교      : {top_frame}")
            print(f"  캡션 전체(잘림)      : {top_full}")
            print(f"  캡션 문장단위(안잘림) : {top_sent}")

    print(f"\n결과 CSV: {OUTPUT_CSV} (에 _full/_sentmax 붙여서 저장됨)")


if __name__ == "__main__":
    main()
