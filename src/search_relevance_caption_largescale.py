"""
검색 관련성 대규모 실험 (캡션 기반) - 100클립 x 9검색어 x 2캡션소스
========================================================================

search_relevance_largescale.py(프레임 직접비교, 이론적 상한선)와 짝을
이루는 스크립트. 이번엔 검색어를 영상이 아니라 "실제 캡션 텍스트"와
비교한다 - caption.txt(회사 프로덕션)와 sb_caption/2_caption.txt(자체
파이프라인) 둘 다.

77토큰 잘림 문제(앞서 확인함) 때문에, 캡션 전체를 그냥 넣지 않고
문장 단위로 쪼개서 최댓값을 쓰는 "문장단위 max-pooling" 방식만 쓴다
(지난 5개 검색어 실험에서 이게 "캡션 전체 그냥 자르기"보다 안정적이었음).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import pandas as pd

from clip_score import load_model
from validate_sentence_split_recall import split_sentences

SAMPLE_JSON = Path("outputs/search_sample_100.json")
OUTPUT_CSV = Path("outputs/search_largescale_caption_results.csv")
OUTPUT_SUMMARY_CSV = Path("outputs/search_largescale_caption_summary.csv")

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

CAPTION_SOURCES = {
    "caption_txt": "caption.txt",
    "sb_caption_2": "sb_caption/2_caption.txt",
}


def batch_text_embed(texts: list[str]):
    """여러 문장을 한 번에 CLIP 텍스트 인코더에 넣어 정규화된 임베딩 행렬을 얻는다."""
    model, processor, device = load_model()
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        feats = model.get_text_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats  # (n_texts, dim), L2 정규화됨


def main():
    with open(SAMPLE_JSON, encoding="utf-8") as f:
        clips = json.load(f)

    query_texts = [q for q, _, _ in QUERIES]
    query_embeds = batch_text_embed(query_texts)  # (9, dim)

    all_rows = []
    for source_name, filename in CAPTION_SOURCES.items():
        print(f"\n{'='*100}\n캡션 소스: {source_name} ({filename})\n{'='*100}")

        # 클립마다 캡션을 문장 단위로 쪼개고, 전부 한꺼번에 임베딩 (효율성)
        clip_sentences = {}
        for c in clips:
            path = Path(c["clip_dir"]) / filename
            text = path.read_text(encoding="utf-8").strip()
            sentences = split_sentences(text) if text else [text]
            clip_sentences[c["clip_dir"]] = sentences

        # 이 소스의 모든 문장을 한 번에 임베딩 (클립 100개 x 평균 5~8문장 = 수백 개, 배치 처리)
        flat_sentences = []
        owner = []  # 각 문장이 어느 클립 것인지
        for clip_dir, sents in clip_sentences.items():
            for s in sents:
                flat_sentences.append(s)
                owner.append(clip_dir)

        print(f"  총 문장 수: {len(flat_sentences)}개 (클립 {len(clips)}개)")
        sent_embeds = batch_text_embed(flat_sentences)  # (n_sentences, dim)

        # 검색어 x 문장 유사도 행렬 한 번에 계산 후, 클립별 최댓값(max-pooling)
        sim_matrix = (query_embeds @ sent_embeds.T).cpu().numpy()  # (9, n_sentences)

        for qi, (query, field, val) in enumerate(QUERIES):
            scores_by_clip = {}
            for si, clip_dir in enumerate(owner):
                s = sim_matrix[qi, si]
                if clip_dir not in scores_by_clip or s > scores_by_clip[clip_dir]:
                    scores_by_clip[clip_dir] = s

            for c in clips:
                score = scores_by_clip.get(c["clip_dir"])
                if score is None:
                    continue
                all_rows.append({
                    "source": source_name, "query": query, "clip_dir": c["clip_dir"],
                    "score": float(score), "is_positive": (c[field] == val),
                })

    df = pd.DataFrame(all_rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    # --- 요약: 소스별 x 검색어별 precision@K ---
    summary_rows = []
    print("\n\n" + "=" * 100)
    print("소스별 x 검색어별 Precision@K 비교")
    print("=" * 100)
    header = f"{'검색어':45s} {'정답수':>6s} {'caption.txt':>14s} {'sb_caption_2':>14s}"
    print(header)
    for query, field, val in QUERIES:
        row_vals = {}
        for source_name in CAPTION_SOURCES:
            sub = df[(df["source"] == source_name) & (df["query"] == query)].sort_values("score", ascending=False)
            n_positive = sub["is_positive"].sum()
            if n_positive == 0:
                row_vals[source_name] = None
                continue
            k = n_positive
            precision = sub.head(k)["is_positive"].sum() / k
            row_vals[source_name] = precision
            summary_rows.append({"query": query, "source": source_name, "n_positive": int(n_positive), "precision_at_k": round(precision, 3)})

        n_pos_display = df[(df["source"]=="caption_txt") & (df["query"]==query)]["is_positive"].sum()
        c1 = row_vals.get("caption_txt")
        c2 = row_vals.get("sb_caption_2")
        print(f"{query:45s} {n_pos_display:>6d} {c1*100 if c1 is not None else -1:>13.1f}% {c2*100 if c2 is not None else -1:>13.1f}%")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    print(f"\n전체 결과: {OUTPUT_CSV}")
    print(f"요약: {OUTPUT_SUMMARY_CSV}")


if __name__ == "__main__":
    main()
