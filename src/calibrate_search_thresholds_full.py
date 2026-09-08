"""
전체 데이터셋(740클립) 기준 컷오프 재보정
=============================================

calibrate_search_thresholds.py의 100클립 버전과 동일한 방법론(짝/홀
calibration/holdout 분리, F1 최대화 임계값)을 outputs/search_full_dataset.json
(회사 mount 전체 740클립)에 적용한다.

주의: turning_right/turning_left/tunnel/bridge/night/snowing 6개 검색어는
애초에 100클립 표본에도 전량 포함돼 있었기 때문에(희귀 카테고리라 전수
샘플링함), 전체 데이터로 확장해도 양성 샘플 수가 그대로다 - 이 6개는
개선을 기대하기 어렵다. stop/rainy/multi-lane highway 3개만 양성 샘플이
크게 늘었으므로(11->126, 35->240, 41->216) 이 세 개의 재보정 효과를
확인하는 게 이번 실험의 핵심이다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent))
from clip_score import load_model
from validate_sentence_split_recall import split_sentences

SAMPLE_JSON = Path("outputs/search_full_dataset.json")
OUTPUT_JSON = Path("outputs/search_thresholds_full.json")

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


def embed(texts, model, processor, device, batch_size=256):
    all_feats = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            feats = model.get_text_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        all_feats.append(feats)
    return torch.cat(all_feats, dim=0)


def best_f1_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    order = np.argsort(scores)
    sorted_scores = scores[order]
    candidates = [sorted_scores[0] - 1.0]
    candidates += [(sorted_scores[i] + sorted_scores[i + 1]) / 2 for i in range(len(sorted_scores) - 1)]
    candidates.append(sorted_scores[-1] + 1.0)

    best_thr, best_f1 = candidates[0], -1.0
    for thr in candidates:
        pred = scores >= thr
        tp = ((pred == 1) & (labels == 1)).sum()
        fp = ((pred == 1) & (labels == 0)).sum()
        fn = ((pred == 0) & (labels == 1)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_thr, best_f1


def main():
    with open(SAMPLE_JSON, encoding="utf-8") as f:
        clips = json.load(f)

    calib_clips = [c for i, c in enumerate(clips) if i % 2 == 0]
    holdout_clips = [c for i, c in enumerate(clips) if i % 2 == 1]
    print(f"calibration: {len(calib_clips)}개 클립, holdout: {len(holdout_clips)}개 클립\n")

    model, processor, device = load_model()

    def build_sentence_index(subset_clips):
        # 캡션 파일 없음/빈 파일 방어 (코드리뷰로 발견 - search_module.py의
        # CaptionSearchIndex._build()엔 있는데 여기엔 없어서 데이터가 지저분하면
        # 실행 도중 크래시할 수 있었음). 지금까지의 실행에선 문제 없었지만
        # (outputs/search_full_dataset.json 740개 전부 정상 파싱됨), 재실행 시
        # 데이터가 바뀌면 위험할 수 있어 미리 막아둔다.
        sentences, owner = [], []
        for c in subset_clips:
            path = Path(c["clip_dir"]) / "caption.txt"
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8").strip()
            if not text:
                continue
            for s in split_sentences(text):
                sentences.append(s)
                owner.append(c["clip_dir"])
        embeds = embed(sentences, model, processor, device).cpu().numpy()
        return sentences, owner, embeds

    print("calibration 세트 임베딩 중...")
    calib_sent, calib_owner, calib_embeds = build_sentence_index(calib_clips)
    print("holdout 세트 임베딩 중...")
    hold_sent, hold_owner, hold_embeds = build_sentence_index(holdout_clips)

    query_texts = [q for q, _, _ in QUERIES]
    q_embeds = embed(query_texts, model, processor, device).cpu().numpy()

    sim_calib_all = q_embeds @ calib_embeds.T
    r_S = sim_calib_all.mean(axis=0)

    sim_hold_all = q_embeds @ hold_embeds.T
    r_S_hold = sim_hold_all.mean(axis=0)

    def clip_scores(sim_row, owner, r_S_local):
        r_T = np.partition(sim_row, -min(10, len(sim_row)))[-min(10, len(sim_row)):].mean()
        corrected = 2 * sim_row - r_T - r_S_local
        best = {}
        for s, clip_dir in zip(corrected, owner):
            if clip_dir not in best or s > best[clip_dir]:
                best[clip_dir] = s
        return best

    thresholds = {}
    print(f"\n{'검색어':45s} {'양성(calib/hold)':>18s} {'calib F1':>10s} {'thr':>10s} {'hold acc':>10s} {'hold prec':>11s} {'hold recall':>13s}")
    for qi, (q, field, val) in enumerate(QUERIES):
        calib_scores_dict = clip_scores(sim_calib_all[qi], calib_owner, r_S)
        calib_labels = np.array([1 if c[field] == val else 0 for c in calib_clips])
        calib_scores_arr = np.array([calib_scores_dict[c["clip_dir"]] for c in calib_clips])
        n_pos_calib = int(calib_labels.sum())

        thr, calib_f1 = best_f1_threshold(calib_scores_arr, calib_labels)

        hold_scores_dict = clip_scores(sim_hold_all[qi], hold_owner, r_S_hold)
        hold_labels = np.array([1 if c[field] == val else 0 for c in holdout_clips])
        hold_scores_arr = np.array([hold_scores_dict[c["clip_dir"]] for c in holdout_clips])
        n_pos_hold = int(hold_labels.sum())

        hold_pred = hold_scores_arr >= thr
        hold_acc = (hold_pred == hold_labels).mean()
        tp = ((hold_pred == 1) & (hold_labels == 1)).sum()
        fp = ((hold_pred == 1) & (hold_labels == 0)).sum()
        fn = ((hold_pred == 0) & (hold_labels == 1)).sum()
        hold_prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        hold_recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")

        thresholds[q] = {"threshold": float(thr), "field": field, "value": val,
                          "n_pos_calib": n_pos_calib, "n_pos_holdout": n_pos_hold,
                          "calib_f1": float(calib_f1), "holdout_accuracy": float(hold_acc),
                          "holdout_precision": float(hold_prec), "holdout_recall": float(hold_recall)}

        print(f"{q:45s} {f'{n_pos_calib}/{n_pos_hold}':>18s} {calib_f1*100:9.1f}% {thr:10.4f} {hold_acc*100:9.1f}% {hold_prec*100:10.1f}% {hold_recall*100:12.1f}%")

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(thresholds, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
