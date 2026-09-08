"""
검색 컷오프(임계값) 보정 - calibration/holdout 분리
======================================================

지금까지 search_module.py는 순위만 매겼다 ("점수 높은 순으로 top-K").
이제 "관련 있음/없음"을 가르는 임계값을 정해서, 순위 없이도 "이 클립은
이 검색어와 관련 있다/없다"를 이분류할 수 있게 한다.

CSLS 보정 점수는 검색어마다 스케일이 다르므로(예: "정지"는 상위권도
0.07대, "좌회전"은 0.10대), 전체 검색어 공통 절대 임계값 하나로는 안
맞는다. 대신 검색어별로 개별 임계값을 정한다 - 단, 이건 지금 모듈이
검증한 범위(9개 도메인 쿼리)에서만 유효하고, 새 검색어를 추가하려면
그 검색어도 똑같이 calibration/holdout으로 다시 보정해야 한다.

방법 (이전 caption-QA 단계에서 쓴 것과 동일한 원칙):
  1) 100클립을 절반씩 calibration/holdout으로 나눈다 (짝/홀 인덱스 분리 -
     비디오 자체를 기준으로 나눔, 검색어를 나누는 게 아님)
  2) calibration 세트에서만 정확도(accuracy)를 최대화하는 임계값을
     탐색한다 (정렬된 점수들의 인접 중간값 전부를 후보로 exhaustive search)
  3) 그 임계값을 holdout 세트에 적용해서 "본 적 없는 클립"에서도
     실제로 통하는지 정직하게 검증한다
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

SAMPLE_JSON = Path("outputs/search_sample_100.json")
OUTPUT_JSON = Path("outputs/search_thresholds.json")

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


def embed(texts, model, processor, device):
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        feats = model.get_text_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats


def best_f1_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """F1(정밀도-재현율 조화평균)을 최대화하는 임계값을 exhaustive search로
    찾는다. accuracy를 쓰면 양성 비율이 낮은 검색어(예: 좌회전 8/100)에서
    "전부 관련없음"으로 찍어도 92% accuracy가 나오는 함정에 빠진다 - 실제로
    처음엔 accuracy 기준으로 했다가 recall이 0%로 무너지는 걸 확인하고
    F1으로 바꿈. (인접한 두 점수의 중간값들만 후보로 삼으면 충분 - 그 사이
    어떤 값을 잡아도 분류 결과는 동일하기 때문)"""
    order = np.argsort(scores)
    sorted_scores = scores[order]
    candidates = [sorted_scores[0] - 1.0]  # "전부 관련 있음"에 해당하는 극단값
    candidates += [(sorted_scores[i] + sorted_scores[i + 1]) / 2 for i in range(len(sorted_scores) - 1)]
    candidates.append(sorted_scores[-1] + 1.0)  # "전부 관련 없음"에 해당하는 극단값

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

    # 짝/홀 인덱스로 calibration/holdout 분리 (50/50, 비디오 기준 분리)
    calib_clips = [c for i, c in enumerate(clips) if i % 2 == 0]
    holdout_clips = [c for i, c in enumerate(clips) if i % 2 == 1]
    print(f"calibration: {len(calib_clips)}개 클립, holdout: {len(holdout_clips)}개 클립\n")

    model, processor, device = load_model()

    def build_sentence_index(subset_clips):
        sentences, owner = [], []
        for c in subset_clips:
            text = (Path(c["clip_dir"]) / "caption.txt").read_text(encoding="utf-8").strip()
            for s in split_sentences(text):
                sentences.append(s)
                owner.append(c["clip_dir"])
        embeds = embed(sentences, model, processor, device).cpu().numpy()
        return sentences, owner, embeds

    calib_sent, calib_owner, calib_embeds = build_sentence_index(calib_clips)
    hold_sent, hold_owner, hold_embeds = build_sentence_index(holdout_clips)

    query_texts = [q for q, _, _ in QUERIES]
    q_embeds = embed(query_texts, model, processor, device).cpu().numpy()

    # CSLS의 r_S(허브 보정)는 calibration 세트 문장만으로 계산 (holdout 정보 유출 방지)
    sim_calib_all = q_embeds @ calib_embeds.T  # (9, n_calib_sentences)
    r_S = sim_calib_all.mean(axis=0)  # anchor query 9개 전체 평균 (k=9=전체)

    def clip_scores(sim_row, owner, r_S_local=None):
        r_T = np.partition(sim_row, -min(10, len(sim_row)))[-min(10, len(sim_row)):].mean()
        if r_S_local is not None:
            corrected = 2 * sim_row - r_T - r_S_local
        else:
            corrected = sim_row
        best = {}
        for s, clip_dir in zip(corrected, owner):
            if clip_dir not in best or s > best[clip_dir]:
                best[clip_dir] = s
        return best

    # holdout 문장들의 r_S는 "calibration에서 배운 일반적 허브 경향"을 그대로 적용해야 공정하다.
    # 하지만 r_S는 문장 단위(sentence-level)이고 calibration/holdout 문장은 겹치지 않으므로,
    # holdout 문장 각각에 대해 새로 계산하되, anchor query는 동일한 9개를 그대로 쓴다
    # (anchor query 자체는 calibration에서도 holdout에서도 미리 정해진 고정값이라 유출이 아님).
    sim_hold_all = q_embeds @ hold_embeds.T
    r_S_hold = sim_hold_all.mean(axis=0)

    thresholds = {}
    print(f"{'검색어':45s} {'calib F1':>10s} {'thr':>10s} {'holdout acc':>12s} {'holdout prec':>12s} {'holdout recall':>14s}")
    for qi, (q, field, val) in enumerate(QUERIES):
        calib_scores_dict = clip_scores(sim_calib_all[qi], calib_owner, r_S)
        calib_labels = np.array([1 if c[field] == val else 0 for c in calib_clips])
        calib_scores_arr = np.array([calib_scores_dict[c["clip_dir"]] for c in calib_clips])

        thr, calib_f1 = best_f1_threshold(calib_scores_arr, calib_labels)

        hold_scores_dict = clip_scores(sim_hold_all[qi], hold_owner, r_S_hold)
        hold_labels = np.array([1 if c[field] == val else 0 for c in holdout_clips])
        hold_scores_arr = np.array([hold_scores_dict[c["clip_dir"]] for c in holdout_clips])

        hold_pred = hold_scores_arr >= thr
        hold_acc = (hold_pred == hold_labels).mean()
        tp = ((hold_pred == 1) & (hold_labels == 1)).sum()
        fp = ((hold_pred == 1) & (hold_labels == 0)).sum()
        fn = ((hold_pred == 0) & (hold_labels == 1)).sum()
        hold_prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        hold_recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")

        thresholds[q] = {"threshold": float(thr), "field": field, "value": val,
                          "calib_f1": float(calib_f1), "holdout_accuracy": float(hold_acc),
                          "holdout_precision": float(hold_prec), "holdout_recall": float(hold_recall)}

        print(f"{q:45s} {calib_f1*100:9.1f}% {thr:10.4f} {hold_acc*100:11.1f}% {hold_prec*100:11.1f}% {hold_recall*100:13.1f}%")

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(thresholds, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
