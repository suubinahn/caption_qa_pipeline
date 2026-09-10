"""
X-CLIP(영상 네이티브 모델)로 "정지" 판정 재시도 - 가벼운 실험
================================================================

진짜 cross-encoder급 VTM(영상-텍스트 매칭)은 쉽게 구할 수 있는 게 없어서,
대신 "프레임을 독립적으로 안 보고, 영상 하나로 묶어서 처리하는" X-CLIP
(듀얼 인코더지만 프레임 간 정보 교환이 내부에 있음, Microsoft)으로
같은 문제(BLIP-ITM이 프레임을 독립적으로만 채점해서 "정지"를 구조적으로
못 보던 문제)를 다르게 풀 수 있는지 확인한다.

이 파일은 완전히 독립적이다 - 기존 파이프라인 파일(blip_itm.py,
search_and_verify.py, experiment_optical_flow_motion.py)은 하나도
건드리지 않는다. frame_extract.py만 프레임 추출 용도로 재사용한다.

비교 기준: 옵티컬 플로우가 이미 holdout 85.0%를 냈으므로, 그것과
직접 비교할 근거가 된다.
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from frame_extract import extract_frames  # 프레임 추출만 재사용

N_PER_CLASS = 15  # 정지 15 + 그 외 15 = 30클립, 가벼운 탐색 규모
N_FRAMES = 8  # X-CLIP base 체크포인트가 기대하는 프레임 수
SEED = 5

CANDIDATE_TEXTS = ["The vehicle is stopped.", "The vehicle is moving straight."]


def main():
    from transformers import XCLIPProcessor, XCLIPModel

    print("X-CLIP 모델 로딩 중...")
    model_name = "microsoft/xclip-base-patch32"
    processor = XCLIPProcessor.from_pretrained(model_name)
    model = XCLIPModel.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    with open("outputs/search_full_dataset.json", encoding="utf-8") as f:
        all_clips = json.load(f)
    stop_clips = [c for c in all_clips if c["motion"] == "stop"]
    other_clips = [c for c in all_clips if c["motion"] == "moving straight"]

    random.seed(SEED)
    sample = [(c, True) for c in random.sample(stop_clips, N_PER_CLASS)] + \
             [(c, False) for c in random.sample(other_clips, N_PER_CLASS)]
    random.shuffle(sample)

    print(f"평가 중... ({len(sample)}클립, motion=stop vs moving straight)")
    t0 = time.time()
    correct = 0
    rows = []
    for c, is_stop in sample:
        video_path = Path(c["clip_dir"]) / "1_clip" / "5.mp4"
        if not video_path.exists():
            continue
        frames = extract_frames(str(video_path), n_frames=N_FRAMES)
        pil_frames = [f.convert("RGB") for f in frames]

        # 이 transformers 버전은 processor(text=..., videos=...) 통합 호출에서
        # videos 인자가 유실되는 버그가 있어(직접 확인함), image_processor와
        # tokenizer를 따로 호출해서 우회한다.
        pixel_inputs = processor.image_processor.preprocess(pil_frames, return_tensors="pt")
        text_inputs = processor.tokenizer(text=CANDIDATE_TEXTS, return_tensors="pt", padding=True)
        inputs = {**pixel_inputs, **text_inputs}
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
            probs = out.logits_per_video.softmax(dim=1)[0]  # [정지 확률, 직진 확률]

        predicted_stop = probs[0].item() > probs[1].item()
        is_correct = predicted_stop == is_stop
        correct += is_correct
        rows.append((c["clip_dir"], is_stop, predicted_stop, probs[0].item(), probs[1].item()))

    elapsed = time.time() - t0
    n = len(rows)
    print(f"\naccuracy: {correct}/{n} = {correct/n*100:.1f}%  ({elapsed:.1f}초)")
    print(f"\n비교 참고: 옵티컬 플로우 holdout accuracy = 85.0% (n=200)")
    print(f"          BLIP-ITM(프레임 독립 채점) motion 전체 holdout accuracy = 68.5% (n=200)")

    print("\n샘플별 상세:")
    for clip_dir, is_stop, predicted_stop, p_stop, p_straight in rows:
        mark = "O" if is_stop == predicted_stop else "X"
        print(f"  [{mark}] 실제={'정지' if is_stop else '직진':4s}  예측={'정지' if predicted_stop else '직진':4s}  "
              f"(정지확률={p_stop:.3f}, 직진확률={p_straight:.3f})")


if __name__ == "__main__":
    main()
