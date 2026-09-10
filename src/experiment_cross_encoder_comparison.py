"""
Cross-encoder 모델 비교 실험 (BLIP-ITM-base vs BLIP-ITM-large vs BLIP-2-ITM)
================================================================================

지금 파이프라인(②검증)이 쓰는 BLIP-ITM-base가 "여러 cross-encoder 후보 중
비교해서 고른 것"이 아니라 "cross-encoder가 필요하다는 걸 확인하고 접근성
좋은 걸 써봤더니 잘 통해서 그대로 쓴 것"이었다는 걸 확인한 뒤, 진짜 다른
설계의 cross-encoder였으면 더 나았을지 궁금해져서 확인해본다.

이 파일은 완전히 독립적이다 - blip_itm.py, search_and_verify.py 등 기존
파이프라인 파일은 하나도 import하지 않고, 모델 로딩부터 채점까지 이 파일
안에서 전부 새로 한다. 기존 파이프라인에 어떤 영향도 주지 않는다.

비교 대상 (전부 Hugging Face에서 바로 받을 수 있는 것만):
  1) Salesforce/blip-itm-base-coco  - 지금 파이프라인이 실제로 쓰는 것
  2) Salesforce/blip-itm-large-coco - 같은 BLIP 계열, 더 큰 버전
  3) Salesforce/blip2-itm-vit-g     - BLIP-2, 아예 다른 설계(Q-Former)

지금 파이프라인에서 가장 약한 지점(motion 판정, holdout 68.5%)에 초점을
맞춰서, 회사 실주행 클립의 motion 참/거짓 문장 쌍으로 세 모델을 비교한다.
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from frame_extract import extract_frames  # 프레임 추출만 재사용 - 비교 대상(cross-encoder)이 아니므로 무관

N_CLIPS = 20
N_FRAMES = 5
SEED = 3

MOTION_TEMPLATES = {
    "moving straight": "The vehicle is moving straight.",
    "turning right": "The vehicle is turning right.",
    "turning left": "The vehicle is turning left.",
    "stop": "The vehicle is stopped.",
    "curved lane driving": "The vehicle is driving through a curve.",
}


# ---------------------------------------------------------------------------
# 모델별 채점 함수 - 전부 이 파일 안에서 독립적으로 로딩한다
# ---------------------------------------------------------------------------

def load_blip_itm(model_name: str):
    from transformers import BlipProcessor, BlipForImageTextRetrieval
    processor = BlipProcessor.from_pretrained(model_name, use_fast=True)
    model = BlipForImageTextRetrieval.from_pretrained(model_name, use_safetensors=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return processor, model, device


def score_blip_itm(processor, model, device, image: Image.Image, text: str) -> float:
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
        # itm_score: (1, 2) logits [불일치, 일치] - softmax 후 '일치' 확률
        probs = F.softmax(out.itm_score, dim=1)
    return probs[0, 1].item()


def load_blip2_itm(model_name: str = "Salesforce/blip2-itm-vit-g"):
    from transformers import Blip2Processor, Blip2ForImageTextRetrieval
    processor = Blip2Processor.from_pretrained(model_name)
    model = Blip2ForImageTextRetrieval.from_pretrained(model_name, torch_dtype=torch.float16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return processor, model, device


def score_blip2_itm(processor, model, device, image: Image.Image, text: str) -> float:
    inputs = processor(images=image, text=text, return_tensors="pt").to(device, torch.float16)
    with torch.no_grad():
        out = model(**inputs, use_image_text_matching_head=True)
        probs = F.softmax(out.logits_per_image, dim=1)
    return probs[0, 1].item()


# ---------------------------------------------------------------------------
# 평가 로직 - BLIP류(base/large)와 BLIP-2는 채점 함수 시그니처가 달라서
# (score_blip_itm vs score_blip2_itm) 각각 별도 러너를 둔다.
# ---------------------------------------------------------------------------

def run_eval(name: str, score_fn, prepared) -> dict:
    print(f"\n[{name}] 채점 중...")
    t0 = time.time()
    margins, correct = [], 0
    for frames, true_sentence, false_sentence in prepared:
        true_score = max(score_fn(f, true_sentence) for f in frames)
        false_score = max(score_fn(f, false_sentence) for f in frames)
        margins.append(true_score - false_score)
        if true_score > false_score:
            correct += 1
    elapsed = time.time() - t0
    n = len(prepared)
    acc, avg_margin = correct / n, sum(margins) / n
    print(f"  accuracy(참>거짓)={acc*100:.1f}%  평균 margin={avg_margin:+.4f}  ({elapsed:.1f}초, {n}클립)")
    return {"name": name, "accuracy": acc, "avg_margin": avg_margin, "elapsed": elapsed}


def main():
    with open("outputs/search_full_dataset.json", encoding="utf-8") as f:
        all_clips = json.load(f)

    random.seed(SEED)
    sample = random.sample(all_clips, N_CLIPS)

    print(f"프레임/문장 준비 중... ({N_CLIPS}클립 x {N_FRAMES}프레임, motion 필드 참/거짓 문장 생성)")
    prepared = []
    for c in sample:
        video_path = Path(c["clip_dir"]) / "1_clip" / "5.mp4"
        if not video_path.exists():
            continue
        true_val = c["motion"]
        if true_val not in MOTION_TEMPLATES:
            continue
        false_val = random.choice([v for v in MOTION_TEMPLATES if v != true_val])
        frames = extract_frames(str(video_path), n_frames=N_FRAMES)
        prepared.append((frames, MOTION_TEMPLATES[true_val], MOTION_TEMPLATES[false_val]))
    print(f"준비 완료: {len(prepared)}개 클립\n")

    results = []

    # 1) BLIP-ITM-base (지금 파이프라인이 실제로 쓰는 것)
    processor, model, device = load_blip_itm("Salesforce/blip-itm-base-coco")
    score_fn = lambda img, text: score_blip_itm(processor, model, device, img, text)
    results.append(run_eval("BLIP-ITM-base (현재 파이프라인)", score_fn, prepared))
    del model
    torch.cuda.empty_cache()

    # 2) BLIP-ITM-large (같은 계열, 더 큼)
    processor, model, device = load_blip_itm("Salesforce/blip-itm-large-coco")
    score_fn = lambda img, text: score_blip_itm(processor, model, device, img, text)
    results.append(run_eval("BLIP-ITM-large (같은 계열, 더 큼)", score_fn, prepared))
    del model
    torch.cuda.empty_cache()

    # 3) BLIP-2-ITM (다른 설계, Q-Former)
    processor, model, device = load_blip2_itm()
    score_fn = lambda img, text: score_blip2_itm(processor, model, device, img, text)
    results.append(run_eval("BLIP-2-ITM (다른 설계, Q-Former)", score_fn, prepared))

    print("\n" + "=" * 60)
    print("요약")
    for r in results:
        print(f"  {r['name']:35s} accuracy={r['accuracy']*100:5.1f}%  margin={r['avg_margin']:+.4f}")


if __name__ == "__main__":
    main()
