"""
CLIP을 이용한 이미지-텍스트 코사인 유사도 채점 모듈
====================================================

[CLIP이 뭔가요?]
CLIP(Contrastive Language-Image Pre-training)은 OpenAI가 공개한 모델로,
"이미지"와 "그 이미지를 설명하는 텍스트"를 같은 벡터 공간(embedding space)에
투영하도록 학습되었다. 즉,
  - 이미지 인코더(vision encoder)가 이미지를 하나의 벡터(임베딩)로 바꾸고
  - 텍스트 인코더(text encoder)가 문장을 같은 차원의 벡터로 바꾼다.
학습 과정에서 "짝이 맞는(진짜 그 이미지를 설명하는) 이미지-텍스트 쌍"의
벡터는 서로 가깝게, "짝이 안 맞는 쌍"은 서로 멀게 만드는 대조 학습
(contrastive learning)을 수행했다. 그 결과 두 벡터 사이의 코사인 유사도
(cosine similarity)가 높을수록 "이 이미지와 이 문장이 잘 맞는다"는 뜻이 된다.

이 성질을 이용하면 "VLM이 생성한 캡션이 실제 영상 장면과 얼마나 맞는지"를
사람이 일일이 보지 않고도 정량적인 점수로 근사할 수 있다. 이게 이 프로젝트의
핵심 아이디어다.

[코사인 유사도란?]
두 벡터 a, b 사이의 코사인 유사도 = (a·b) / (|a| * |b|)
벡터의 "방향"이 얼마나 비슷한지를 -1~1 사이 값으로 나타낸다.
(CLIP의 경우 실제로는 대부분 0~0.4 사이 값이 나오는 경우가 많다 - 아래 참고)

[openai/clip-vit-base-patch32 모델 구조]
- "ViT-B/32": Vision Transformer(ViT) 기반 이미지 인코더, 이미지를 32x32
  픽셀 크기의 패치(patch)로 잘라 Transformer에 입력한다. "base"는 모델
  크기(파라미터 수)가 중간 크기라는 뜻(large보다 작고 빠름).
- 텍스트 인코더는 GPT류와 비슷한 Transformer 구조.
- HuggingFace transformers 라이브러리의 CLIPModel + CLIPProcessor로 바로
  불러와 쓸 수 있다.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"

# 전역에 모델을 한 번만 로드해서 재사용한다(매번 새로 로드하면 느리고 GPU 메모리도 낭비됨).
_model = None
_processor = None
_device = None


def load_model():
    """CLIP 모델과 전처리기(processor)를 (한 번만) 로드해서 반환한다.

    - CLIPModel: 실제 신경망 가중치를 담은 PyTorch 모델.
    - CLIPProcessor: 이미지 크기 조정/정규화 + 텍스트 토큰화를 모델이 원하는
      형식으로 맞춰주는 전처리 도구. 모델마다 학습 시 사용한 전처리 방식이
      다르기 때문에, 반드시 같은 모델 이름으로 만든 processor를 써야 한다.
    """
    global _model, _processor, _device
    if _model is not None:
        return _model, _processor, _device

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[clip_score] 모델 로딩 중... (device={_device})")

    # use_safetensors=True: 모델 가중치를 .bin(pickle 기반, 임의 코드 실행 위험이
    # 있다고 알려진 포맷) 대신 .safetensors(순수 텐서 데이터만 담는 안전한 포맷)로
    # 받아오도록 강제한다. 최신 transformers는 구버전 torch에서 .bin 로딩 시
    # 보안 경고와 함께 아예 막아버리는데, 이 옵션으로 그 문제를 피할 수 있다.
    _model = CLIPModel.from_pretrained(MODEL_NAME, use_safetensors=True)
    _model.to(_device)
    _model.eval()  # 추론 모드: dropout 등 학습 전용 레이어를 비활성화

    _processor = CLIPProcessor.from_pretrained(MODEL_NAME, use_fast=True)

    return _model, _processor, _device


def compute_similarity(image: Union[str, Path, Image.Image], text: str) -> float:
    """이미지 1장 + 문장 1개를 받아 코사인 유사도 점수(float)를 반환한다.

    파라미터:
      image: 이미지 파일 경로(str/Path) 또는 이미 로드된 PIL.Image 객체
      text : 이미지 내용을 설명하는 (영어) 문장

    반환값:
      -1.0 ~ 1.0 사이의 코사인 유사도. 실제로는 CLIP 특성상 대체로
      0.15 ~ 0.35 구간에 값이 몰리는 경우가 많다(벡터들이 임베딩 공간의
      한쪽에 뭉쳐있는 "콘 효과(cone effect)"라 불리는 현상 때문).
      그래서 "절대값"보다는 "같은 이미지에 대해 여러 문장을 비교했을 때
      상대적으로 어떤 문장이 더 높은 점수를 받는가"가 더 중요한 지표다.
      (참고: 이 상대비교의 한계를 보정하는 것이 다음 단계인 CSLS다.)
    """
    model, processor, device = load_model()

    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")

    # processor가 이미지 정규화 + 텍스트 토큰화를 한 번에 처리해서
    # 모델에 바로 넣을 수 있는 텐서(tensor) 딕셔너리를 만들어준다.
    # padding=True: 여러 문장을 배치로 넣을 때 길이를 맞춰줌 (여기선 문장 1개뿐이라도 안전하게 켜둠)
    inputs = processor(text=[text], images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():  # 추론만 할 것이므로 그래디언트 계산을 꺼서 속도/메모리 절약
        outputs = model(**inputs)
        image_embeds = outputs.image_embeds  # shape: (1, 512) - 이미지 벡터
        text_embeds = outputs.text_embeds    # shape: (1, 512) - 텍스트 벡터

        # L2 정규화: 벡터의 길이(norm)를 1로 만들어서, 이후 내적(dot product)이
        # 곧바로 코사인 유사도가 되도록 한다. (코사인 유사도 = 내적 / (길이*길이))
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        # 정규화된 두 벡터의 내적 = 코사인 유사도
        similarity = (image_embeds @ text_embeds.T).item()

    return similarity


def compute_similarity_matrix(
    images: List[Union[str, Path, Image.Image]],
    texts: List[str],
) -> np.ndarray:
    """여러 이미지 x 여러 텍스트의 코사인 유사도 행렬을 한 번에 계산한다.

    왜 compute_similarity()를 (이미지 개수 x 텍스트 개수)번 반복 호출하지
    않고 이 함수를 따로 만들었는가?

    - compute_similarity()를 이중 for문으로 돌리면, 예를 들어 이미지 8장 x
      텍스트 32개일 때 CLIP 모델을 8*32 = 256번 호출해야 한다. 그런데 같은
      이미지를 여러 문장과 비교할 때마다 그 이미지를 매번 다시 인코딩하는
      건 낭비다 (이미지 임베딩은 문장과 무관하게 한 번만 계산하면 됨).
    - 이 함수는 이미지 임베딩을 "한 번의 배치"로 8개 다 계산하고, 텍스트
      임베딩도 "한 번의 배치"로 32개 다 계산한 다음, 두 임베딩 행렬을
      행렬곱(matrix multiplication) 한 번으로 곱해서 8x32 유사도 행렬을
      통째로 얻는다. 모델 forward 호출 횟수가 256번 -> 2번(이미지 배치 1번
      + 텍스트 배치 1번)으로 줄어드는 셈이다.

    Parameters
    ----------
    images: 이미지 경로 또는 PIL.Image 객체의 리스트 (길이 = n_images)
    texts : 문장(캡션) 문자열의 리스트 (길이 = n_texts)

    Returns
    -------
    np.ndarray, shape (n_images, n_texts)
        sim_matrix[i, j] = images[i]와 texts[j]의 코사인 유사도.
    """
    model, processor, device = load_model()

    pil_images = [
        Image.open(p).convert("RGB") if isinstance(p, (str, Path)) else p
        for p in images
    ]

    with torch.no_grad():
        # --- 이미지 임베딩을 배치로 한 번에 계산 ---
        image_inputs = processor(images=pil_images, return_tensors="pt")
        image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
        image_embeds = model.get_image_features(**image_inputs)  # (n_images, 512)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)  # L2 정규화

        # --- 텍스트 임베딩을 배치로 한 번에 계산 ---
        text_inputs = processor(text=texts, return_tensors="pt", padding=True)
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        text_embeds = model.get_text_features(**text_inputs)  # (n_texts, 512)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)  # L2 정규화

        # 정규화된 두 임베딩 행렬을 곱하면, 결과의 [i, j] 원소가 바로
        # images[i]와 texts[j]의 코사인 유사도가 된다.
        # (n_images, 512) @ (512, n_texts) -> (n_images, n_texts)
        sim_matrix = image_embeds @ text_embeds.T

    return sim_matrix.cpu().numpy()


if __name__ == "__main__":
    # 간단한 동작 테스트: data/frames 안의 첫 번째 이미지에 대해
    # 그럴듯한 문장 하나와 엉뚱한 문장 하나를 비교해본다.
    root = Path(__file__).resolve().parent.parent
    frame_dir = root / "data" / "frames"
    sample_img = sorted(frame_dir.glob("*.jpg"))[0]

    print(f"테스트 이미지: {sample_img.name}")
    for text in [
        "a photo of an animal",
        "a photo of a car engine on a workbench",
    ]:
        score = compute_similarity(sample_img, text)
        print(f"  '{text}' -> similarity = {score:.4f}")
