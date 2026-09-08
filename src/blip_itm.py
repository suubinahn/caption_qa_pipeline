"""
BLIP ITM (Image-Text Matching) 채점 모듈
==========================================

[CLIP과 구조적으로 무엇이 다른가?]

CLIP은 "듀얼 인코더(dual-encoder)" 구조다: 이미지 인코더와 텍스트
인코더가 서로를 한 번도 쳐다보지 않고 완전히 독립적으로 각자 벡터
(임베딩)를 만든 다음, 맨 마지막에 그 두 벡터의 코사인 유사도만 계산한다.

    이미지 -> [이미지 인코더] -> 벡터 A  ┐
                                          ├─ 코사인 유사도
    텍스트 -> [텍스트 인코더] -> 벡터 B  ┘

이 구조 덕분에 CLIP은 "이미지 임베딩을 미리 다 계산해두고, 나중에
아무 문장과나 빠르게 비교"하는 대규모 검색(retrieval)에 매우 유리하다
(2단계에서 image_embeds @ text_embeds.T 행렬곱 한 번으로 8x26 유사도
행렬을 통째로 뽑아낼 수 있었던 것도 바로 이 구조 덕분이다).

하지만 대가가 있다: 이미지 인코더가 텍스트를 전혀 못 보고, 텍스트
인코더도 이미지를 전혀 못 본 채 각자 벡터를 만들기 때문에, "이 문장의
이 단어가 이미지의 이 부분과 실제로 맞아떨어지는지" 같은 세밀한 대조를
할 기회 자체가 없다. 결과물인 벡터 하나에 "이미지 전체의 분위기"가
뭉뚱그려 들어가 있을 뿐이라, "자전거를 타고 있다 vs 자전거 옆에
서있다"처럼 전체적인 구도(사람+자전거+거리)는 거의 같고 세부 동작만
다른 경우 두 벡터가 비슷해져 버려서 잘 구분하지 못한다.

BLIP의 ITM(Image-Text Matching) 헤드는 반대로 "크로스 인코더
(cross-encoder)" 구조를 쓴다:

    이미지 -> [비전 인코더] -> 이미지 패치 벡터 시퀀스 (문장이 볼 수 있음)
                                        │
                                        ▼ cross-attention (매 레이어마다)
    텍스트 -> [텍스트 인코더, cross-attention 포함] -> [CLS] 벡터 -> itm_head(2class) -> 매칭 확률

    1) 이미지를 비전 인코더(ViT)로 패치 임베딩 시퀀스로 만든다.
    2) 텍스트를 토큰 임베딩 시퀀스로 만들면서, 텍스트 쪽 Transformer의
       각 레이어에 "cross-attention" 층을 끼워 넣는다. 이 층은 텍스트의
       각 단어 토큰이 "이미지 패치 시퀀스 전체"를 직접 들여다보고
       "지금 이 단어와 가장 관련 있는 이미지 영역이 어디인지"를 매
       레이어마다 다시 계산하게 해준다. (반대로 이미지 쪽은 텍스트를
       보지 않고 자기 패치들끼리만 self-attention 한다 - 비대칭 구조.)
    3) 문장 전체를 요약하는 [CLS] 토큰의 최종 벡터에 작은 분류기
       (itm_head, "안 맞음"/"맞음" 2개 클래스)를 얹어서 "이 이미지와 이
       문장이 진짜 매칭되는 쌍인지"를 직접 이진 분류로 판단한다.

즉 CLIP은 "두 벡터가 최종적으로 얼마나 비슷한 방향을 가리키는가"라는
한 번의 요약된 비교만 하고, BLIP-ITM은 "문장의 단어 하나하나를 이미지의
어느 부분과 대응시켜야 하는지 다시 짚어가며" 판단한다. 이 차이 때문에
BLIP-ITM이 "타는 중 vs 서있는 중"처럼 이미지 전체 분위기(피사체, 배경,
색감)는 거의 동일하지만 동작만 다른 hard negative를 훨씬 더 잘 잡아낼
잠재력을 갖는다.

[대가: 왜 CSLS처럼 유사도 "행렬"을 한 번에 못 뽑는가]
크로스 인코더는 이미지-텍스트 "쌍"마다 cross-attention을 다시 계산해야
하므로, 이미지 임베딩을 텍스트와 무관하게 미리 계산해서 재사용하는 게
원천적으로 불가능하다. 그래서 CLIP처럼 "이미지 임베딩 행렬 x 텍스트
임베딩 행렬"을 한 번의 행렬곱으로 처리할 수 없고, 비교하고 싶은 (이미지,
텍스트) 쌍의 개수만큼 매번 모델 forward pass를 돌려야 한다 - 느리지만
더 세밀하다는 트레이드오프.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import BlipForImageTextRetrieval, BlipProcessor

MODEL_NAME = "Salesforce/blip-itm-base-coco"

_model = None
_processor = None
_device = None


def load_model():
    """BLIP-ITM 모델과 전처리기를 (한 번만) 로드해서 반환한다."""
    global _model, _processor, _device
    if _model is not None:
        return _model, _processor, _device

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[blip_itm] 모델 로딩 중... (device={_device})")

    _processor = BlipProcessor.from_pretrained(MODEL_NAME, use_fast=True)
    # use_safetensors=True: clip_score.py와 동일한 이유(구버전 torch에서
    # .bin 체크포인트 로딩이 보안상 막혀있어 안전한 포맷을 명시적으로 요청)
    _model = BlipForImageTextRetrieval.from_pretrained(MODEL_NAME, use_safetensors=True)
    _model.to(_device)
    _model.eval()

    return _model, _processor, _device


def compute_itm_score(image: Union[str, Path, Image.Image], text: str) -> float:
    """이미지 1장 + 문장 1개를 받아 BLIP의 ITM 매칭 확률(0~1)을 반환한다.

    CLIP의 compute_similarity()가 "-1~1 범위의 코사인 유사도"를 돌려주는
    것과 달리, 이 함수는 "이 이미지와 이 문장이 매칭되는 쌍일 확률"이라는
    명확한 의미를 가진 0~1 사이 값을 돌려준다 (itm_head가 2-클래스
    분류기이고, 그 출력에 softmax를 취해 "매칭" 클래스의 확률만 뽑기 때문).

    동작 원리:
      1) processor가 이미지와 텍스트를 모델 입력 형식으로 함께 변환한다
         (CLIP과 달리 이미지·텍스트를 한 번의 processor 호출에 같이 넣는다 -
         내부적으로 두 모달리티가 나중에 cross-attention으로 얽히기 때문에
         애초에 "한 쌍"으로 취급하는 것이 자연스럽다).
      2) model(**inputs, use_itm_head=True)로 forward pass를 돌리면
         outputs.itm_score에 (1, 2) 크기의 로짓(logit)이 나온다.
         인덱스 0 = "매칭 안 됨" 클래스, 인덱스 1 = "매칭됨" 클래스.
      3) softmax로 로짓을 확률로 바꾸고, "매칭됨" 클래스(인덱스 1)의
         확률만 꺼내서 반환한다.
    """
    model, processor, device = load_model()

    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")

    inputs = processor(images=image, text=text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, use_itm_head=True)
        # itm_score: shape (1, 2) - [배치, (안맞음 로짓, 맞음 로짓)]
        probs = F.softmax(outputs.itm_score, dim=1)
        match_prob = probs[0, 1].item()

    return match_prob


if __name__ == "__main__":
    # 간단한 동작 테스트: bicycle_ride 프레임에 대해
    # "타고 있다"(정답)와 "옆에 서있다"(행동만 바뀐 hard negative)를 비교.
    # CLIP은 이 둘을 거의 구분하지 못했다(margin이 음수에 가까웠음) -
    # BLIP-ITM은 얼마나 다르게 판단하는지 확인해본다.
    root = Path(__file__).resolve().parent.parent
    frame = root / "data" / "frames" / "bicycle_ride.jpg"

    correct = "A man in a red shirt is riding a bicycle down a city street."
    wrong_action = "A man in a red shirt is standing still next to a bicycle on a city street."

    score_correct = compute_itm_score(frame, correct)
    score_wrong = compute_itm_score(frame, wrong_action)

    print(f"정답 캡션 매칭 확률   : {score_correct:.4f}  ({correct})")
    print(f"행동만 바뀐 오답 확률 : {score_wrong:.4f}  ({wrong_action})")
    print(f"margin (정답-오답)   : {score_correct - score_wrong:+.4f}")
