"""
이미지 스타일(실사/카툰/일러스트) 자동 판별 모듈
====================================================

[왜 필요한가]
8단계에서 cooking_thermometer 샘플의 객체 검증("a photo of a burger")이
거의 작동하지 않는 걸 발견했다. 원인은 그 영상이 실사가 아니라 애니메이션
광고였기 때문 - "photo"라는 단어 자체가 이미지 도메인과 안 맞았다. 수동으로
"a cartoon of a X"로 바꿔봤더니 margin이 22배 커졌다. 이걸 매번 사람이
영상을 보고 판단하지 않고, **CLIP으로 스타일 자체를 자동 분류**해서
객체 검증 템플릿을 동적으로 고르도록 만든다.

[CLIP으로 스타일을 분류하는 원리]
CLIP은 원래 "제로샷 분류(zero-shot classification)"에 강점이 있는 모델이다.
이미지 하나와 "후보 문구 여러 개"의 코사인 유사도를 각각 구한 뒤, 유사도가
가장 높은 문구를 "이 이미지에 대한 설명으로 가장 그럴듯한 것"으로 고르면
분류가 된다 (별도의 분류기를 학습시킬 필요 없이, CLIP이 사전학습 때 배운
"이미지-텍스트 매칭" 능력을 그대로 재사용하는 것). 여기서는 "a photo",
"a cartoon", "an illustration" 세 가지를 후보로 놓고 이미지가 어느 쪽에
가장 가까운지 고른다.

softmax를 쓰는 이유: 코사인 유사도 자체는 절대적인 확률이 아니지만, 같은
이미지에 대해 여러 후보를 비교할 때는 "다른 후보들 대비 상대적으로 얼마나
그럴듯한가"가 중요하다. softmax(온도 파라미터로 스케일 조정)를 적용하면
"후보들 중 이게 몇 % 확률로 맞다고 보는지"처럼 해석하기 쉬운 형태가 된다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union
from collections import Counter

import numpy as np
from PIL import Image

from frame_extract import extract_frames
from clip_score import compute_similarity_matrix

# 스타일 후보. 각 항목: (라벨, CLIP에 물어볼 문구, 명사 앞에 붙일 관사)
STYLE_CANDIDATES = [
    ("photo", "a photo", "a"),
    ("cartoon", "a cartoon", "a"),
    ("illustration", "an illustration", "an"),
]

# softmax 온도(temperature). CLIP 코사인 유사도는 값의 편차가 작아서
# (대체로 0.2~0.35 사이) 그냥 softmax를 취하면 거의 균등분포가 나온다.
# 온도를 낮춰서(값을 크게 키워서) 차이를 증폭시켜야 "어느 게 더 그럴듯한지"가
# 뚜렷하게 드러난다. (CLIP 논문에서 학습 시 쓰는 logit_scale과 같은 역할)
SOFTMAX_TEMPERATURE = 0.05


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x / SOFTMAX_TEMPERATURE
    x = x - x.max()  # 오버플로 방지
    e = np.exp(x)
    return e / e.sum()


def classify_frame_style(image: Image.Image) -> tuple[str, dict[str, float]]:
    """이미지 1장의 스타일을 CLIP 제로샷 분류로 판별한다.
    반환값: (가장 그럴듯한 라벨, {라벨: 확률} 딕셔너리)
    """
    phrases = [c[1] for c in STYLE_CANDIDATES]
    labels = [c[0] for c in STYLE_CANDIDATES]

    sim = compute_similarity_matrix([image], phrases)[0]  # shape (3,)
    probs = _softmax(sim)

    best_idx = int(np.argmax(probs))
    return labels[best_idx], dict(zip(labels, probs.tolist()))


def detect_video_style(video_path: Union[str, Path], n_frames: int = 5) -> dict:
    """영상 전체의 스타일을 판별한다. 대표 프레임 n장을 뽑아 각각 분류한 뒤,
    다수결(majority vote)로 영상 전체의 스타일을 정한다.

    프레임 1장이 아니라 여러 장의 다수결을 쓰는 이유: 5단계에서 이미
    확인했듯 프레임 1장은 우연히 대표성이 없을 수 있다. 스타일 판별도
    같은 함정에 빠질 수 있으므로(예: 특정 프레임만 텍스트 자막이 크게
    나와서 오분류), 여러 프레임의 다수결로 안정성을 높인다.
    """
    frames = extract_frames(video_path, n_frames=n_frames)

    per_frame = [classify_frame_style(f) for f in frames]
    labels = [label for label, _ in per_frame]
    vote_counts = Counter(labels)
    majority_label = vote_counts.most_common(1)[0][0]

    return {
        "style": majority_label,
        "vote_counts": dict(vote_counts),
        "per_frame": [{"label": l, "probs": p} for l, p in per_frame],
    }


def style_to_article(style: str) -> str:
    for label, _, article in STYLE_CANDIDATES:
        if label == style:
            return article
    return "a"


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    for name in ["cooking_thermometer.webm", "bicycle_ride.ogv", "cat_string_toy.ogv"]:
        video = root / "data" / "videos" / name
        result = detect_video_style(video)
        print(f"\n{name}")
        print(f"  판별된 스타일: {result['style']}  (투표: {result['vote_counts']})")
        for i, pf in enumerate(result["per_frame"]):
            probs_str = ", ".join(f"{k}={v:.3f}" for k, v in pf["probs"].items())
            print(f"    프레임 {i}: {pf['label']:12s} ({probs_str})")
