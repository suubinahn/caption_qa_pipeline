"""
CSLS (Cross-domain Similarity Local Scaling) 구현
====================================================

[허브(hub) 문제란?]
CLIP 임베딩 공간에서는 가끔 어떤 텍스트(또는 이미지)가 실제 내용과
상관없이 "거의 모든 상대"와 두루 유사도가 높게 나오는 경우가 있다.
이런 항목을 허브(hub)라고 부른다. 예를 들어 표현이 애매하거나 아주
일반적인 단어로 이루어진 문장은, 특정 이미지 하나만 잘 설명하는 게
아니라 여러 이미지와 두루 비슷하게 높은 코사인 유사도를 갖게 된다.
이러면 "유사도가 제일 높으니까 이게 정답이다" 같은 단순 랭킹/margin
비교가 왜곡된다 - 정말 잘 맞아서 점수가 높은 건지, 원래 이 항목이
후하게 점수를 잘 주는 허브라서 높은 건지 구분이 안 되기 때문이다.

CSLS는 원래 비지도 단어 번역 논문(Conneau et al., 2018, "Word
Translation Without Parallel Data")에서 제안된 방법이고, 이후
이미지-텍스트 검색(image-text retrieval) 평가에도 널리 쓰인다.
아이디어는 간단하다:

    "어떤 항목이 원래 두루 인기가 많다(허브다)면, 그 인기만큼
     점수를 깎아서 상쇄하자."

[공식]
    CSLS(x, y) = 2 * cos(x, y) - r_T(x) - r_S(y)

    - cos(x, y): 이미지 x와 텍스트 y의 원래 코사인 유사도
    - r_T(x)   : 이미지 x가 "텍스트 축(target)"에서 가장 가까운 상위 K개
                 텍스트들과 갖는 평균 유사도. = "이미지 x가 평균적으로
                 얼마나 후하게 점수를 주는 경향이 있는지"
    - r_S(y)   : 텍스트 y가 "이미지 축(source)"에서 가장 가까운 상위 K개
                 이미지들과 갖는 평균 유사도. = "텍스트 y가 얼마나 인기
                 많은 허브인지"

직관: 원래 유사도(2*cos)에서 "이 이미지가 원래 후한 편인지(r_T)"와
"이 텍스트가 원래 인기 많은 허브인지(r_S)"를 각각 빼준다. 그러면
"이 특정 이미지-텍스트 쌍이 서로에게 유독 잘 맞는가"라는, 상대적으로
공정한 신호만 남는다. 지난 대화에서 설명한 "이웃 평균 유사도 빼기"가
정확히 이 r_T, r_S 두 항을 계산해서 빼는 과정이다.
"""

from __future__ import annotations

import numpy as np


def _mean_of_top_k(sim: np.ndarray, k: int, axis: int) -> np.ndarray:
    """유사도 행렬에서 axis 방향으로, 각 줄(행 또는 열)마다 가장 큰 k개
    값의 평균을 구한다. (이게 바로 "K-최근접 이웃과의 평균 유사도" 계산이다)

    axis=1 (행 방향, r_T 계산용):
        각 이미지(행)마다 그 행 안에서 가장 비슷한 상위 k개 텍스트를 골라
        평균을 낸다. 결과는 이미지 개수만큼의 1차원 배열 -> r_T(x)
    axis=0 (열 방향, r_S 계산용):
        각 텍스트(열)마다 그 열 안에서 가장 비슷한 상위 k개 이미지를 골라
        평균을 낸다. 결과는 텍스트 개수만큼의 1차원 배열 -> r_S(y)
    """
    if axis not in (0, 1):
        raise ValueError("axis must be 0 or 1")

    axis_size = sim.shape[axis]
    k = max(1, min(k, axis_size))  # k가 그 축의 크기보다 크면 자동으로 축 크기까지 줄인다

    # np.partition(arr, -k)는 배열을 완전히 정렬하지 않고, "가장 큰 k개"만
    # 뒤쪽에 모아준다 (전체 정렬보다 빠름). 우리는 정확한 순서가 아니라
    # 상위 k개의 평균만 필요하므로 이걸로 충분하다.
    if axis == 1:
        top_k = np.partition(sim, -k, axis=1)[:, -k:]  # (n_images, k)
        return top_k.mean(axis=1)  # -> (n_images,)
    else:
        top_k = np.partition(sim, -k, axis=0)[-k:, :]  # (k, n_texts)
        return top_k.mean(axis=0)  # -> (n_texts,)


def csls(sim_matrix: np.ndarray, k: int = 10) -> np.ndarray:
    """이미지 x 텍스트 코사인 유사도 행렬에 CSLS 보정을 적용한다.

    Parameters
    ----------
    sim_matrix : np.ndarray, shape (n_images, n_texts)
        원본 코사인 유사도 행렬. sim_matrix[i, j] = 이미지 i와 텍스트 j의 코사인 유사도.
        (clip_score.compute_similarity_matrix()로 만들 수 있다)
    k : int
        이웃 평균을 낼 때 볼 최근접 이웃 개수. 원 논문 기본값은 10이지만,
        데이터셋이 작으면(이미지 또는 텍스트 개수가 k보다 적으면) 해당 축의
        크기까지 자동으로 줄어든다.

    Returns
    -------
    np.ndarray, shape (n_images, n_texts)
        CSLS로 보정된 점수 행렬. 원본 코사인 유사도(-1~1)와 달리 범위가
        더 넓어질 수 있다(2*cos - r_T - r_S 형태라서). 하지만 절대값보다는
        "같은 이미지 행 안에서 어떤 텍스트가 상대적으로 점수가 높은가"를
        비교하는 용도로 쓰는 건 원래 코사인 유사도와 동일하다.
    """
    if sim_matrix.ndim != 2:
        raise ValueError(f"sim_matrix는 2차원(이미지 x 텍스트)이어야 합니다. 현재 shape={sim_matrix.shape}")

    # r_T(x): 각 이미지(행)마다, 텍스트 축(열) 방향 상위 k개 이웃과의 평균 유사도. shape: (n_images,)
    r_T = _mean_of_top_k(sim_matrix, k=k, axis=1)

    # r_S(y): 각 텍스트(열)마다, 이미지 축(행) 방향 상위 k개 이웃과의 평균 유사도. shape: (n_texts,)
    r_S = _mean_of_top_k(sim_matrix, k=k, axis=0)

    # 브로드캐스팅으로 모든 (i, j) 쌍에 대해 r_T[i]와 r_S[j]를 동시에 뺀다.
    #   r_T[:, None] : (n_images, 1)로 reshape -> 각 "행" 전체에 같은 값을 뺌
    #   r_S[None, :] : (1, n_texts)로 reshape -> 각 "열" 전체에 같은 값을 뺌
    return 2 * sim_matrix - r_T[:, None] - r_S[None, :]


if __name__ == "__main__":
    # 허브 문제를 인위적으로 재현한 3x3 장난감 예제로 동작을 확인해본다.
    # text1이 "허브"라서 모든 이미지와 두루 유사도가 높다고 가정.
    example = np.array([
        [0.35, 0.60, 0.10],  # image0: text1과 원래 가장 비슷함 (0.60)
        [0.20, 0.55, 0.32],  # image1: text1과 원래 가장 비슷함 (0.55)
        [0.15, 0.50, 0.48],  # image2: text1(0.50)과 text2(0.48)가 원래는 거의 비슷함
    ])
    print("원본 코사인 유사도:\n", example)

    result = csls(example, k=2)
    print("\nCSLS 보정 후 (k=2):\n", np.round(result, 3))

    print("\nimage2 행 비교: 원본에서는 text1(0.50) > text2(0.48)로 text1이 근소 우위였지만,")
    print("text1이 다른 이미지들과도 두루 비슷도가 높은 '허브'라서 보정 후에는")
    print(f"text1={result[2,1]:.3f}, text2={result[2,2]:.3f} 로 순위가 뒤집힌다 (허브 보정 효과).")
