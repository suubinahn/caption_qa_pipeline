"""
객체 단위(명사) 검증 모듈
==========================

[왜 필요한가]
7단계에서 발견한 문제: `dog_park`의 "개 세 마리가 논다"를 "고양이 세
마리가 논다"로 바꾼 오답이 BLIP-ITM 문장 전체 점수로는 0.9933이라는
높은 확신을 받아 PASS로 잘못 판정됐다. 문장 전체를 하나의 벡터/확률로
뭉뚱그려 채점하면, "행동/배경/구도가 전부 똑같고 명사 하나만 다른" 경우
그 차이가 희석돼 버린다 - 문장의 나머지 90%가 다 맞으니 점수도 전체적으로
높게 나오는 것이다.

그래서 문장 전체 점수와 별개로, **캡션 안의 핵심 명사(객체)만 따로 뽑아서
"이 객체가 실제로 이미지에 있는가"를 독립적으로 채점**한다. "개"라는
단어 하나만 놓고 보면, 그 객체가 진짜 있는지 없는지는 문장 전체보다
훨씬 뚜렷하게 갈릴 것이라는 가설이다.

[명사 추출 방법]
NLTK의 averaged_perceptron_tagger로 품사 태깅을 한다. 복잡한 의존구문
분석(dependency parsing)까지는 필요 없고, "명사(NN/NNS/NNP/NNPS) 태그가
붙은 단어만 뽑는다"는 단순한 규칙으로 충분하다 - 우리 캡션들은 문장
구조가 단순해서 이 정도로도 핵심 객체를 잘 잡아낸다.

[객체 존재 여부 채점 방법]
새로운 채점 함수를 만들지 않고, **이미 검증된 inspect_caption()을 그대로
재사용**한다. "이 명사가 이미지에 있는가"라는 질문을, "a photo of a
{명사}" 같은 짧은 캡션으로 바꿔서 inspect_caption()에 그대로 넣으면 된다.
5프레임 추출 + max pooling + BLIP-ITM 채점이라는, 이미 검증된 파이프라인을
그대로 재사용하는 것이다 (같은 프레임 샘플링 전략을 써야 문장 단위
점수와 공정하게 비교할 수 있다).
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import nltk

from inspector import inspect_caption, InspectionResult
from style_detect import detect_video_style, style_to_article

NOUN_TAGS = {"NN", "NNS", "NNP", "NNPS"}
PLURAL_TAGS = {"NNS", "NNPS"}

# 문장 구조를 이루는 데만 쓰이고 "화면에 있는지 없는지"를 검증할 대상으로는
# 의미가 없는 명사들 (예: "scene", "temperature"는 명사이긴 하지만 이미지에
# "있다/없다"를 눈으로 확인할 수 있는 구체적 사물이 아니다). 완벽한 필터는
# 아니지만, 이 정도만 걸러도 결과가 훨씬 깔끔해진다.
ABSTRACT_NOUN_STOPLIST = {
    "scene", "temperature", "check", "way", "bit", "moment", "time",
}


def extract_nouns(caption: str) -> list[tuple[str, str]]:
    """캡션 문장에서 명사만 추출한다. (단어, 품사태그) 튜플의 리스트를 반환.

    NLTK pos_tag는 문맥을 보고 품사를 추정하므로, 같은 "dogs"라도 문장 안에서
    동사가 아니라 명사로 잘 인식된다. 태그 의미: NN=단수보통명사,
    NNS=복수보통명사, NNP=단수고유명사, NNPS=복수고유명사.
    """
    tokens = nltk.word_tokenize(caption)
    tagged = nltk.pos_tag(tokens)
    nouns = [
        (word.lower(), tag) for word, tag in tagged
        if tag in NOUN_TAGS and word.lower() not in ABSTRACT_NOUN_STOPLIST
    ]
    return nouns


def noun_to_query(word: str, tag: str, style: str = "photo") -> str:
    """명사 하나를 "a {style} of a {명사}" 형태의 짧은 검증용 문장으로 바꾼다.

    style 인자를 받는 이유: 9단계에서 확인했듯, 실사 영상에는 "photo"가
    맞지만 애니메이션/카툰 영상에 "a photo of a X"를 물으면 도메인이 안 맞아
    객체가 실제로 있어도 점수가 낮게 나온다. style_detect.detect_video_style()
    로 미리 판별한 스타일을 여기 넣어주면 그 도메인에 맞는 문구를 만든다.

    복수형(NNS/NNPS)이면 관사 없이 "a {style} of {복수명사}" 형태를 쓴다.
    """
    article = style_to_article(style)
    if tag in PLURAL_TAGS:
        return f"{article} {style} of {word}"
    return f"{article} {style} of a {word}"


def check_objects(
    video_path: Union[str, Path],
    caption: str,
    n_frames: int = 5,
    style: str = "photo",
) -> list[dict]:
    """캡션에서 명사를 전부 추출해, 각 명사가 영상에 실제로 있는지
    inspect_caption()으로 하나씩 독립 검증한다.

    style을 고정값으로 받는 버전 - 스타일을 이미 알고 있거나(또는 기존처럼
    "photo"로 고정해서 쓰고 싶을 때) 사용. 스타일을 자동으로 판별하고
    싶으면 check_objects_auto_style()을 대신 쓴다.

    반환값: 명사별로 {word, tag, query, score, verdict, best_frame_index} 딕셔너리 리스트.
    """
    nouns = extract_nouns(caption)
    results = []
    for word, tag in nouns:
        query = noun_to_query(word, tag, style=style)
        r: InspectionResult = inspect_caption(video_path, query, n_frames=n_frames)
        results.append({
            "word": word,
            "tag": tag,
            "query": query,
            "score": r["score"],
            "verdict": r["verdict"],
            "best_frame_index": r["best_frame_index"],
        })
    return results


def check_objects_auto_style(
    video_path: Union[str, Path],
    caption: str,
    n_frames: int = 5,
) -> dict:
    """영상 스타일을 CLIP으로 먼저 자동 판별한 뒤, 그 스타일에 맞는
    템플릿으로 객체 검증을 수행한다 (9단계에서 새로 추가된 엔드투엔드 버전).

    반환값: {"style": 판별된 스타일, "style_votes": 프레임별 투표 결과,
             "objects": check_objects()와 같은 형식의 리스트}
    """
    style_info = detect_video_style(video_path, n_frames=n_frames)
    detected_style = style_info["style"]

    objects = check_objects(video_path, caption, n_frames=n_frames, style=detected_style)

    return {
        "style": detected_style,
        "style_votes": style_info["vote_counts"],
        "objects": objects,
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    video = root / "data" / "videos" / "dog_park.ogv"
    caption = "Three dogs are playing together outdoors in a park with bare trees and fallen leaves on the ground."

    print(f"캡션: {caption}")
    for r in check_objects(video, caption):
        print(f"  [{r['tag']:4s}] {r['word']:12s} -> \"{r['query']}\"  score={r['score']:.4f}  verdict={r['verdict']}")
