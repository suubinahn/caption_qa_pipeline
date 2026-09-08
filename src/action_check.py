"""
행동 단위(동사) 검증 모듈
==========================

[설계 원칙 - object_check.py와 동일한 패턴을 동사에 적용]
8~9단계에서 "캡션의 핵심 명사만 떼어내 별도로 검증"했더니 문장 전체
점수가 놓친 객체 오류(개↔고양이)를 잡아낼 수 있었다. 이번엔 같은 원리를
동사(행동)에 적용한다: "말이 달린다" -> "running"만 떼어내서, 그 행동이
실제로 영상에 있는지 독립적으로 물어본다.

[왜 주어(subject)를 같이 넣지 않는가]
처음엔 "horses running"처럼 주어+동사를 묶어서 물어보려 했다. 하지만
NLTK 품사 태깅만으로는 "누가 그 행동을 하는지"(문법적 주어)를 안정적으로
찾기 어렵다. 예를 들어 "A man in a red shirt is riding a bicycle"에서
동사 바로 앞 명사는 "shirt"이지, 진짜 주어인 "man"이 아니다(중간에
"in a red shirt"라는 전치사구가 끼어있기 때문). 이런 문제를 제대로
풀려면 의존구문분석(dependency parsing, 예: spaCy)이 필요한데, 이번
실험은 "간단한 품사 분석"이라는 범위를 지키기 위해 object_check.py와
완전히 동일하게 **동사 하나만 단독으로** 검증한다.

[동사 추출 방법]
NLTK 품사 태그 중 VBG(현재분사/동명사, 예: running, riding, standing)를
사용한다. 우리 캡션들은 전부 "~이 ~하고 있다" 형태의 현재진행형으로
쓰여 있어서, 핵심 행동은 거의 항상 VBG 태그로 잡힌다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import nltk

from inspector import inspect_caption, InspectionResult
from style_detect import detect_video_style, style_to_article

ACTION_TAGS = {"VBG"}  # 현재분사/동명사 - "~하고 있는 중" 행동을 나타냄


def extract_actions(caption: str) -> list[tuple[str, str]]:
    """캡션 문장에서 진행 중인 행동(동사)만 추출한다.
    (단어, 품사태그) 튜플의 리스트를 반환. object_check.extract_nouns()와
    완전히 동일한 구조 - 필터링하는 품사 태그 집합만 다르다.
    """
    tokens = nltk.word_tokenize(caption)
    tagged = nltk.pos_tag(tokens)
    return [(word.lower(), tag) for word, tag in tagged if tag in ACTION_TAGS]


def action_to_query(verb: str, style: str = "photo") -> str:
    """동사 하나를 "a {style} of {동사}" 형태의 짧은 검증용 문장으로 바꾼다.
    예: ("running", "photo") -> "a photo of running"

    object_check.noun_to_query()와 동일한 스타일 인식 구조를 그대로 쓴다 -
    카툰/일러스트 영상에서는 "a cartoon of running"처럼 도메인에 맞는
    문구를 자동으로 만들 수 있다.
    """
    article = style_to_article(style)
    return f"{article} {style} of {verb}"


def check_actions(
    video_path: Union[str, Path],
    caption: str,
    n_frames: int = 5,
    style: str = "photo",
) -> list[dict]:
    """캡션에서 동사(행동)를 전부 추출해, 각 행동이 영상에 실제로
    나타나는지 inspect_caption()으로 하나씩 독립 검증한다.
    """
    actions = extract_actions(caption)
    results = []
    for verb, tag in actions:
        query = action_to_query(verb, style=style)
        r: InspectionResult = inspect_caption(video_path, query, n_frames=n_frames)
        results.append({
            "word": verb,
            "tag": tag,
            "query": query,
            "score": r["score"],
            "verdict": r["verdict"],
            "best_frame_index": r["best_frame_index"],
        })
    return results


def check_actions_auto_style(
    video_path: Union[str, Path],
    caption: str,
    n_frames: int = 5,
) -> dict:
    """영상 스타일을 CLIP으로 먼저 자동 판별한 뒤, 그 스타일에 맞는
    템플릿으로 행동 검증을 수행한다 (object_check.check_objects_auto_style()과
    동일한 구조).
    """
    style_info = detect_video_style(video_path, n_frames=n_frames)
    detected_style = style_info["style"]

    actions = check_actions(video_path, caption, n_frames=n_frames, style=detected_style)

    return {
        "style": detected_style,
        "style_votes": style_info["vote_counts"],
        "actions": actions,
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    video = root / "data" / "videos" / "horse_herd.webm"
    caption = "A small group of horses is running together across a green pasture near a tree line."

    print(f"캡션: {caption}")
    for r in check_actions(video, caption):
        print(f"  [{r['tag']}] {r['word']:12s} -> \"{r['query']}\"  score={r['score']:.4f}  verdict={r['verdict']}")
