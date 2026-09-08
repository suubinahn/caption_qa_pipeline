"""
캡션 검수 파이프라인 - 최종 통합 모듈
========================================

MSVD/Wikimedia 20개 샘플(정답 20 + 오답 60)로 검증을 마친 최종 버전이다.
공개 데이터 검증은 여기서 마무리하고, 다음 단계(회사 주행영상 파일럿)에
그대로 갖다 쓸 수 있도록 정리했다.

[공개 API]
    inspect_caption_full(video_path, caption, n_frames=5) -> FullInspectionResult

    입력: 영상 파일 경로(str/Path, 로컬 파일이면 무엇이든) + 검수하려는
          캡션 텍스트(str) 하나. 그 외에 이 프로젝트의 데이터셋(captions.json)에
          전혀 의존하지 않는다 - 완전히 범용적인 함수다.
    출력: 최종 판정(PASS/REVIEW/FAIL)과, 그 판정에 도달한 근거(문장/객체/
          행동 각 단계의 점수와 의심 구성요소)를 담은 딕셔너리.

[처리 흐름]
    1) 문장 전체 점수 (inspector.inspect_caption) - BLIP-ITM + 5프레임
       max-pooling + 재보정된 임계값(inspector.LOW_THRESHOLD/HIGH_THRESHOLD)
       으로 1차 PASS/REVIEW/FAIL 판정.
    2) 판정과 무관하게 객체 단위 검증(object_check.check_objects)을 돌린다.
       PASS였던 문장은 객체 검증 결과에 따라 REVIEW로 강등될 수 있다
       (PASS_BRANCH_THRESHOLD 기준).
    3) 판정과 무관하게 행동 단위 검증(action_check.check_actions)도 돌린다.
       PASS 분기에서는 강등에 반영되고, REVIEW/FAIL 분기에서는 원인 진단
       정보로만 덧붙는다.
    4) 스타일(사진/카툰/일러스트, style_detect.detect_video_style)을 먼저
       판별해서 객체·행동 검증 문구("a {style} of a {word}")에 반영한다.

[이 설계에 반영된 근거 요약]
  - BLIP-ITM(크로스 인코더)을 채점기로 채택: CLIP(듀얼 인코더)보다 행동
    hard negative를 구조적으로 더 잘 구분함이 확인됨.
  - 5프레임 균등 샘플링 + "검수 대상 캡션 점수 자체의 최댓값" 프레임 선택:
    오답 캡션 없이도 "오답을 활용한 이론적 최선(오라클)"의 90%까지 도달하는
    실전 가능한 최선의 전략으로 확인됨. (반대로 이미지 선명도 기반 필터링은
    오히려 손해였다 - 별해상도가 아니라 캡션 관련성이 핵심이기 때문.)
  - 문장 단위 임계값(0.9101/0.9560)은 정답 8/오답 24, 총 32개 표본으로,
    객체·행동 단위 임계값(0.0214)은 정답 51/가짜 73, 총 124개 표본으로
    각각 정확도를 최대화하는 값을 전수탐색해 정했다.

[알려진 한계 - 반드시 KNOWN_LIMITATIONS.md를 함께 읽을 것]
  이 파이프라인은 다음 3가지를 구조적으로 못 잡는다는 게 실증됐다:
    1. 순간적 사건 동사(도착하다/출발하다 등) - 정지 프레임 어디에도
       명확한 시각적 증거가 없을 수 있음
    2. 상태 전환이 있는 영상에서 잘못된 행동 단어 - 문장 점수가 이미
       매우 높을 때(PASS 분기) 특히 취약함이 8배 넓은 재검증에서 확인됨
    3. 여러 인물/사물이 동시에 다른 행동을 하는 장면에서의 주어 결합 오류
  그리고 회사 데이터 적용 전 반드시 확인해야 할 새로운 제약(캡션 언어 등)도
  KNOWN_LIMITATIONS.md에 정리해두었다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, TypedDict, Union

from inspector import inspect_caption, InspectionResult, Verdict
from object_check import check_objects
from action_check import check_actions
from style_detect import detect_video_style

ComponentTier = Literal["OK", "BORDERLINE", "SUSPECT"]

# --- 진단 표시용 3단계 임계값 (참고 설명 전용 - 강등 여부 결정에는 안 쓰임) ---
COMPONENT_SUSPECT_THRESHOLD = 0.05
COMPONENT_BORDERLINE_THRESHOLD = 0.15

# --- PASS 분기 강등 여부를 결정하는 임계값 (124개 표본으로 재보정) ---
PASS_BRANCH_THRESHOLD = 0.0214

# 문장 점수가 이 값 이상이면 행동 검증에 더 엄격한(=쉽게 SUSPECT 안 찍는)
# 기준을 쓴다는 안전장치. 단, PASS 판정 자체가 이미 inspector.HIGH_THRESHOLD
# (0.9560~) 이상이어야 나오므로, 0.95라는 컷오프는 PASS 분기에 도달하는
# 모든 문장에 사실상 항상 적용된다 - 즉 이 조건은 PASS 분기 "안에서"
# 사례를 구분하는 역할을 하지 못한다. 실험적으로 확인된 사실이며, 다음
# 재보정 때 이 메커니즘 자체를 재설계할 필요가 있다 (KNOWN_LIMITATIONS.md 참고).
SENTENCE_HIGH_CONFIDENCE_CUTOFF = 0.95
PASS_BRANCH_ACTION_LENIENT_THRESHOLD = 0.005

# 캡션 문장 전체가 한국어인 경우뿐 아니라, 회사 주행영상 캡션처럼 "문장은
# 영어인데 표지판/도로명 등 고유명사 일부만 한글로 섞여 들어간" 경우도
# 문제가 된다 - NLTK 품사 태거가 그 한글 토큰을 크래시 없이 이상한 품사로
# 잘못 분류하면, object_check이 "a photo of a 강남대로" 같은 의미 없는
# 질의를 만들어 엉뚱하게 낮은 점수를 받고, 멀쩡한 캡션이 SUSPECT로 잘못
# 찍혀 REVIEW로 강등될 수 있다. 그래서 "비율"이 아니라 "한글 글자가 하나
# 라도 섞여 있는지"를 본다 (한글 완성형/자모 유니코드 범위 기준).
_HANGUL_RANGES = [(0xAC00, 0xD7A3), (0x1100, 0x11FF), (0x3130, 0x318F)]


def _contains_hangul(text: str) -> bool:
    return any(any(lo <= ord(ch) <= hi for lo, hi in _HANGUL_RANGES) for ch in text)


def _warn_if_likely_non_english(caption: str) -> None:
    if not caption:
        return
    if _contains_hangul(caption):
        print(
            f"[full_inspector] 경고: 캡션에 한글 토큰이 포함돼 있습니다. "
            f"문장 전체가 아니라 표지판/도로명 등 일부 단어만 한글이어도, "
            f"객체/행동 추출(NLTK 영어 품사 태거) 단계에서 그 토큰이 잘못 분류돼 "
            f"의미 없는 질의를 만들 수 있습니다 (조용히 틀린 SUSPECT 판정으로 이어질 위험). "
            f"KNOWN_LIMITATIONS.md의 '캡션 언어 의존성' 항목을 확인하세요: {caption[:60]!r}"
        )


def classify_component(score: float) -> ComponentTier:
    """구성요소(객체/행동) 점수 하나를 참고용 3단계로 분류한다 (설명 표시 전용)."""
    if score < COMPONENT_SUSPECT_THRESHOLD:
        return "SUSPECT"
    if score < COMPONENT_BORDERLINE_THRESHOLD:
        return "BORDERLINE"
    return "OK"


class FullInspectionResult(TypedDict):
    sentence_score: float
    sentence_verdict: Verdict
    final_verdict: Verdict
    style: str
    objects_checked: bool
    actions_checked: bool
    object_results: Optional[list[dict]]
    action_results: Optional[list[dict]]
    suspect_components: list[str]   # 예: ["object:cats(SUSPECT, 0.018)"]


def inspect_caption_full(
    video_path: Union[str, Path],
    caption: str,
    n_frames: int = 5,
) -> FullInspectionResult:
    """영상 하나와 검수 대상 캡션 하나를 받아 최종 판정을 내린다.

    Parameters
    ----------
    video_path : str | Path
        로컬 영상 파일 경로. mp4/webm/ogv 등 imageio(pyav)가 디코딩할 수
        있는 포맷이면 무엇이든 된다 (frame_extract.py 참고).
    caption : str
        검수하려는 캡션 텍스트 1개 (영어). 이 프로젝트의 캡션 데이터셋과
        완전히 무관하게, 임의의 (영상, 캡션) 쌍에 바로 쓸 수 있다.
        **주의**: 객체/행동 추출(object_check.py, action_check.py)이 NLTK
        영어 품사 태거를 쓰기 때문에, 영어가 아닌 캡션(예: 한국어)을 넣으면
        크래시 없이 조용히 잘못된 결과를 낸다 - KNOWN_LIMITATIONS.md의
        "캡션 언어 의존성" 항목을 반드시 먼저 읽을 것.
    n_frames : int
        영상에서 균등 샘플링할 프레임 수 (기본 5). 짧은 클립(수 초~수십 초)
        기준으로 검증됐다 - 훨씬 긴 영상에 그대로 쓸 때의 적절성은 검증되지
        않았다 (KNOWN_LIMITATIONS.md 참고).

    Returns
    -------
    FullInspectionResult
        - final_verdict: "PASS" | "REVIEW" | "FAIL" (최종 판정)
        - sentence_score / sentence_verdict: 1차(문장 단위) 결과
        - suspect_components: 강등/진단에 관여한 구성요소 설명 목록
        - object_results / action_results: 구성요소별 원점수 전체 (설명 가능성용)
    """
    _warn_if_likely_non_english(caption)

    sentence: InspectionResult = inspect_caption(video_path, caption, n_frames=n_frames)

    style_info = detect_video_style(video_path, n_frames=n_frames)
    style = style_info["style"]

    object_results = None
    action_results = None
    final_verdict: Verdict = sentence["verdict"]

    if sentence["verdict"] == "PASS":
        # 문장은 통과했지만, 객체 + 행동 둘 다 한 번 더 확인해서 강등 여부를 결정한다.
        object_results = check_objects(video_path, caption, n_frames=n_frames, style=style)
        action_results = check_actions(video_path, caption, n_frames=n_frames, style=style)

        obj_suspects = [r for r in object_results if r["score"] < PASS_BRANCH_THRESHOLD]

        action_threshold = (
            PASS_BRANCH_ACTION_LENIENT_THRESHOLD
            if sentence["score"] >= SENTENCE_HIGH_CONFIDENCE_CUTOFF
            else PASS_BRANCH_THRESHOLD
        )
        act_suspects = [r for r in action_results if r["score"] < action_threshold]

        if obj_suspects or act_suspects:
            final_verdict = "REVIEW"

    else:
        # REVIEW/FAIL이면 객체+행동을 원인 진단용으로만 돌린다 (강등/승격 없음).
        object_results = check_objects(video_path, caption, n_frames=n_frames, style=style)
        action_results = check_actions(video_path, caption, n_frames=n_frames, style=style)

    suspect_components = []
    for r in (object_results or []):
        tier = classify_component(r["score"])
        if tier in ("SUSPECT", "BORDERLINE"):
            suspect_components.append(f"object:{r['word']}({tier}, {r['score']:.3f})")
    for r in (action_results or []):
        tier = classify_component(r["score"])
        if tier in ("SUSPECT", "BORDERLINE"):
            suspect_components.append(f"action:{r['word']}({tier}, {r['score']:.3f})")

    return {
        "sentence_score": sentence["score"],
        "sentence_verdict": sentence["verdict"],
        "final_verdict": final_verdict,
        "style": style,
        "objects_checked": object_results is not None,
        "actions_checked": action_results is not None,
        "object_results": object_results,
        "action_results": action_results,
        "suspect_components": suspect_components,
    }


if __name__ == "__main__":
    # 사용 예시 - 회사 데이터에도 이 형태(영상 경로 + 캡션 문자열)로 그대로 호출하면 된다.
    root = Path(__file__).resolve().parent.parent
    video = root / "data" / "videos" / "dog_park.ogv"
    caption = "Three cats are playing together outdoors in a park with bare trees and fallen leaves on the ground."

    result = inspect_caption_full(video, caption)
    print(f"캡션: {caption}")
    print(f"문장 점수/판정: {result['sentence_score']:.4f} / {result['sentence_verdict']}")
    print(f"최종 판정: {result['final_verdict']}")
    print(f"의심 구성요소: {result['suspect_components']}")
