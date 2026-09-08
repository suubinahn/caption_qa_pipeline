"""
통합 캡션 검수 함수
====================

1~6단계 실험에서 "실전에 배포할 수 있다"고 검증된 조합만 골라 하나의
함수로 통합한다:

  - **BLIP-ITM** 채점 (4단계: CLIP 듀얼 인코더보다 크로스 인코더가
    "타는 중 vs 서있는 중" 같은 행동 hard negative를 훨씬 잘 구분함을 확인)
  - **다중 프레임 5장 균등 샘플링** (5단계: 중간 프레임 1장은 "운"에 좌우됨을
    horse_herd 사례로 실증 - 하필 애매한 순간을 뽑으면 완전히 틀린 결론이 남)
  - **"검수 대상 캡션 점수만으로 최댓값 프레임 선택"** (6단계 방법2: 오답
    캡션 없이도 "오답을 알고 고른 이론적 최선(오라클)"의 90.5%까지 도달하는,
    실전에서 쓸 수 있는 최선의 프레임 선택 전략으로 확인됨)

의도적으로 포함하지 않은 것들:
  - CLIP(+CSLS): 구조적으로 행동 변화에 약하다는 게 3~4단계에서 확인돼
    최종 채점기로 채택하지 않았다 (CLIP은 대량 스크리닝용 1차 필터로는
    여전히 유효할 수 있지만, 이 함수의 목적은 "최종 정합성 판단"이다).
  - "오답을 알고 프레임을 고르는 방식"(5단계 오라클): 이론적 최선이지만
    실전 검수 시점엔 오답 캡션이 없으므로 애초에 계산이 불가능하다.
  - 선명도 필터링 / 평균 pooling: 6단계에서 각각 오라클 대비 55.8%,
    61.0%로 "정답점수 max" 방식(90.5%)보다 뚜렷이 못했다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, TypedDict, Union

import numpy as np

from frame_extract import extract_frames
from blip_itm import compute_itm_score

N_FRAMES_DEFAULT = 5

# --- PASS/REVIEW/FAIL 판정 임계값 ---
# src/calibrate_thresholds.py로 8개 샘플의 정답 캡션 점수 8개 + 오답 캡션
# 점수 24개(오답 3종 x 8개)를 이 파일과 동일한 inspect_caption()으로 전부
# 채점한 뒤 데이터에서 직접 구했다:
#   - HIGH_THRESHOLD = 관측된 정답 점수의 최솟값(bicycle_ride, 0.955969...)
#     (이 값 이상이면 지금까지 정답이 아니었던 적이 없다 -> PASS)
#   - LOW_THRESHOLD  = "정답 최저점보다 낮은" 오답들 중 최댓값(dog_park의
#     행동변경 오답, 0.910144...) (이 값 미만이면 지금까지 정답이었던
#     적이 없다 -> FAIL)
#   - 그 사이는 두 그룹이 실제로 섞여있는 구간이라 REVIEW로 둔다 (예:
#     "고양이가 잔다"처럼 행동만 살짝 바꾼 hard negative 오답 4개가
#     오히려 이 구간보다도 높게 나온 경우가 있었다 - 이런 애매한 경우까지
#     자동으로 PASS/FAIL 확정하지 않고 사람 검토로 넘기기 위한 안전장치다)
#
# 소수점 4자리로 반올림한 값(0.9560, 0.9101)을 그대로 상수로 박아넣지
# 않는 이유: bicycle_ride의 실제 점수(0.955969...)가 반올림된 0.9560보다
# 미세하게 낮아서, 정작 그 값을 만들어낸 샘플 자신이 경계에 걸려 REVIEW로
# 잘못 분류되는 반올림 버그가 실제로 있었다. 그래서 원본 float 값을 그대로 쓴다.
LOW_THRESHOLD = 0.9101440906524658    # dog_park, wrong_caption_action 점수
HIGH_THRESHOLD = 0.9559694528579712   # bicycle_ride, correct_caption 점수

Verdict = Literal["PASS", "REVIEW", "FAIL"]


def classify_score(score: float, low: float = LOW_THRESHOLD, high: float = HIGH_THRESHOLD) -> Verdict:
    """점수 하나를 받아 PASS/REVIEW/FAIL 중 하나로 판정한다.

      score >= high         -> "PASS"   (확실히 정답과 일치)
      low <= score < high   -> "REVIEW" (애매함, 사람 재검토 필요)
      score < low           -> "FAIL"   (확실히 정답과 불일치)
    """
    if score >= high:
        return "PASS"
    if score >= low:
        return "REVIEW"
    return "FAIL"


class InspectionResult(TypedDict):
    score: float                # 최종 정합성 점수 (0~1, BLIP-ITM 매칭 확률)
    verdict: Verdict            # "PASS" / "REVIEW" / "FAIL" 판정
    best_frame_index: int       # n_frames장 중 최댓값을 받은 프레임의 인덱스
    frame_scores: list[float]   # 프레임별 점수 전체 (설명 가능성/디버깅용)
    n_frames: int


def inspect_caption(
    video_path: Union[str, Path],
    caption: str,
    n_frames: int = N_FRAMES_DEFAULT,
    low_threshold: float = LOW_THRESHOLD,
    high_threshold: float = HIGH_THRESHOLD,
) -> InspectionResult:
    """영상 하나와 검수하려는 캡션 하나를 받아, 그 캡션이 영상 내용과
    얼마나 일치하는지 0~1 사이의 정합성 점수로 반환한다.

    처리 과정 (1~6단계에서 검증된 순서 그대로):
      1) extract_frames()로 영상의 10~90% 구간에서 시간축으로 균등하게
         n_frames장(기본 5장)을 뽑는다. 맨 처음/끝은 페이드인/아웃일 수
         있어 일부러 피한다 (frame_extract.py 참고).
      2) 각 프레임에 대해 compute_itm_score()로 BLIP-ITM 매칭 확률을
         계산한다. (크로스 인코더라 프레임마다 forward pass가 필요하다 -
         CLIP처럼 임베딩을 미리 계산해 재사용할 수 없음)
      3) n_frames개 점수 중 최댓값을 최종 점수로 채택한다. "이 캡션이
         조금이라도 뚜렷하게 확인되는 순간이 한 프레임에라도 있다면 그
         증거를 놓치지 않는다"는 원칙 - 6단계에서 오답 캡션 없이도 가장
         좋은 성능을 낸 전략이다.
      4) classify_score()로 최종 점수를 PASS/REVIEW/FAIL로 판정한다.

    반환값에 best_frame_index와 frame_scores를 함께 담는 이유: 실전
    검수 파이프라인에서는 "왜 이 점수가 나왔는지" 설명할 수 있어야 한다.
    점수가 낮게 나온 캡션을 사람이 재검토할 때, 최댓값이 나온 프레임을
    같이 보여주면 훨씬 빠르게 판단할 수 있다.

    low_threshold/high_threshold를 매개변수로 노출해둔 이유: 기본값은
    지금 8개 샘플로 보정한 값이지만, 실제 회사 데이터로 재보정한 임계값을
    나중에 그대로 주입해 쓸 수 있도록 하기 위함이다.
    """
    if n_frames < 1:
        raise ValueError("n_frames는 1 이상이어야 합니다.")

    frames = extract_frames(video_path, n_frames=n_frames)
    frame_scores = [compute_itm_score(frame, caption) for frame in frames]

    best_idx = int(np.argmax(frame_scores))
    score = frame_scores[best_idx]
    return {
        "score": score,
        "verdict": classify_score(score, low=low_threshold, high=high_threshold),
        "best_frame_index": best_idx,
        "frame_scores": frame_scores,
        "n_frames": n_frames,
    }


if __name__ == "__main__":
    # 간단한 사용 예시
    root = Path(__file__).resolve().parent.parent
    video = root / "data" / "videos" / "bicycle_ride.ogv"
    caption = "A man in a red shirt is riding a bicycle down a city street."

    result = inspect_caption(video, caption)

    print(f"영상: {video.name}")
    print(f"검수 캡션: {caption}")
    print(f"정합성 점수: {result['score']:.4f}")
    print(f"판정: {result['verdict']}")
    print(f"최댓값 프레임: {result['best_frame_index']} / {result['n_frames']}")
    print(f"프레임별 점수: {[round(s, 4) for s in result['frame_scores']]}")
