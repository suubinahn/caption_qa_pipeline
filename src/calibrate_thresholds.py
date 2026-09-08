"""
PASS/REVIEW/FAIL 판정 임계값 보정(calibration) 스크립트
=========================================================

목표: 지금까지 만든 8개 샘플의 "정답 캡션 점수"(8개)와 "오답 캡션 점수"
(24개, 오답 3종 x 8개)를 inspect_caption()으로 전부 계산한 뒤, 두 그룹을
가장 잘 가르는 임계값을 데이터에서 직접 찾는다.

[왜 임계값이 하나가 아니라 두 개(low, high)인가]
실제로 계산해보면 두 그룹이 완벽하게 갈리지 않는다 - "고양이가 잔다"
(행동만 살짝 바꾼 오답) 같은 아주 미세한 오답은 정답과 거의 구분이 안 될
만큼 높은 점수를 받기도 한다. 이런 상황에서 임계값 하나로 "이 값 이상이면
무조건 통과"라고 정해버리면, 애매하게 겹치는 구간의 오답을 잘못 통과시킬
위험이 있다. 그래서:
  - high_threshold 이상: 오답이 도달한 사례가 거의 없는 구간 -> PASS
  - low_threshold 미만: 정답이 내려간 사례가 없는 구간 -> FAIL
  - 그 사이(low ~ high): 두 그룹이 실제로 섞여있는 구간이므로, 자동으로
    확정 판정하지 않고 사람이 봐야 하는 REVIEW로 넘긴다.
이렇게 "애매하면 사람에게 넘긴다"는 게 실전 검수 파이프라인에서 훨씬
안전한 설계다 (자동 PASS/FAIL만 있으면 애매한 경우를 억지로 한쪽으로
분류해야 해서 오류가 생긴다).

[주의] 표본이 정답 8개, 오답 24개뿐이라 통계적으로 매우 작은 데이터셋이다.
여기서 나온 임계값은 "이 8개 샘플에서 관찰된 경향"일 뿐, 회사 실데이터에
그대로 적용하기 전에 훨씬 더 큰 표본으로 재보정해야 한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from inspector import inspect_caption

ROOT = Path(__file__).resolve().parent.parent
CAPTIONS_JSON = ROOT / "data" / "captions.json"
VIDEO_DIR = ROOT / "data" / "videos"


def collect_scores():
    with open(CAPTIONS_JSON, "r", encoding="utf-8") as f:
        samples = json.load(f)["samples"]

    correct, wrong = [], []
    for s in samples:
        video = VIDEO_DIR / s["video_file"]
        correct.append(inspect_caption(video, s["correct_caption"])["score"])
        for key in ["wrong_caption", "wrong_caption_action", "wrong_caption_attribute"]:
            wrong.append(inspect_caption(video, s[key])["score"])

    return np.array(correct), np.array(wrong)


def best_single_threshold(correct: np.ndarray, wrong: np.ndarray) -> tuple[float, float]:
    """정답을 PASS(>=t), 오답을 FAIL(<t)로 나누는 이진 분류 정확도를
    최대화하는 임계값 t를 전수 탐색으로 찾는다 (참고용 - 3단계 판정에는
    이것과 다른, 좀 더 보수적인 low/high 두 임계값을 쓴다).
    """
    candidates = np.unique(np.concatenate([correct, wrong]))
    # 후보값들 사이의 중간점들을 임계값으로 시도 (관측된 점수 자체를 경계로
    # 쓰면 등호 처리가 애매해지므로, 항상 두 값 사이의 중간점을 쓴다)
    midpoints = (candidates[:-1] + candidates[1:]) / 2
    best_t, best_acc = 0.5, -1
    for t in midpoints:
        acc = ((correct >= t).sum() + (wrong < t).sum()) / (len(correct) + len(wrong))
        if acc > best_acc:
            best_acc = acc
            best_t = t
    return float(best_t), float(best_acc)


def main():
    print("점수 계산 중 (정답 8개 + 오답 24개)...")
    correct, wrong = collect_scores()

    print("\n" + "=" * 70)
    print("1) 그룹별 점수 분포")
    print("=" * 70)
    print(f"정답 그룹 (n={len(correct)}): min={correct.min():.4f}  max={correct.max():.4f}  mean={correct.mean():.4f}  median={np.median(correct):.4f}")
    print(f"오답 그룹 (n={len(wrong)}): min={wrong.min():.4f}  max={wrong.max():.4f}  mean={wrong.mean():.4f}  median={np.median(wrong):.4f}")

    n_overlap = int((wrong >= correct.min()).sum())
    print(f"\n오답 중 정답 최저점({correct.min():.4f}) 이상인 개수: {n_overlap}/{len(wrong)} (겹치는 hard negative)")

    print("\n" + "=" * 70)
    print("2) 임계값 후보")
    print("=" * 70)
    best_t, best_acc = best_single_threshold(correct, wrong)
    print(f"단일 임계값(정확도 최대화): t={best_t:.4f}  (정확도 {best_acc*100:.1f}%, {len(correct)+len(wrong)}개 중)")

    # low: "오답이 거의 도달하지 않는" 상한 -> 이 값 미만이면 FAIL
    #      = 겹치는 hard negative들(정답 최저점 이상)을 제외한, 나머지
    #        오답 중 최댓값을 기준으로 삼는다 (그 이하는 전부 명백한 오답)
    clean_wrong = wrong[wrong < correct.min()]
    low_threshold = float(clean_wrong.max()) if len(clean_wrong) else float(wrong.max())

    # high: "정답이 확실히 도달하는" 하한 -> 이 값 이상이면 PASS
    #       = 정답 그룹의 최솟값을 그대로 쓴다 (지금까지 관측된 정답은
    #         전부 이 값 이상이었다)
    high_threshold = float(correct.min())

    # 주의: 여기서 소수점을 반올림해서 inspector.py에 옮겨 적으면, 반올림된
    # 값이 그 값을 만들어낸 샘플 자신의 원본 float보다 커져버려 그 샘플이
    # 자기 경계에 걸려 잘못 분류되는 버그가 생길 수 있다(실제로 한 번 발생함).
    # inspector.py의 LOW_THRESHOLD/HIGH_THRESHOLD 상수는 항상 아래 repr() 값을
    # (반올림하지 말고) 그대로 복사해 넣을 것.
    print(f"\nlow_threshold  (이 미만 -> FAIL)  = {low_threshold!r}  (겹치지 않는 오답들의 최댓값)")
    print(f"high_threshold (이 이상 -> PASS)  = {high_threshold!r}  (정답 그룹의 최솟값)")
    print(f"-> {low_threshold:.4f} ~ {high_threshold:.4f} 구간은 REVIEW (사람이 재검토)")

    print(f"\n[참고] REVIEW 구간에 들어가는 오답 개수: {int(((wrong >= low_threshold) & (wrong < high_threshold)).sum())}개")
    print(f"[참고] REVIEW 구간에 들어가는 정답 개수: {int(((correct >= low_threshold) & (correct < high_threshold)).sum())}개")
    print(f"[참고] high_threshold 이상인데 실제로는 오답인 개수(자동 PASS 오류): {int((wrong >= high_threshold).sum())}개")
    print(f"[참고] low_threshold 미만인데 실제로는 정답인 개수(자동 FAIL 오류): {int((correct < low_threshold).sum())}개")

    return low_threshold, high_threshold


if __name__ == "__main__":
    main()
