"""
motion(정지/이동) 판별 - 옵티컬 플로우 실험
================================================

search_and_verify.py의 match_score(BLIP-ITM, 프레임 독립 채점)가 "정지"
관련 문장을 실제 전수 감사(30개)에서 잘못 판정한 사례(#11)를 발견한 뒤,
원인을 분석했다: 프레임을 여러 장 봐도 서로 비교하지 않고 각자 독립적으로
채점하기 때문에, "배경이 프레임 사이에 안 움직였다"(=자차가 정지해있다)는
증거 자체를 볼 방법이 구조적으로 없다.

1차 시도(단순 픽셀 차이, 배경 영역만 비교)는 실패했다 - 조명 변화/노이즈에
너무 취약해서 정지/이동 클립을 깨끗하게 구분 못 했다.

이번엔 정식 컴퓨터비전 기법인 옵티컬 플로우(연속 프레임 사이 각 픽셀이
어디로 움직였는지 추정)로 재시도한다. 핵심 아이디어:
  - 자차가 이동하면: 화면 전체(특히 배경/하늘/건물 영역)에서 뚜렷한
    방향성 있는 흐름이 나타난다 (원근 소실점에서 바깥으로 퍼지는 패턴)
  - 자차가 정지해있으면: 배경은 흐름이 거의 0이고, 지나가는 다른 차량이
    있는 좁은 영역에서만 국소적으로 흐름이 생긴다

그래서 "프레임 상단(하늘/건물, 배경일 가능성 높음) 영역의 평균 흐름 크기"를
신호로 써본다 - 도로 중앙(다른 차량이 지나갈 가능성 높은 영역)은 제외.

이 스크립트는 실험용이다 - 결과가 실제로 쓸 만한 신호로 확인되기 전까지는
search_and_verify.py의 프로덕션 경로에 연결하지 않는다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from frame_extract import extract_frames


def flow_magnitude_from_frames(frames: list, top_fraction: float = 0.35) -> float:
    """background_flow_magnitude()와 동일한 계산이지만, 이미 추출된 프레임
    리스트를 받는다 - search_and_verify()가 match_score용으로 이미 뽑아둔
    프레임을 재사용해 중복 디코딩을 피하려고 분리함."""
    grays = []
    for f in frames:
        arr = np.asarray(f.convert("L").resize((160, 160)))
        grays.append(arr)

    h = int(160 * top_fraction)
    mags = []
    for i in range(len(grays) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            grays[i], grays[i + 1], None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        bg_flow = flow[:h, :, :]  # 상단(배경) 영역만
        mag = np.sqrt(bg_flow[..., 0] ** 2 + bg_flow[..., 1] ** 2)
        mags.append(mag.mean())

    return float(np.mean(mags))


def background_flow_magnitude(video_path: str, n_frames: int = 5, top_fraction: float = 0.35) -> float:
    """video_path만 있고 프레임을 아직 안 뽑은 경우를 위한 편의 함수.
    프레임을 이미 갖고 있다면 flow_magnitude_from_frames()를 직접 쓰는 게
    더 빠르다(search_and_verify()가 그렇게 함)."""
    frames = extract_frames(video_path, n_frames=n_frames)
    return flow_magnitude_from_frames(frames, top_fraction=top_fraction)


if __name__ == "__main__":
    import pickle

    with open("/tmp/audit_results.pkl", "rb") as f:
        data = pickle.load(f)

    test_cases = {
        11: "실제 정지 (감사에서 확인)",
        12: "실제 이동(moving straight)",
        14: "실제 이동(moving straight)",
        1: "실제 이동(터널)",
        18: "실제 이동(야간)",
        4: "실제 이동(터널 접근)",
    }
    print(f"{'idx':>4s} {'배경 흐름 크기':>14s}  설명")
    for i, label in test_cases.items():
        q, clip_dir, score, is_rel, sent, ms, ivv, ct = data[i]
        video_path = clip_dir + "/1_clip/5.mp4"
        mag = background_flow_magnitude(video_path)
        print(f"#{i:<3d} {mag:14.4f}  {label}")
