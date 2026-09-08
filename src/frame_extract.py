"""
영상에서 대표 프레임을 뽑는 모듈
================================

[배경 지식]
영상(video)은 결국 이미지(frame)를 초당 N장(FPS, Frames Per Second)씩
이어붙인 것이다. CLIP 같은 이미지-텍스트 모델은 "동영상"을 직접 입력받지
못하고 "정지 이미지 한 장"만 입력받는다. 그래서 영상 캡션을 검수하려면
영상에서 그 내용을 대표할 만한 프레임을 1장(혹은 여러 장) 뽑아서
이미지로 취급해야 한다.

가장 간단한 전략은 "중간 프레임(middle frame)"을 뽑는 것이다.
- 첫 프레임/마지막 프레임은 영상이 끝나기 직전에 화면이 어둡거나(fade)
  아직 장면이 시작 안 된 경우가 많아서 대표성이 떨어질 수 있다.
- 중간 지점은 보통 영상의 핵심 장면일 가능성이 높아 baseline으로 무난하다.

[중간 프레임 1장만으로는 부족했던 이유 - 실제로 겪은 문제]
그런데 실제로 파이프라인을 돌려보니(horse_herd, dog_park 샘플) 중간
프레임 딱 1장이 하필 핵심 동작이 안 보이는 애매한 순간을 찍어버리는
경우가 있었다. horse_herd 영상을 5개 지점(10/30/50/70/90%)에서 각각
뽑아 확인해보니, 다른 4개 지점은 "말이 달리는 중"이라는 게 명확했는데
정확히 중간(50%) 지점만 말들이 서로 붙어서 다리 동작이 가려진 애매한
프레임이었다. 즉 "어느 프레임을 대표로 뽑을지"에 따라 채점 결과가
크게 흔들릴 수 있다는 뜻이다.

그래서 이 모듈은 이제 두 가지 함수를 제공한다:
- extract_middle_frame(): 기존의 "1장만" 방식 (빠르지만 운에 좌우됨)
- extract_frames(): 영상을 시간축으로 균등하게 나눠 N장을 뽑는 방식.
  이렇게 뽑은 여러 프레임의 점수를 max pooling(최댓값)으로 합치면,
  "그 행동이 순간적으로라도 뚜렷하게 보이는 프레임이 하나라도 있다면"
  그 프레임의 높은 점수를 살릴 수 있다. (평균이 아니라 최댓값을 쓰는 이유:
  N장 중 몇 장이 우연히 애매해도, 확실하게 행동이 보이는 프레임 하나가
  있으면 그 신호를 놓치지 않기 위함)

[왜 imageio(+imageio-ffmpeg)를 쓰는가]
- OpenCV(cv2)의 VideoCapture도 흔히 쓰이지만, pip로 설치되는 opencv-python
  빌드는 컨테이너 포맷(mp4/webm/ogv 등)에 따라 디코딩이 안 되는 경우가 있다.
- imageio 라이브러리는 imageio-ffmpeg 플러그인을 통해 정적(static) ffmpeg
  바이너리를 자동으로 함께 설치하므로, 시스템에 ffmpeg가 없어도 거의 모든
  영상 포맷을 안정적으로 읽을 수 있다. (이번 환경엔 시스템 ffmpeg가 없었음)
"""

from __future__ import annotations  # Python 3.9에서도 `str | Path` 같은 최신 타입 힌트 문법을 쓰기 위함

from pathlib import Path
import imageio.v3 as iio
from PIL import Image
import numpy as np


def extract_middle_frame(video_path: str | Path) -> Image.Image:
    """영상 파일에서 정확히 중간 지점의 프레임 1장을 PIL 이미지로 반환한다.

    동작 원리:
    1) iio.improps(video_path, plugin="pyav")로 영상의 메타데이터(총 프레임 수 등)를 읽는다.
       ogv/webm처럼 인덱스가 불확실한 컨테이너도 있어, 총 프레임 수를 모르면
       전체를 순회하며 세는 방식(fallback)도 함께 둔다.
    2) 중간 인덱스 = 총 프레임 수 // 2 를 계산한다.
    3) iio.imread(video_path, index=중간 인덱스, plugin="pyav")로 해당 프레임만 디코딩한다.
       (전체 영상을 메모리에 다 올리지 않고 필요한 프레임까지만 디코딩하므로 효율적)

    반환값: PIL.Image.Image (RGB) - CLIP 전처리기가 바로 받을 수 있는 형태.
    """
    video_path = str(video_path)

    # plugin="pyav"는 imageio가 내부적으로 PyAV(ffmpeg 바인딩)를 사용해
    # 프레임 단위 랜덤 액세스(index로 특정 프레임 요청)를 지원하게 해준다.
    props = iio.improps(video_path, plugin="pyav")
    n_frames = props.shape[0] if props.shape and props.shape[0] > 0 else None

    if n_frames is None or n_frames <= 0:
        # 일부 포맷은 총 프레임 수를 미리 알려주지 않는다(가변 프레임레이트 등).
        # 이 경우 전체를 순회하며 프레임 수를 직접 센다. 샘플 영상들이 짧아서
        # (10~120초) 비용 부담이 크지 않다.
        frames = [f for f in iio.imiter(video_path, plugin="pyav")]
        n_frames = len(frames)
        middle_idx = n_frames // 2
        frame = frames[middle_idx]
    else:
        middle_idx = n_frames // 2
        frame = iio.imread(video_path, index=middle_idx, plugin="pyav")

    # frame은 numpy array (H, W, 3) uint8, RGB 순서. PIL 이미지로 감싸서 반환.
    return Image.fromarray(np.asarray(frame))


def extract_frames(video_path: str | Path, n_frames: int = 5) -> list[Image.Image]:
    """영상 전체 구간에서 시간축으로 균등하게 n_frames장을 뽑아 리스트로 반환한다.

    맨 처음(0%)과 맨 끝(100%)은 일부러 피한다 - 페이드인/아웃, 인트로
    타이틀 카드처럼 실제 장면이 아닌 구간일 가능성이 있어서다. 대신
    np.linspace(0.1, 0.9, n_frames)로 10%~90% 구간을 균등하게 나눈다.
    예) n_frames=5 -> 10%, 30%, 50%, 70%, 90% 지점 (horse_herd 진단에
    썼던 것과 동일한 지점).

    구현 방식: extract_middle_frame()처럼 iio.improps로 총 프레임 수를
    먼저 알아내려고 시도하고, 실패하면(ogv/webm 등 일부 컨테이너) 전체를
    한 번 순회해서 프레임 리스트를 만든 뒤 필요한 인덱스만 골라낸다.
    """
    video_path = str(video_path)

    props = iio.improps(video_path, plugin="pyav")
    n_total = props.shape[0] if props.shape and props.shape[0] > 0 else None

    fractions = np.linspace(0.1, 0.9, n_frames)

    if n_total is None or n_total <= 0:
        all_frames = [f for f in iio.imiter(video_path, plugin="pyav")]
        n_total = len(all_frames)
        indices = [int(n_total * f) for f in fractions]
        selected = [all_frames[i] for i in indices]
    else:
        indices = [int(n_total * f) for f in fractions]
        selected = [iio.imread(video_path, index=i, plugin="pyav") for i in indices]

    return [Image.fromarray(np.asarray(f)) for f in selected]


def save_frame(video_path: str | Path, out_path: str | Path) -> Image.Image:
    """중간 프레임을 뽑아서 이미지 파일로 저장하고, 그 이미지 객체도 반환한다.
    (눈으로 직접 확인하거나, CLIP 입력으로 재사용할 때 편하도록 둘 다 제공)
    """
    img = extract_middle_frame(video_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return img


if __name__ == "__main__":
    # 간단한 동작 테스트: data/videos 안의 모든 영상에서 중간 프레임을 뽑아
    # data/frames 에 jpg로 저장한다.
    root = Path(__file__).resolve().parent.parent
    video_dir = root / "data" / "videos"
    frame_dir = root / "data" / "frames"

    for video_path in sorted(video_dir.glob("*")):
        out_path = frame_dir / f"{video_path.stem}.jpg"
        print(f"[frame] {video_path.name} -> {out_path.name}")
        try:
            save_frame(video_path, out_path)
        except Exception as e:
            print(f"        실패: {e}")
