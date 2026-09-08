"""
샘플 영상 다운로드 스크립트
=========================
왜 MSVD 원본 대신 이 방식을 쓰는가?
- MSVD(Microsoft Video Description) 데이터셋은 유튜브 영상 ID 목록을 배포하고,
  각 사용자가 youtube-dl/yt-dlp로 직접 영상을 받아야 한다. 하지만 데이터셋이
  2011년에 수집되어 상당수 원본 영상이 삭제/비공개 상태라 다운로드가 불안정하다.
- 그래서 여기서는 Wikimedia Commons(위키미디어 커먼즈)에 올라온 CC 라이선스
  영상 중 "고양이가 장난감을 갖고 논다", "사람이 기타를 친다"처럼 장면이 명확하고
  용량이 작은 것들을 골라 직접 다운로드한다. MSVD와 마찬가지로
  "짧은 영상 + 그 안의 행동을 설명하는 캡션" 구조를 그대로 재현할 수 있다.
- 실제 회사 데이터에 적용하기 전, 지금 단계에서 중요한 건 "진짜 MSVD인지"가
  아니라 "파이프라인(프레임 추출 -> CLIP 채점 -> 정답/오답 비교)이 제대로
  동작하는지" 검증하는 것이므로 이 대체재로 충분하다.
"""

import requests
from pathlib import Path

# 다운로드할 폴더 (프로젝트 루트 기준 data/videos)
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "videos"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Wikimedia Commons API로 미리 검색해서 확인한, 실제로 존재하고
# 용량이 작으며(수백 KB ~ 30MB) 내용이 명확한 영상들의 직접 다운로드 URL 목록.
# key: 저장할 파일명(=영상 id로도 사용), value: (원본 URL, 원본 파일 설명)
SAMPLES = {
    "cat_string_toy": (
        "https://upload.wikimedia.org/wikipedia/commons/1/13/Cat_Playing_with_String_Toy_1_2014-02-04.ogv",
        "고양이가 끈 장난감을 가지고 노는 영상",
    ),
    "cooking_thermometer": (
        "https://upload.wikimedia.org/wikipedia/commons/b/bb/Always_use_a_food_thermometer_to_check_temp_while_cooking.webm",
        "요리하며 음식 온도계로 온도를 재는 영상",
    ),
    "van_driving": (
        "https://upload.wikimedia.org/wikipedia/commons/8/8e/FSR_Tarpan_239_D_van_%28driving%29.webm",
        "밴(승합차)이 도로를 달리는 영상",
    ),
    "dog_park": (
        "https://upload.wikimedia.org/wikipedia/commons/5/53/Dog_park.theora.ogv",
        "개들이 공원에서 뛰노는 영상",
    ),
    "horse_herd": (
        "https://upload.wikimedia.org/wikipedia/commons/4/45/Dominance_hierarchy_in_a_herd_of_horses.webm",
        "말 여러 마리가 목초지에 모여있는 영상",
    ),
    "guitar_play": (
        "https://upload.wikimedia.org/wikipedia/commons/4/46/Secretary_Kerry_Plays_Musician%27s_Guitar_for_Chinese_Vice_Premier_Liu_in_Beijing_%2814432851718%29.webm",
        "실내에서 남성이 기타를 연주하는 영상",
    ),
    "people_dancing": (
        "https://upload.wikimedia.org/wikipedia/commons/b/b9/Maasai_people_singing_and_dancing.webm",
        "야외에서 사람들이 노래하고 춤추는 영상",
    ),
    "roller_skating": (
        "https://upload.wikimedia.org/wikipedia/commons/c/c5/Rolschaats_demonstraties-509131.ogv",
        "사람들이 야외에서 롤러스케이트를 타는 영상",
    ),
}

HEADERS = {
    # 위키미디어는 User-Agent 없는 요청을 차단할 수 있어 명시적으로 지정한다.
    "User-Agent": "caption-qa-pipeline-learning/1.0 (educational use; contact: susanahn11@gmail.com)"
}


def download(url: str, dest: Path) -> None:
    """URL에서 파일을 스트리밍 방식으로 받아 dest 경로에 저장한다.

    stream=True + iter_content를 쓰는 이유: 영상 파일은 수십 MB가 될 수 있어
    한 번에 메모리에 올리지 않고 조각(chunk) 단위로 받아 디스크에 바로 쓰기 위함.
    """
    with requests.get(url, headers=HEADERS, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):  # 64KB씩
                f.write(chunk)


def main():
    for name, (url, desc) in SAMPLES.items():
        ext = url.split("?")[0].rsplit(".", 1)[-1]
        dest = OUT_DIR / f"{name}.{ext}"
        if dest.exists() and dest.stat().st_size > 0:
            print(f"[skip] {dest.name} 이미 존재함")
            continue
        print(f"[down] {name} <- {desc}")
        try:
            download(url, dest)
            size_mb = dest.stat().st_size / 1e6
            print(f"       완료: {dest.name} ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"       실패: {e}")


if __name__ == "__main__":
    main()
