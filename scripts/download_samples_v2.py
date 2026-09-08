"""
16단계: 대규모 재검증용 샘플 12개 추가 다운로드
=================================================
기존 8개(scripts/download_samples.py)와 겹치지 않는 새 샘플. 다양성 기준:
정적/동적/다중 피사체(주어결합 위험)/상태전환/동물·사물·사람 혼합.
"""

import requests
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "videos"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLES = {
    "reading_book": (
        "https://upload.wikimedia.org/wikipedia/commons/e/ec/Giuseppe_Mazzini_State_Middle_School_Book_Reading_%28992225%29.webm",
        "정적 - 학생이 앉아서 책을 읽는 영상",
    ),
    "soccer_kick": (
        "https://upload.wikimedia.org/wikipedia/commons/b/ba/Amateurfu%C3%9Fball_Torschuss_von_oben.webm",
        "동적 - 축구공을 차는 영상 (위에서 촬영)",
    ),
    "band_stage": (
        "https://upload.wikimedia.org/wikipedia/commons/0/04/Live_band_performs_on_stage..webm",
        "다중 인물(주어결합 위험) - 밴드가 무대에서 공연하는 영상",
    ),
    "cat_wakeup": (
        "https://upload.wikimedia.org/wikipedia/commons/f/ff/Salad_waking_up_from_a_nap.webm",
        "상태 전환 - 고양이가 낮잠에서 깨어나는 영상",
    ),
    "train_arriving": (
        "https://upload.wikimedia.org/wikipedia/commons/6/63/KRL_Commuterline_train_arriving_at_Sudirman_Station.webm",
        "동적/사물 - 기차가 역에 도착하는 영상",
    ),
    "children_playground": (
        "https://upload.wikimedia.org/wikipedia/commons/b/bd/Two_children_in_the_playground_at_school.webm",
        "동적/다중 인물 - 아이 둘이 놀이터에서 노는 영상",
    ),
    "pelican_flying": (
        "https://upload.wikimedia.org/wikipedia/commons/e/e1/Pelican_flying_video_01.webm",
        "동물/동적 - 사다새가 날아가는 영상",
    ),
    "jellyfish_aquarium": (
        "https://upload.wikimedia.org/wikipedia/commons/9/94/Sea_Nettle_%28%22Chrysaora_Fuscescens%22%29%2C_Monterey_Bay_Aquarium.ogv",
        "동물/느린 움직임 - 해파리가 수족관에서 유영하는 영상",
    ),
    "chef_cooking": (
        "https://upload.wikimedia.org/wikipedia/commons/1/12/Test_Kitchen-_Pork_Belly_Sandwich.webm",
        "사람(실사)/다중 인물 가능 - 요리사가 주방에서 요리하는 영상",
    ),
    "washing_dishes": (
        "https://upload.wikimedia.org/wikipedia/commons/9/9f/Child_washing_dishes.webm",
        "정적/가정 - 아이가 싱크대에서 설거지하는 영상",
    ),
    "running_form": (
        "https://upload.wikimedia.org/wikipedia/commons/b/b5/Running_form.ogv",
        "동적 - 사람이 달리는 영상 (러닝 자세 교육 영상)",
    ),
    "violin_solo": (
        "https://upload.wikimedia.org/wikipedia/commons/7/7e/Playing_a_Stroh_violin.ogv",
        "정적/단일 인물 - 한 사람이 바이올린을 연주하는 영상",
    ),
}

HEADERS = {"User-Agent": "caption-qa-pipeline-learning/1.0 (educational use; contact: susanahn11@gmail.com)"}


def download(url: str, dest: Path) -> None:
    with requests.get(url, headers=HEADERS, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
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
            print(f"       완료: {dest.name} ({dest.stat().st_size/1e6:.1f} MB)")
        except Exception as e:
            print(f"       실패: {e}")


if __name__ == "__main__":
    main()
