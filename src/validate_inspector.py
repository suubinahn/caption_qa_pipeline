"""
inspect_caption() 검증 스크립트
=================================

inspector.py의 inspect_caption()이 6단계(compare_practical_frame_selection.py)의
"방법2: 정답점수만으로 max 선택" 로직을 그대로 재현하는지 확인한다.

검증 방법: 6단계 로직을 이 스크립트 안에서 inspect_caption()과 완전히
독립적으로(같은 함수를 호출하지 않고, extract_frames + compute_itm_score를
직접 다시 호출해서) 재계산한 뒤, inspect_caption()의 출력과 소수점 4자리까지
정확히 일치하는지 비교한다. 일치하면 "리팩터링이 로직을 바꾸지 않았다"는
회귀 검증(regression test)이 되는 것이고, 어긋나면 통합 과정에서 버그가
생겼다는 뜻이다.

추가로, 6단계 결과 CSV(outputs/stage6_practical_frame_selection.csv)에 이미
저장된 "정답점수 최고 idx"와도 프레임 인덱스가 일치하는지 함께 확인한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from frame_extract import extract_frames
from blip_itm import compute_itm_score
from inspector import inspect_caption

ROOT = Path(__file__).resolve().parent.parent
CAPTIONS_JSON = ROOT / "data" / "captions.json"
VIDEO_DIR = ROOT / "data" / "videos"
STAGE6_CSV = ROOT / "outputs" / "stage6_practical_frame_selection.csv"

N_FRAMES = 5


def independent_recompute(video_path: Path, caption: str) -> tuple[float, int, list[float]]:
    """6단계 '방법2' 로직을 inspect_caption()과 완전히 독립적으로 재구현.
    (inspector.py를 import하지 않고 처음부터 다시 계산 - 진짜 독립 검증)
    """
    frames = extract_frames(video_path, n_frames=N_FRAMES)
    scores = [compute_itm_score(f, caption) for f in frames]
    best_idx = int(np.argmax(scores))
    return scores[best_idx], best_idx, scores


def main():
    with open(CAPTIONS_JSON, "r", encoding="utf-8") as f:
        samples = json.load(f)["samples"]

    stage6_df = pd.read_csv(STAGE6_CSV)

    print(f"{'id':20s} {'inspect_caption()':>18s} {'독립 재계산':>14s} {'차이':>10s} {'idx 일치?':>10s} {'6단계 idx 일치?':>14s}")
    print("-" * 95)

    all_pass = True
    for s in samples:
        sid = s["id"]
        video_path = VIDEO_DIR / s["video_file"]
        caption = s["correct_caption"]

        # 1) inspect_caption() 결과
        result = inspect_caption(video_path, caption, n_frames=N_FRAMES)

        # 2) 완전히 독립적으로 재계산한 결과
        indep_score, indep_idx, _ = independent_recompute(video_path, caption)

        # 3) 6단계 CSV에 저장된 "정답점수 최고 idx"와 비교
        stage6_row = stage6_df[stage6_df["id"] == sid].iloc[0]
        stage6_idx = int(stage6_row["정답점수 최고 idx"])

        diff = abs(result["score"] - indep_score)
        idx_match = result["best_frame_index"] == indep_idx
        stage6_idx_match = result["best_frame_index"] == stage6_idx

        status_ok = diff < 1e-6 and idx_match and stage6_idx_match
        all_pass = all_pass and status_ok

        print(f"{sid:20s} {result['score']:>18.6f} {indep_score:>14.6f} {diff:>10.2e} "
              f"{'O' if idx_match else 'X':>10s} {'O' if stage6_idx_match else 'X (참고)':>14s}")

    print("-" * 95)
    if all_pass:
        print("\n✅ 8개 샘플 전부: inspect_caption() == 독립 재계산 (완전 일치), 6단계 프레임 인덱스와도 일치.")
    else:
        print("\n⚠️  일부 샘플에서 불일치 발견 - 위 표에서 X 표시된 항목을 확인할 것.")


if __name__ == "__main__":
    main()
