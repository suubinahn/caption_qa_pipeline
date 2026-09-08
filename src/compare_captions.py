"""
1단계 파이프라인 통합 실행 스크립트
====================================

이 스크립트가 하는 일 (전체 흐름):
  1) data/captions.json 에 정의된 각 샘플(영상 + 정답 캡션 + 오답 캡션)을 읽는다.
  2) 각 영상에서 대표 프레임(중간 프레임)을 뽑는다 (frame_extract.py).
  3) 그 프레임과 "정답 캡션"의 CLIP 유사도, "오답 캡션"의 CLIP 유사도를 각각 구한다 (clip_score.py).
  4) 두 점수의 차이(margin = 정답 점수 - 오답 점수)를 계산한다.
     margin이 양수이고 클수록 "CLIP이 진짜 정답 캡션을 더 잘 알아본다"는 뜻이고,
     margin이 0에 가깝거나 음수이면 "이 파이프라인(프레임 선택+CLIP)으로는
     정답과 오답을 잘 구분하지 못한다"는 뜻 -> 개선이 필요한 신호.
  5) 전체 결과를 표(pandas DataFrame)로 정리해서 콘솔에 출력하고 CSV로도 저장한다.

왜 "정답 vs 오답 margin"을 보는가?
  CLIP 유사도 점수 자체는 절대적인 기준이 없다 (0.25가 "좋다/나쁘다"를 딱
  잘라 말하기 어렵다). 하지만 "같은 이미지에 대해 맞는 문장이 틀린 문장보다
  점수가 높은가"는 비교적 명확하게 검증 가능한 질문이다. 이게 되어야만
  이후 단계에서 "VLM이 생성한 캡션 점수가 낮으면 실제로 캡션이 이상하다는
  뜻이다"라고 신뢰할 수 있다. 즉 이번 1단계는 "CLIP 채점기 자체가 믿을만한
  판별력을 갖고 있는지"를 먼저 검증하는 것이다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from frame_extract import save_frame
from clip_score import compute_similarity

ROOT = Path(__file__).resolve().parent.parent
CAPTIONS_JSON = ROOT / "data" / "captions.json"
VIDEO_DIR = ROOT / "data" / "videos"
FRAME_DIR = ROOT / "data" / "frames"
OUTPUT_CSV = ROOT / "outputs" / "stage1_results.csv"


def main():
    with open(CAPTIONS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = data["samples"]

    rows = []
    for s in samples:
        video_path = VIDEO_DIR / s["video_file"]
        frame_path = FRAME_DIR / f"{s['id']}.jpg"

        # 프레임이 이미 추출되어 있으면 재사용하고, 없으면 새로 추출한다.
        # (frame_extract.py를 이미 한 번 돌려서 data/frames에 저장해둔 상태라면 스킵됨)
        if not frame_path.exists():
            save_frame(video_path, frame_path)

        correct_score = compute_similarity(frame_path, s["correct_caption"])
        wrong_score = compute_similarity(frame_path, s["wrong_caption"])
        margin = correct_score - wrong_score

        rows.append(
            {
                "id": s["id"],
                "correct_caption": s["correct_caption"],
                "wrong_caption": s["wrong_caption"],
                "correct_score": round(correct_score, 4),
                "wrong_score": round(wrong_score, 4),
                "margin (correct-wrong)": round(margin, 4),
                "정답이 더 높음?": "O" if margin > 0 else "X",
                "note": s.get("note", ""),
            }
        )
        print(f"[done] {s['id']:20s} correct={correct_score:.4f}  wrong={wrong_score:.4f}  margin={margin:+.4f}")

    df = pd.DataFrame(rows)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 100)
    print("전체 결과 표 (정답 캡션/오답 캡션 CLIP 유사도 비교)")
    print("=" * 100)
    # 표 전체를 콘솔에서 보기 좋게 출력 (긴 캡션 텍스트는 잘라서 표시)
    display_df = df.copy()
    display_df["correct_caption"] = display_df["correct_caption"].str.slice(0, 45) + "..."
    display_df["wrong_caption"] = display_df["wrong_caption"].str.slice(0, 45) + "..."
    print(
        display_df[
            ["id", "correct_score", "wrong_score", "margin (correct-wrong)", "정답이 더 높음?"]
        ].to_string(index=False)
    )

    n_correct = (df["margin (correct-wrong)"] > 0).sum()
    print(f"\n총 {len(df)}개 샘플 중 {n_correct}개에서 정답 캡션 점수가 오답보다 높았음.")
    print(f"평균 margin: {df['margin (correct-wrong)'].mean():.4f}")
    print(f"\n결과 CSV 저장 위치: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
