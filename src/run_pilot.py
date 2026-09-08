"""
회사 주행영상 파일럿 실행 스크립트
====================================

data/pilot_captions_template.csv 같은 형식(id, video_path, caption)의 CSV를
읽어서, 각 행마다 inspect_caption_full()을 호출하고 결과를 CSV로 저장한다.

사용법:
    python src/run_pilot.py [입력 CSV 경로] [출력 CSV 경로]

    (인자를 안 주면 기본값으로 data/pilot_captions_template.csv를 읽고
    outputs/pilot_results.csv에 저장한다 - 형식 확인용 예시)

CSV의 video_path는 이 스크립트를 실행하는 위치(보통 프로젝트 루트) 기준
상대경로거나 절대경로면 된다.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd

from full_inspector import inspect_caption_full

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "data" / "pilot_captions_template.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "pilot_results.csv"


def main():
    input_csv = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT
    output_csv = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUTPUT

    if not input_csv.exists():
        print(f"입력 CSV를 찾을 수 없습니다: {input_csv}")
        print("형식: id,video_path,caption  (헤더 포함)")
        sys.exit(1)

    with open(input_csv, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    print(f"{len(rows)}개 항목을 {input_csv}에서 읽음. 검수를 시작합니다...\n")

    results = []
    for i, row in enumerate(rows):
        vid = row["id"]
        video_path = ROOT / row["video_path"] if not Path(row["video_path"]).is_absolute() else Path(row["video_path"])
        caption = row["caption"]

        if not video_path.exists():
            print(f"[{i+1}/{len(rows)}] [{vid}] 영상 파일을 찾을 수 없음: {video_path} -> 건너뜀")
            results.append({"id": vid, "video_path": str(video_path), "caption": caption, "error": "video not found"})
            continue

        try:
            r = inspect_caption_full(video_path, caption)
            results.append({
                "id": vid,
                "video_path": str(video_path),
                "caption": caption,
                "sentence_score": round(r["sentence_score"], 4),
                "sentence_verdict": r["sentence_verdict"],
                "final_verdict": r["final_verdict"],
                "style": r["style"],
                "suspect_components": "; ".join(r["suspect_components"]) if r["suspect_components"] else "-",
                "error": "",
            })
            print(f"[{i+1}/{len(rows)}] [{vid}] 문장={r['sentence_score']:.4f}({r['sentence_verdict']}) "
                  f"-> 최종={r['final_verdict']}")
        except Exception as e:
            print(f"[{i+1}/{len(rows)}] [{vid}] 처리 중 오류: {e}")
            results.append({"id": vid, "video_path": str(video_path), "caption": caption, "error": str(e)})

    df = pd.DataFrame(results)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"\n결과 저장 위치: {output_csv}")
    if "final_verdict" in df.columns:
        print("\n판정 분포:")
        print(df["final_verdict"].value_counts())


if __name__ == "__main__":
    main()
