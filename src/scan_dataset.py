"""
데이터셋 매니페스트 자동 스캔
================================

지금까지 outputs/search_full_dataset.json(클립 740개 목록 + info.txt 값)은
정적 스냅샷이었다 - 이걸 만든 스캔 스크립트가 코드베이스에 없어서, 새
클립이 생겨도 이 파일이 저절로 갱신되지 않았다(사람이 매번 손으로
갱신해야 했음). 이 스크립트가 그 빈 자리를 채운다.

루트 디렉토리(예: 회사 mount의 sample_videos/) 아래를 재귀적으로 훑어서,
caption.txt와 info.txt가 둘 다 있는 폴더를 "유효한 클립"으로 인식하고,
info.txt(motion/road_context/road_type/time_of_day/surface/weather 6줄
고정 순서)를 파싱해 build_cached_index()와 search_and_verify()가 바로
쓸 수 있는 JSON 매니페스트로 저장한다.

매번 전체를 처음부터 다시 스캔한다(부분 갱신/병합 없음) - 목록 자체가
가벼워서(클립 하나당 몇 바이트) 전체 재스캔 비용이 낮고, 병합 로직이
없어야 "삭제된 클립"도 자동으로 목록에서 빠진다는 장점이 있다. 클립
내용이 실제로 바뀌었는지(caption.txt 재생성 등)는 이 스크립트가 아니라
build_cached_index()의 지문(fingerprint)이 별도로 감지한다 - 이 스크립트는
"클립이 있는지/없는지, info.txt 값이 뭔지"만 책임진다.

사용법:
    python src/scan_dataset.py /path/to/driving_footage_root
    python src/scan_dataset.py /path/to/root --out outputs/search_full_dataset.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

INFO_FIELDS = ["motion", "road_context", "road_type", "time_of_day", "surface", "weather"]


def _parse_info_txt(path: Path) -> dict[str, str] | None:
    """info.txt(6줄 고정 순서)를 파싱한다. 줄 수가 안 맞으면 None을 반환하고
    호출자가 건너뛰게 한다 - 깨진/불완전한 클립 하나 때문에 전체 스캔이
    죽으면 안 되므로."""
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(lines) != len(INFO_FIELDS):
        return None
    return dict(zip(INFO_FIELDS, lines))


def scan_dataset(root: Path, caption_filename: str = "caption.txt") -> tuple[list[dict], list[str]]:
    """root 아래를 재귀 스캔해서 (클립 목록, 건너뛴 폴더 사유 목록)을 반환한다.
    caption.txt + info.txt가 둘 다 있는 폴더만 유효한 클립으로 취급한다."""
    clips = []
    skipped = []
    for caption_path in sorted(root.rglob(caption_filename)):
        clip_dir = caption_path.parent
        info_path = clip_dir / "info.txt"
        if not info_path.exists():
            skipped.append(f"{clip_dir}: info.txt 없음")
            continue
        info = _parse_info_txt(info_path)
        if info is None:
            skipped.append(f"{clip_dir}: info.txt 형식이 예상(6줄)과 다름")
            continue
        clips.append({"clip_dir": str(clip_dir), **info})
    return clips, skipped


def main():
    p = argparse.ArgumentParser(description="루트 디렉토리를 스캔해서 데이터셋 매니페스트 JSON을 만든다.")
    p.add_argument("root", type=Path, help="스캔할 루트 디렉토리 (예: .../sample_videos)")
    p.add_argument("--out", type=Path, default=Path("outputs/search_full_dataset.json"),
                   help="저장할 매니페스트 경로 (기본: outputs/search_full_dataset.json)")
    p.add_argument("--caption_filename", default="caption.txt", help="클립 판별 기준이 되는 캡션 파일명 (기본 caption.txt)")
    args = p.parse_args()

    if not args.root.exists():
        print(f"루트 디렉토리를 찾을 수 없습니다: {args.root}")
        sys.exit(1)

    print(f"스캔 중... ({args.root})")
    clips, skipped = scan_dataset(args.root, caption_filename=args.caption_filename)

    old_dirs: set[str] = set()
    if args.out.exists():
        with open(args.out, encoding="utf-8") as f:
            old_dirs = {c["clip_dir"] for c in json.load(f)}
    new_dirs = {c["clip_dir"] for c in clips}
    added, removed = new_dirs - old_dirs, old_dirs - new_dirs

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(clips, f, ensure_ascii=False, indent=2)

    print(f"\n총 {len(clips)}개 클립 발견 (건너뜀 {len(skipped)}개)")
    if old_dirs:
        print(f"이전 매니페스트({len(old_dirs)}개) 대비: 추가 {len(added)}개, 제거 {len(removed)}개")
    if skipped:
        print(f"\n건너뛴 폴더 (최대 10개 표시):")
        for s in skipped[:10]:
            print(f"  - {s}")
        if len(skipped) > 10:
            print(f"  ... 외 {len(skipped) - 10}개")
    print(f"\n저장 완료: {args.out}")


if __name__ == "__main__":
    main()
