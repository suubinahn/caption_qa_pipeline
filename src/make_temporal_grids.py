"""
타임라인 그라운드트루스 라벨링용 - 클립별 10프레임 콘택트시트 생성
========================================================================

temporal_groundtruth_sample.csv의 각 (클립, 문장) 쌍마다, 그 클립의 vis/
프레임 10장을 프레임 번호 라벨과 함께 2x5 그리드 이미지 하나로 합친다.
사람(에이전트)이 그리드 하나만 보고 "이 문장이 실제로 몇 번 프레임에
해당하는지" 빠르게 판단할 수 있게 하기 위함 - 프레임을 하나씩 보는 것보다
훨씬 적은 호출로 검토 가능.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

OUT_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/temporal_grids")
SAMPLE_CSV = Path("outputs/temporal_groundtruth_sample.csv")

THUMB_W = 320
COLS, ROWS = 5, 2


def make_grid(clip_dir: str, out_path: Path) -> bool:
    vis_dir = Path(clip_dir) / "vis"
    frame_paths = sorted(vis_dir.glob("frame_*.jpg"))
    if len(frame_paths) < 10:
        return False

    thumbs = []
    for p in frame_paths[:10]:
        img = Image.open(p).convert("RGB")
        ratio = THUMB_W / img.width
        img = img.resize((THUMB_W, int(img.height * ratio)))
        thumbs.append(img)

    thumb_h = thumbs[0].height
    grid = Image.new("RGB", (THUMB_W * COLS, thumb_h * ROWS), "white")
    draw = ImageDraw.Draw(grid)
    for i, t in enumerate(thumbs):
        row, col = divmod(i, COLS)
        x, y = col * THUMB_W, row * thumb_h
        grid.paste(t, (x, y))
        label = f"#{i} ({int(i/9*80+10)}%)"
        draw.rectangle([x, y, x + 90, y + 26], fill="black")
        draw.text((x + 4, y + 4), label, fill="yellow")

    grid.save(out_path, quality=85)
    return True


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SAMPLE_CSV)

    manifest = []
    for i, row in df.iterrows():
        out_path = OUT_DIR / f"grid_{i:02d}.jpg"
        ok = make_grid(row["clip_dir"], out_path)
        manifest.append({
            "idx": i, "clip_dir": row["clip_dir"], "sentence": row["sentence"],
            "blip_frame": row["best_frame_idx"], "grid_path": str(out_path) if ok else None,
        })
        status = "OK" if ok else "실패(프레임없음)"
        print(f"[{i:2d}] {status}  {row['sentence'][:70]}")

    pd.DataFrame(manifest).to_csv("outputs/temporal_groundtruth_manifest.csv", index=False, encoding="utf-8-sig")
    print(f"\n총 {len(manifest)}개 처리, 매니페스트: outputs/temporal_groundtruth_manifest.csv")


if __name__ == "__main__":
    main()
