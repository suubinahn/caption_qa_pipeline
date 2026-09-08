"""
4단계 파이프라인: CLIP(raw) vs CLIP+CSLS vs BLIP-ITM 종합 비교
================================================================

지금까지의 흐름을 정리하면:
  1단계: CLIP 코사인 유사도로 정답/오답 캡션을 채점 -> 대체로 잘 구분함
  2단계: CSLS로 "허브 문제"를 보정 -> margin이 전반적으로 커짐
  3단계: 오답을 "행동 변경" vs "속성 변경"으로 나눠보니, CLIP은 특히
          "행동 변경" hard negative(예: 타고 있다 vs 서있다)를 구분하는 데
          더 취약했다 (margin이 속성 변경보다 훨씬 작았고, bicycle_ride는
          오답 점수가 정답보다 높기까지 했다)

4단계에서는 "정말 CLIP 구조 자체의 한계인지"를 확인하기 위해, 구조가
다른 모델인 BLIP의 ITM(크로스 인코더)으로 같은 샘플들을 다시 채점해서
margin을 비교한다. CLIP(코사인 유사도)과 BLIP-ITM(매칭 확률)은 값의
범위와 의미가 다르므로(-1~1 vs 0~1) margin의 "절대 크기"를 직접 비교하는
건 조심해야 하지만, "정답과 오답을 얼마나 명확하게 갈라놓는가"라는
상대적 판별력은 margin의 크기로 비교할 수 있다.

각 샘플의 margin = 정답 점수 - (오답 3개: 재활용 오답 + 행동 오답 + 속성
오답) 의 평균 점수. 2단계(compare_captions_csls.py)와 동일한 정의를 써서
세 방법의 margin을 공정하게 나란히 비교할 수 있게 했다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from frame_extract import save_frame
from clip_score import compute_similarity_matrix
from csls import csls
from blip_itm import compute_itm_score
from compare_captions_csls import build_text_pool, K_NEIGHBORS

ROOT = Path(__file__).resolve().parent.parent
CAPTIONS_JSON = ROOT / "data" / "captions.json"
VIDEO_DIR = ROOT / "data" / "videos"
FRAME_DIR = ROOT / "data" / "frames"
OUTPUT_CSV = ROOT / "outputs" / "stage4_clip_vs_blip.csv"

# 이번 비교에서 특히 눈여겨볼 샘플 (3단계에서 CLIP이 약했던 "행동 변경" 사례)
HIGHLIGHT_IDS = {"bicycle_ride", "cat_string_toy"}


def main():
    with open(CAPTIONS_JSON, "r", encoding="utf-8") as f:
        samples = json.load(f)["samples"]

    frame_paths = []
    for s in samples:
        frame_path = FRAME_DIR / f"{s['id']}.jpg"
        if not frame_path.exists():
            save_frame(VIDEO_DIR / s["video_file"], frame_path)
        frame_paths.append(frame_path)

    # --- CLIP raw / CSLS: 2단계와 동일하게 유사도 행렬을 한 번에 계산 ---
    text_pool, text_to_idx = build_text_pool(samples)
    print(f"[CLIP] 이미지 {len(frame_paths)}개 x 텍스트 {len(text_pool)}개 유사도 행렬 계산 중...")
    sim_raw = compute_similarity_matrix(frame_paths, text_pool)
    sim_csls = csls(sim_raw, k=K_NEIGHBORS)

    # --- BLIP-ITM: 크로스 인코더라 (이미지, 텍스트) 쌍마다 개별 forward pass 필요 ---
    print(f"[BLIP-ITM] 샘플당 4개 문장(정답+오답3) x {len(samples)}개 이미지 = 총 {len(samples)*4}회 채점 중...")

    rows = []
    for i, s in enumerate(samples):
        wrong_texts = [s["wrong_caption"], s["wrong_caption_action"], s["wrong_caption_attribute"]]

        # CLIP raw / CSLS margin (2단계와 동일한 방식: 오답 3개 평균과 비교)
        correct_idx = text_to_idx[s["correct_caption"]]
        wrong_idxs = [text_to_idx[t] for t in wrong_texts]
        margin_clip_raw = sim_raw[i, correct_idx] - sim_raw[i, wrong_idxs].mean()
        margin_clip_csls = sim_csls[i, correct_idx] - sim_csls[i, wrong_idxs].mean()

        # BLIP-ITM margin (직접 forward pass로 채점)
        frame = frame_paths[i]
        blip_correct = compute_itm_score(frame, s["correct_caption"])
        blip_wrong_scores = [compute_itm_score(frame, t) for t in wrong_texts]
        blip_wrong_mean = sum(blip_wrong_scores) / len(blip_wrong_scores)
        margin_blip = blip_correct - blip_wrong_mean

        rows.append({
            "id": s["id"],
            "margin_clip_raw": round(margin_clip_raw, 4),
            "margin_clip_csls": round(margin_clip_csls, 4),
            "margin_blip_itm": round(margin_blip, 4),
            "blip_correct_prob": round(blip_correct, 4),
            "blip_wrong_mean_prob": round(blip_wrong_mean, 4),
        })
        mark = " <-- 관심 샘플" if s["id"] in HIGHLIGHT_IDS else ""
        print(f"  [{s['id']:20s}] clip_raw={margin_clip_raw:+.4f}  clip_csls={margin_clip_csls:+.4f}  blip_itm={margin_blip:+.4f}{mark}")

    df = pd.DataFrame(rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 100)
    print("최종 비교: CLIP(raw) vs CLIP+CSLS vs BLIP-ITM margin")
    print("=" * 100)
    print(df[["id", "margin_clip_raw", "margin_clip_csls", "margin_blip_itm"]].to_string(index=False))

    print("\n" + "-" * 100)
    print("관심 샘플 (3단계에서 CLIP이 '행동 변경' hard negative에 취약했던 케이스)")
    print("-" * 100)
    for hid in HIGHLIGHT_IDS:
        row = df[df["id"] == hid].iloc[0]
        print(f"{hid}: CLIP raw margin={row['margin_clip_raw']:+.4f} -> CLIP+CSLS={row['margin_clip_csls']:+.4f} -> BLIP-ITM={row['margin_blip_itm']:+.4f}")

    n_blip_better_than_csls = (df["margin_blip_itm"] > df["margin_clip_csls"]).sum()
    print(f"\n{len(df)}개 샘플 중 {n_blip_better_than_csls}개에서 BLIP-ITM margin이 CLIP+CSLS margin보다 큼.")
    print(f"평균: clip_raw={df['margin_clip_raw'].mean():.4f}  clip_csls={df['margin_clip_csls'].mean():.4f}  blip_itm={df['margin_blip_itm'].mean():.4f}")
    print(f"\n결과 CSV 저장 위치: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
