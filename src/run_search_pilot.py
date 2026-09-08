"""
search_and_verify() 파일럿 CLI
================================

회사 주행영상 데이터셋에 검색어를 던져서 ①검색+②검증 결과를 터미널에서
바로 확인하는 스크립트. `run_pilot.py`(1단계 full_inspector.py용)와는
별개 - 이건 최종 통합 파이프라인(search_and_verify())용이다.

사용법:
    python src/run_search_pilot.py "It is raining."
    python src/run_search_pilot.py "The vehicle is stopped." --top_k 10 --detailed_claims
    python src/run_search_pilot.py "정체가 심하다" --lang ko
    python src/run_search_pilot.py "It is raining." --out outputs/pilot_rain.csv
    python src/run_search_pilot.py "It is raining." --dataset outputs/search_sample_100.json  # 빠른 테스트용(100클립)

새 데이터셋을 쓰려면 이 스크립트보다 먼저 scan_dataset.py로 매니페스트
JSON을 만들어야 한다(--dataset이 가리키는 파일이 그것):
    python src/scan_dataset.py <데이터 루트 경로> --out outputs/search_full_dataset.json

인덱스는 outputs/.index_cache/에 디스크 캐싱된다(search_module.py의
build_cached_index) - 클립 목록과 각 caption.txt의 수정시각/크기로 만든
지문(fingerprint)이 같으면 임베딩 재계산을 건너뛴다. 같은 데이터셋에
검색어만 바꿔가며 여러 번 돌리면 두 번째 실행부터 훨씬 빠르다. 새 데이터가
들어와 클립 목록/캡션 내용이 바뀌면 지문이 달라져 그 시점엔 다시 한 번
처음부터 빌드한다 - 이건 캐싱으로도 피할 수 없는 1회성 비용이고, 그
이후부터는 다시 캐싱 효과를 본다. 캐시는 최근에 실제로 쓰인 것 위주로
최대 --max_cache_entries개(기본 5)만 남기고 자동 정리된다 - 여러
데이터셋을 번갈아 테스트해도 디스크가 무한정 차지 않는다.

--feedback: 방금 본 결과에 관련있음/없음을 표시하면(Rocchio 스타일)
쿼리를 그 방향으로 조정해서 다음 결과를 다시 보여준다(①검색에만
적용, ②검증과 무관). calibrate_relevance_feedback.py(9개 검증 쿼리)로
확인한 바로는 "정지"/"야간"/"눈"처럼 원래 약했던 쿼리에서 크게
개선되지만(+30~90%p), "터널"/"다리"에서는 오히려 나빠지는 사례도 있었고
언제 성공/실패할지 예측하는 안전한 규칙을 못 찾았다. 그래서 결과를
조용히 바꿔치기하지 않고, **원본과 피드백 반영 결과를 나란히 보여줘서
사람이 직접 판단**하게 만들었다 - 이게 유일하게 검증된 안전장치다.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from search_module import build_cached_index, VALIDATED_ANCHOR_QUERIES
from search_and_verify import (
    search_and_verify,
    format_verification_label,
    format_claims_breakdown,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT / "outputs" / "search_full_dataset.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="search_and_verify() 파일럿 - 검색어를 넣으면 ①검색+②검증 결과를 출력한다.",
    )
    p.add_argument("query", help='검색어. 예: "It is raining." 또는 (--lang ko와 함께) "비가 온다"')
    p.add_argument("--top_k", type=int, default=5, help="상위 몇 개 결과를 볼지 (기본 5)")
    p.add_argument("--lang", choices=["en", "ko"], default="en", help="검색어 언어 (기본 en)")
    p.add_argument("--no_csls", action="store_true", help="CSLS 허브 보정을 끈다 (기본은 켜짐 - 프로덕션 기본값과 동일)")
    p.add_argument("--detailed_claims", action="store_true", help="문장을 필드별로 쪼갠 세부 검증 결과도 같이 보여준다")
    p.add_argument("--n_frames", type=int, default=5, help="②검증에 쓸 프레임 수 (기본 5, 프로덕션 기본값)")
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                    help=f"인덱싱할 클립 목록 JSON (기본: {DEFAULT_DATASET.relative_to(ROOT)}, 전체 740클립). "
                         f"빠른 테스트는 outputs/search_sample_100.json(100클립) 권장")
    p.add_argument("--caption_filename", default="caption.txt", help="클립 폴더 안 캡션 파일명 (기본 caption.txt)")
    p.add_argument("--video_filename", default="1_clip/5.mp4", help="클립 폴더 안 영상 파일 상대경로 (기본 1_clip/5.mp4)")
    p.add_argument("--out", type=Path, default=None, help="결과를 CSV로도 저장할 경로 (선택)")
    p.add_argument("--max_cache_entries", type=int, default=5,
                    help="인덱스 캐시를 최근 사용 순으로 몇 개까지 보관할지 (기본 5, 초과분은 자동 삭제)")
    p.add_argument("--feedback", action="store_true",
                    help="방금 본 결과에 관련있음/없음을 표시해서 쿼리를 재조정해보는 대화형 모드 (①검색 전용, 실험적 기능)")
    p.add_argument("--feedback_alpha", type=float, default=1.0, help="피드백 반영 강도 (기본 1.0)")
    return p.parse_args()


def run_feedback_mode(index, query: str, lang: str, use_csls: bool, seen_results, alpha: float) -> None:
    """방금 보여준 seen_results에 대해 관련있음/없음을 물어보고, Rocchio
    방식으로 쿼리를 재조정한 뒤 '원본 다음 결과'와 '피드백 반영 결과'를
    나란히 보여준다. 어느 쪽이 나은지는 사람이 판단한다 - 시스템이 자동
    판단할 안전한 기준을 못 찾았기 때문에(모듈 docstring 참고), 항상 둘 다
    보여주고 사람이 고르게 하는 게 유일하게 검증된 안전장치다."""
    print("\n" + "=" * 78)
    print("[관련성 피드백] 위 결과 각각에 대해 관련 있는지 표시해주세요.")
    print("  y = 관련있음, n = 관련없음, 엔터 = 건너뛰기")
    print("=" * 78)

    positive_sentences, negative_sentences = [], []
    for rank, r in enumerate(seen_results, 1):
        try:
            answer = input(f"  [{rank}위] \"{r.matched_sentence[:70]}\" -> (y/n/엔터): ").strip().lower()
        except EOFError:
            break
        if answer == "y":
            positive_sentences.append(r.matched_sentence)
        elif answer == "n":
            negative_sentences.append(r.matched_sentence)

    if not positive_sentences or not negative_sentences:
        print("\n관련있음/없음 라벨이 각각 최소 1개씩 필요합니다 - 피드백을 적용하지 않고 종료합니다.")
        return

    q_feat = index.embed_query(query, lang=lang)
    new_q = index.refine_embedding_with_feedback(q_feat, positive_sentences, negative_sentences, alpha=alpha)

    seen_dirs = {r.clip_dir for r in seen_results}
    original_full = index.search(query, top_k=None, use_csls=use_csls, lang=lang)
    original_next = [r for r in original_full if r.clip_dir not in seen_dirs][:len(seen_results)]

    refined_full = index.search_by_embedding(new_q, top_k=None, use_csls=use_csls)
    refined_next = [r for r in refined_full if r.clip_dir not in seen_dirs][:len(seen_results)]

    print(f"\n(라벨: 관련있음 {len(positive_sentences)}개, 관련없음 {len(negative_sentences)}개 -> 쿼리 재조정)")
    print("\n[원본 순위 - 다음 결과]" + " " * 20 + "[피드백 반영 - 다음 결과]")
    for i in range(len(seen_results)):
        left = right = "-"
        if i < len(original_next):
            cd = "/".join(original_next[i].clip_dir.rstrip("/").split("/")[-2:])
            left = f"{cd}: {original_next[i].matched_sentence[:45]}"
        if i < len(refined_next):
            cd = "/".join(refined_next[i].clip_dir.rstrip("/").split("/")[-2:])
            right = f"{cd}: {refined_next[i].matched_sentence[:45]}"
        print(f"{i+1}. {left}")
        print(f"   -> {right}")
    print("\n어느 쪽이 더 나은지는 직접 판단해서 참고해주세요 - 자동으로 어느 게 맞는지 알려주는 기준은 검증되지 않았습니다.")


def main():
    args = parse_args()

    if not args.dataset.exists():
        print(f"데이터셋 JSON을 찾을 수 없습니다: {args.dataset}")
        print("scan_dataset.py로 먼저 매니페스트를 만들어야 합니다. 예:")
        print(f"  python src/scan_dataset.py <데이터 루트 경로> --out {args.dataset}")
        sys.exit(1)

    import json
    with open(args.dataset, encoding="utf-8") as f:
        clips = json.load(f)
    clip_dirs = [c["clip_dir"] for c in clips]

    t0 = time.time()
    print(f"검색 인덱스 준비 중... ({len(clip_dirs)}개 클립, {args.dataset.name} 기준)")
    index = build_cached_index(
        clip_dirs,
        caption_filename=args.caption_filename,
        anchor_queries=VALIDATED_ANCHOR_QUERIES,
        max_cache_entries=args.max_cache_entries,
    )
    print(f"준비 완료 ({time.time()-t0:.1f}초).\n")

    results = search_and_verify(
        index,
        args.query,
        top_k=args.top_k,
        use_csls=not args.no_csls,
        lang=args.lang,
        video_filename=args.video_filename,
        n_frames=args.n_frames,
        detailed_claims=args.detailed_claims,
    )

    print("=" * 78)
    print(f'검색어: "{args.query}"' + (f"  (lang=ko)" if args.lang == "ko" else ""))
    print("=" * 78)

    if not results:
        print("결과 없음 - 이 데이터셋에 caption.txt가 있는 클립이 있는지 확인하세요.")
        return

    for rank, r in enumerate(results, 1):
        clip_id = "/".join(r.clip_dir.rstrip("/").split("/")[-2:])
        rel = "관련있음" if r.is_relevant is True else ("관련없음" if r.is_relevant is False else "판정불가(컷오프 없음)")
        print(f"\n[{rank}위] 클립: {clip_id}")
        print(f"  검색점수: {r.search_score:.4f}  |  검색 관련성: {rel}")
        print(f'  매칭 문장: "{r.matched_sentence}"')
        if r.is_video_verified is None:
            print(f"  영상 검증: 판단불가 (영상 없음: {r.video_path})")
        else:
            print(f"  영상 검증: {format_verification_label(r)}  (match_score={r.match_score:.4f})")
            if args.detailed_claims:
                print(f"  세부 분석: {format_claims_breakdown(r)}")
    print()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([
                "rank", "clip_dir", "search_score", "is_relevant", "matched_sentence",
                "match_score", "is_video_verified", "confidence_tier", "claims_breakdown", "video_path",
            ])
            for rank, r in enumerate(results, 1):
                writer.writerow([
                    rank, r.clip_dir, r.search_score, r.is_relevant, r.matched_sentence,
                    r.match_score, r.is_video_verified, r.confidence_tier,
                    format_claims_breakdown(r) if args.detailed_claims else "", r.video_path,
                ])
        print(f"결과 저장: {args.out}")

    if args.feedback:
        run_feedback_mode(
            index, args.query, lang=args.lang, use_csls=not args.no_csls,
            seen_results=results, alpha=args.feedback_alpha,
        )


if __name__ == "__main__":
    main()
