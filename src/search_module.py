"""
캡션 기반 영상 검색 모듈 (최종 정리본)
========================================

outputs/search_largescale_*.csv 실험에서 검증된, 가장 안정적인 조합을
재사용 가능한 형태로 정리한 모듈. 검색어 하나를 여러 번 바꿔가며 던져도
캡션 임베딩을 매번 다시 계산하지 않도록 인덱스를 한 번 빌드해두고 재사용한다.

핵심 로직:
  1) 캡션을 문장 단위로 쪼갠다 (split_sentences) - CLIP 77토큰 잘림 회피
  2) 검색어와 모든 문장을 CLIP 텍스트 인코더로 배치 임베딩
  3) 클립별로 가장 잘 맞는 문장 하나의 점수를 그 클립 점수로 사용 (max-pooling)
  4) 점수 내림차순 정렬

기본 캡션 소스는 caption.txt (회사 프로덕션 캡션). sb_caption/2_caption.txt도
쓸 수 있지만, 100클립 x 9검색어 대규모 실험에서 어떤 검색어도 caption.txt
결과를 능가하지 못했다 (outputs/search_largescale_caption_summary.csv 참고).
sb_caption_2가 info.txt 값을 덜 그대로 베끼는 것("공정한 테스트")은 맞지만,
그게 더 나은 검색 결과로 이어지진 않았다.

CSLS(허브 보정) 지원: 텍스트-텍스트 유사도에서도 "여러 검색어에 걸쳐 두루
높은 점수를 받는 문장(허브)" 문제가 실제로 확인됨 - anchor_queries를 주면
r_S(문장별 허브 정도)를 미리 계산해두고, search(use_csls=True)로 보정된
점수를 쓸 수 있다. leave-one-out 검증 결과 9개 검색어 전부 정밀도가
개선되거나 동일했고(악화 없음), 특히 "정지"(9.1%->18.2%), "야간"
(33.3%->58.3%)에서 크게 개선됨. 단, 지금까지 검증한 anchor_queries는
이 모듈이 지원하는 9개 도메인 쿼리뿐이라 - 실서비스에서 전혀 다른
자유형 쿼리에도 이 정도 효과가 유지될지는 미검증.

알려진 한계 (근거는 이번 조사에서 직접 확인함, KNOWN_LIMITATIONS.md 참고):
  - 상태/동작 동사(정지 vs 이동, 좌회전 vs 직진) 같은 미세한 의미 차이
    구분이 약하다. CLIP 텍스트 인코더가 문장 하나의 핵심 의미보다 문장
    전체 표면 어휘 유사도에 더 좌우되는 경향을 "정지"와 "좌회전" 검색어
    둘 다에서 직접 확인함 - 예를 들어 "좌회전" 검색에서 실제로는 "직진"을
    묘사하는 문장이 1위(0.9135)를 차지하고, 진짜 좌회전 문장은 2위
    (0.9104)로 밀린 사례가 있음. 관련/무관 문장 간 점수 차이가 0.003~0.05
    수준으로 사실상 없다시피 함 (문장 템플릿이 거의 동일하기 때문으로 추정).
  - road_context/surface 필드는 캡션에 거의 그대로(85~97%) 주입되어 있어,
    이 필드 관련 검색어는 "영상 이해"보다 "문구 일치 검사"에 가깝다.
  - road_type(차선 수 기반)처럼 라벨 경계 자체가 시각적으로 애매한 카테고리는
    낮은 precision이 검색 실패가 아니라 라벨 모호성 때문일 수 있다는 가설을
    세웠고, 실제 프레임 하나(고속도로처럼 보이는 장면이 'three-lane road'로
    라벨링된 사례)로는 뒷받침됨. 단, 'multi-lane highway'+'three-lane road'를
    합쳐 "정답"으로 재정의해 재검증한 결과, precision 수치 자체는 크게
    오르지만(19.5%->76.9%, 53.7%->78.2%) 병합 카테고리의 랜덤 베이스라인도
    78%로 같이 뛰어서, 베이스라인 대비 lift로 보면 오히려 "거의 랜덤과
    동일"(0.99~1.00배)로 떨어짐 - 원래 라벨 기준 캡션 검색의 1.31배 우위가
    사라짐. 즉 라벨 경계 모호성이 유일한 원인은 아니고, 병합해도 모델이
    "고속도로스러움"을 랜덤보다 잘 구분하지 못하는 근본적인 약점이 남아있다.
  - info.txt/캡션 자체도 완전한 정답이 아니다 - precision 수치는 근사치로만
    참고할 것.
  - 지금까지 검증된 쿼리는 info.txt 카테고리에 맞춘 영어 템플릿 문장뿐이다.
    자유형 한국어 문장이나 여러 조건이 섞인 복합 검색어는 아직 미검증.
  - [해결됨] 한국어 검색어: CLIP 자체는 한국어를 이해 못 하지만(직접 입력
    시 "터널 통과" 81.2%->6.2%로 붕괴), search(query, lang="ko")로 번역
    전처리(translate_query.py, Helsinki-NLP/opus-mt-ko-en)를 거치면 9개
    검색어 중 7개가 영어 원본 수준으로 회복됨. "정지" 관련 쿼리는 번역과
    무관하게 원래 약한 쿼리라 그대로 남아있음 (위 상태동사 항목 참고).
    번역 품질이 완벽하진 않고("정지해 있는 장면"->"Car's still on." 같은
    오역 존재), 번역된 문장이 VALIDATED_QUERY_THRESHOLDS의 등록된 영어
    문자열과 정확히 일치하지 않으면 컷오프는 적용 안 되고 순위만 나온다.
  - [부분 해결] 컷오프: 처음엔 100클립 표본으로 calibration/holdout(50/50)
    F1-최대화 임계값을 시도해 "터널"만 검증됐었음. 이후 회사 mount 전체
    740클립으로 재검증(calibrate_search_thresholds_full.py, 370/370 분리) -
    "비"가 새로 안정적으로 검증됨(precision/recall 68.3%/68.3%), "터널"도
    유지(precision 100%, recall 37.5%, 값은 370클립 기준으로 갱신). 다만
    100클립 표본은 희귀 카테고리(우회전/좌회전/터널/다리/야간/눈)를 이미
    전수 포함하고 있었기 때문에 - 이 6개는 740클립으로 확장해도 양성
    샘플 수가 그대로라 여전히 불안정함. 특히 "야간"은 100클립 기준으론
    "약하게 검증됨"으로 봤었는데 370클립 재검증에서 precision이
    54.5%->5.7%로 무너짐 - 작은 negative pool에서 나온 착시였던 것으로
    확인됨 (표본을 늘리지 않았으면 못 잡았을 함정). "다차선고속도로"는
    개선됐지만(precision 43.3%, recall 48.1%) 아직 확신하기엔 애매해서
    보류. VALIDATED_QUERY_THRESHOLDS에 없는 검색어는 컷오프 없이 순위만
    제공할 것 - 검증 안 된 임계값을 쓰는 것보다 "모른다"고 하는 게 낫다.
  - [부분 해결] 부정문(negation) 처리: 자유형 쿼리 실전 테스트에서 CLIP이
    "not", "no" 같은 부정어를 사실상 무시한다는 걸 발견함("교통 정체" 검색에
    "정체 아님"이 1위, 코사인 0.9248로 자기자신과 비교한 1.0000에 근접).
    search(..., penalize_negation=True, 기본값)로 완화 - 검색어 핵심 단어
    앞쪽에 부정어가 있는 문장을 대표 문장 후보에서 배제한다. 9개 검증
    검색어로 회귀 테스트: 8개 변화 없음, "터널"만 81.2%->75.0%로 하락했는데
    이건 오탐이 아니라 실제 캡션 결함(터널 얘기가 "안 보인다"뿐이었음)을
    새로 잡아낸 것으로 확인됨. 명시적 부정어만 잡고 암묵적 반대 표현/부분
    부정은 여전히 못 잡는 저비용 완화책이다 (KNOWN_LIMITATIONS.md 10번).
"""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import re

from clip_score import load_model
from validate_sentence_split_recall import split_sentences
from translate_query import translate_ko_to_en

# [해결 시도] 부정문 감지 - 자유형 쿼리 실전 테스트에서 발견한 문제
# (KNOWN_LIMITATIONS.md 10번): CLIP 텍스트 인코더가 "not"/"no" 같은 부정어를
# 사실상 무시해서, "교통 정체" 검색에 "정체 아님"이라는 정반대 문장이 1위로
# 뜨는 사례를 실측함(코사인 유사도 0.9248, 자기 자신과 비교한 1.0000에 근접).
# 완벽한 해결책은 아니고(부분 부정, 암묵적 부정은 못 잡음) 명시적 부정어
# 근처에 검색어 핵심 단어가 있는 가장 뚜렷한 경우만 걸러내는 저비용 완화책.
_NEGATION_WORDS = {"not", "no", "never", "without", "none", "nothing",
                    "n't", "hardly", "barely", "cannot", "can't"}
_QUERY_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "being", "been",
    "it", "this", "that", "there", "on", "in", "at", "of", "to", "and",
    "with", "as", "vehicle", "ego-vehicle", "car", "scene", "road",
}


def _extract_query_keywords(query: str) -> list[str]:
    """검색어에서 핵심 내용어만 추출한다 (불용어/일반적인 주행 도메인
    단어 제외) - 부정문 감지에서 "이 개념이 부정되고 있는지" 확인할 대상."""
    words = re.findall(r"[a-z']+", query.lower())
    return [w for w in words if w not in _QUERY_STOPWORDS and w not in _NEGATION_WORDS and len(w) > 2]


def _stems_match(a: str, b: str, min_overlap: int = 5) -> bool:
    """"congestion"과 "congested"처럼 어형만 다른 단어를 같은 개념으로
    본다 - 앞 min_overlap글자가 같으면 어간이 같다고 근사한다(간이 스테밍).
    실제로 "not congested"가 "congestion" 키워드와 정확히 안 겹쳐서 부정문
    감지를 빠져나가는 사례를 발견한 뒤 추가함."""
    if len(a) < min_overlap or len(b) < min_overlap:
        return a == b
    return a[:min_overlap] == b[:min_overlap]


def _sentence_negates_keywords(sentence: str, keywords: list[str], window: int = 4) -> bool:
    """문장 안에서 검색어 핵심 단어(어형 변화 포함) 앞쪽(window 단어 이내)에
    부정어가 있으면 True. 완벽한 구문 분석이 아니라 근접도 기반 휴리스틱이다.

    부정어보다 "앞"이 아니라 "뒤"에서만 찾는 이유: 영어 부정어는 보통 뒤에
    오는 단어를 부정한다("not congested"). 앞뒤 다 보면 "a clear night with
    no stars visible" 같은 문장에서 "no"가 실제로는 "stars"를 부정하는
    건데 "night"까지 부정으로 오탐하는 문제가 실측으로 확인됨 - 방향을
    제한해서 고침.
    """
    if not keywords:
        return False
    tokens = re.findall(r"[a-z']+", sentence.lower())
    for i, tok in enumerate(tokens):
        if any(_stems_match(tok, kw) for kw in keywords):
            start = max(0, i - window)
            if any(t in _NEGATION_WORDS for t in tokens[start:i]):
                return True
    return False


# calibrate_search_thresholds_full.py로 회사 mount 전체 740클립을 짝/홀
# calibration/holdout(370/370) 분리해서 재검증한 결과. 점수는 use_csls=True
# 기준. 100클립 표본에서는 6개 검색어(우회전/좌회전/터널/다리/야간/눈)가
# 이미 전수 샘플링돼 있어서 - 전체 데이터로 확장해도 양성 샘플 수 자체는
# 안 늘어남 - 재검증해도 그대로 불안정. 반대로 정지/비/다차선고속도로는
# 양성 샘플이 크게 늘어(11->126, 35->240, 41->216) 재검증 의미가 있었음.
# 결과: "비"가 새로 안정적으로 검증됨(precision/recall 둘 다 68.3%로 균형
# 잡힘). "터널"은 여전히 안정적(precision 100%, recall 37.5%) - 값은
# 370클립 기준으로 갱신. "다차선고속도로"는 개선됐지만(precision 43.3%,
# recall 48.1%) 아직 확신하기엔 애매해서 보류. "야간"은 오히려 정밀도가
# 54.5%->5.7%로 무너졌다 - 100클립 표본 당시의 "약하게 검증됨" 판단 자체가
# 작고 만만한 negative pool에서 나온 착시였던 것으로 드러남 (표본을 늘리지
# 않았다면 못 잡았을 함정). 나머지 6개는 여전히 표본 부족으로 미검증.
VALIDATED_QUERY_THRESHOLDS = {
    "The car drives through a tunnel.": 0.0744,  # holdout(370) precision 100%, recall 37.5%
    "It is raining.": -0.0799,  # holdout(370) precision 68.3%, recall 68.3%
}


@dataclass
class SearchResult:
    clip_dir: str
    score: float
    matched_sentence: str  # 이 클립에서 검색어와 가장 잘 맞았던 문장 (근거 확인용)
    is_relevant: bool | None = None  # 검증된 컷오프가 있는 검색어만 채워짐, 없으면 None


class CaptionSearchIndex:
    """캡션들을 한 번 임베딩해두고 여러 검색어에 재사용하는 인덱스.

    anchor_queries를 주면 CSLS 허브 보정에 쓸 r_S(문장별 허브 정도)를
    미리 계산해둔다. r_S는 "이 문장이 anchor_queries 전반에 걸쳐 얼마나
    두루 높은 점수를 받는 경향이 있는가"를 뜻하며, 인덱스 시점에 한 번만
    계산하고 이후 모든 search() 호출에서 재사용한다 (실시간 검색어 자체는
    r_S 계산에 관여하지 않는다 - 그래야 라이브 검색에서도 쓸 수 있다).
    """

    def __init__(
        self,
        clip_dirs: list[str],
        caption_filename: str = "caption.txt",
        anchor_queries: list[str] | None = None,
        csls_k: int | None = None,
    ):
        self.clip_dirs = clip_dirs
        self.caption_filename = caption_filename
        self._sentences: list[str] = []
        self._owner: list[str] = []
        self._embeds: torch.Tensor | None = None
        self._build()

        self._r_S: np.ndarray | None = None
        if anchor_queries:
            self._build_r_S(anchor_queries, csls_k)

    def _build(self) -> None:
        for clip_dir in self.clip_dirs:
            path = Path(clip_dir) / self.caption_filename
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8").strip()
            if not text:
                continue
            for sentence in split_sentences(text):
                self._sentences.append(sentence)
                self._owner.append(clip_dir)

        if not self._sentences:
            raise ValueError("인덱싱할 캡션 문장이 하나도 없습니다 (경로/파일명을 확인하세요).")

        self._embeds = self._embed_texts(self._sentences)  # (n_sentences, dim), L2 정규화됨

    def _embed_texts(self, texts: list[str], batch_size: int = 256) -> torch.Tensor:
        """텍스트를 청크로 나눠 임베딩한다. 한 번에 다 넣으면(특히 문장 수가
        많을 때) GPU 메모리가 터진다 - 740클립 규모에서 실제로 CUDA OOM을
        겪은 뒤 이 배치 처리를 추가함."""
        model, processor, device = load_model()
        all_feats = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                feats = model.get_text_features(**inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            all_feats.append(feats)
        return torch.cat(all_feats, dim=0)

    def _build_r_S(self, anchor_queries: list[str], k: int | None) -> None:
        """r_S(y) = 문장 y가 anchor_queries 상위 k개와 갖는 평균 유사도 ("허브 정도")."""
        anchor_embeds = self._embed_texts(anchor_queries)  # (n_anchor, dim)
        sim = (anchor_embeds @ self._embeds.T).cpu().numpy()  # (n_anchor, n_sentences)
        # k or len(...)이 아니라 k is None으로 체크 - k=0을 명시적으로 넘겼을 때
        # falsy라서 무시되고 기본값으로 대체되는 버그가 코드리뷰로 발견됨.
        k = min(k if k is not None else len(anchor_queries), len(anchor_queries))
        top_k = np.partition(sim, -k, axis=0)[-k:, :]  # (k, n_sentences)
        self._r_S = top_k.mean(axis=0)  # (n_sentences,)

    def _save_cache(self, path: Path) -> None:
        """빌드된 인덱스 상태를 디스크에 저장한다 - build_cached_index()가
        다음 실행에서 재사용할 수 있도록. 임베딩은 CPU로 옮겨서 저장한다
        (GPU 없는 환경에서 로드하거나, 로드 시점에 다른 디바이스를 쓸 수
        있으므로 - 로드할 때 다시 적절한 디바이스로 옮긴다)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "clip_dirs": self.clip_dirs,
            "caption_filename": self.caption_filename,
            "sentences": self._sentences,
            "owner": self._owner,
            "embeds": self._embeds.detach().cpu(),
            "r_S": self._r_S,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def _load_cache(cls, path: Path) -> "CaptionSearchIndex":
        """_save_cache()로 저장된 상태를 그대로 복원한다 - __init__()의
        _build()/_embed_texts()/_build_r_S() 재계산(임베딩 재계산 포함)을
        전부 건너뛴다. build_cached_index()가 로드 후 임베딩을 현재
        디바이스로 옮긴다."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls.__new__(cls)
        obj.clip_dirs = state["clip_dirs"]
        obj.caption_filename = state["caption_filename"]
        obj._sentences = state["sentences"]
        obj._owner = state["owner"]
        obj._embeds = state["embeds"]
        obj._r_S = state["r_S"]
        return obj

    def _rank_by_embedding(
        self,
        q_feat: torch.Tensor,
        use_csls: bool = False,
        csls_k: int = 10,
        negated_mask: list[bool] | None = None,
    ) -> list[SearchResult]:
        """이미 계산된 쿼리 임베딩 하나로 클립을 점수순 정렬한다 - search()와
        search_by_embedding() 둘 다 이 핵심 로직을 공유한다(코드 중복 방지).
        is_relevant(컷오프)은 여기서 안 채운다 - 그건 원본 검색어 문자열이
        있어야 VALIDATED_QUERY_THRESHOLDS를 찾을 수 있는데, 조정된 임베딩은
        어떤 문자열에도 대응하지 않기 때문이다(호출자가 필요하면 채운다)."""
        sims = (self._embeds @ q_feat).cpu().numpy()  # (n_sentences,)

        if use_csls:
            if self._r_S is None:
                raise ValueError("use_csls=True이려면 인덱스 생성 시 anchor_queries를 넘겨야 합니다.")
            k = min(csls_k, len(sims))
            r_T = np.partition(sims, -k)[-k:].mean()  # 이 쿼리 자체의 "후한 정도"
            sims = 2 * sims - r_T - self._r_S

        best_per_clip: dict[str, tuple[float, str]] = {}
        for i, (sim, clip_dir, sentence) in enumerate(zip(sims, self._owner, self._sentences)):
            sim = float(sim)
            if negated_mask is not None and negated_mask[i]:
                sim = -1e6  # 스케일(원본/CSLS)에 무관하게 사실상 배제되도록 큰 음수로 고정
            if clip_dir not in best_per_clip or sim > best_per_clip[clip_dir][0]:
                best_per_clip[clip_dir] = (sim, sentence)

        results = [
            SearchResult(clip_dir=clip_dir, score=score, matched_sentence=sentence)
            for clip_dir, (score, sentence) in best_per_clip.items()
        ]
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def search(self, query: str, top_k: int | None = None, use_csls: bool = False, csls_k: int = 10, lang: str = "en", penalize_negation: bool = True) -> list[SearchResult]:
        """검색어 하나를 받아 인덱싱된 클립들을 점수순으로 정렬해 반환한다.

        use_csls=True면 CSLS 허브 보정을 적용한다 - 인덱스 생성 시
        anchor_queries를 넘겼어야 한다 (r_S가 미리 계산되어 있어야 함).
        검증 결과(leave-one-out): 9개 도메인 검색어 전부 정밀도 개선 또는
        동일, 악화 없음 - 기본값은 False지만 실사용 시 켜는 것을 권장.

        lang="ko"면 검색어를 영어로 번역한 뒤 검색한다 (translate_query.py,
        Helsinki-NLP/opus-mt-ko-en). 9개 도메인 검색어로 실측: 7개는 영어
        원본과 거의 같거나 더 나은 precision 회복, "정지" 관련 쿼리만
        번역과 무관하게 원래 약함. 번역된 문장은 VALIDATED_QUERY_THRESHOLDS의
        등록된 영어 문자열과 토씨까지 같지 않으면 컷오프(is_relevant)가
        적용되지 않는다 - 순위 결과 자체는 정상.

        penalize_negation=True(기본값)면, 검색어의 핵심 단어 근처에 부정어가
        있는 문장(예: "not congested")을 그 클립의 대표 문장 후보에서
        사실상 배제한다(KNOWN_LIMITATIONS.md 10번 - CLIP이 부정문을 못 걷러내는
        문제의 완화책). 완벽하지 않다 - 명시적 부정어만 잡고, 암묵적 반대
        표현("정체 없이 원활하다"의 "원활")은 못 잡는다.
        """
        if lang == "ko":
            query = translate_ko_to_en([query])[0]

        q_feat = self._embed_texts([query])[0]  # (dim,)

        negated_mask = None
        if penalize_negation:
            keywords = _extract_query_keywords(query)
            negated_mask = [_sentence_negates_keywords(s, keywords) for s in self._sentences]

        results = self._rank_by_embedding(q_feat, use_csls=use_csls, csls_k=csls_k, negated_mask=negated_mask)

        threshold = VALIDATED_QUERY_THRESHOLDS.get(query) if use_csls else None
        if threshold is not None:
            for r in results:
                r.is_relevant = r.score >= threshold

        return results[:top_k] if top_k is not None else results

    def embed_query(self, query: str, lang: str = "en") -> torch.Tensor:
        """검색어를 임베딩 벡터로 변환만 한다 (검색은 안 함) - 관련성
        피드백처럼 임베딩 벡터 자체를 직접 조작해야 하는 고급 용도용.
        보통의 검색에는 search()를 쓰면 된다."""
        if lang == "ko":
            query = translate_ko_to_en([query])[0]
        return self._embed_texts([query])[0]

    def refine_embedding_with_feedback(
        self,
        q_feat: torch.Tensor,
        positive_sentences: list[str],
        negative_sentences: list[str],
        alpha: float = 1.0,
    ) -> torch.Tensor:
        """관련성 피드백(Rocchio 스타일) - "관련있다"고 표시된 문장들의 평균
        임베딩에서 "관련없다"고 표시된 문장들의 평균 임베딩을 뺀 방향으로
        쿼리 벡터를 alpha만큼 밀어서 재조정한다.

        calibrate_relevance_feedback.py(9개 검증 쿼리, info.txt 자동 라벨링)로
        검증: "정지"/"야간"/"눈"처럼 원래 약했던 쿼리에서 뚜렷한 개선을
        확인했지만(+30~90%p), "터널"/"다리"에서는 오히려 나빠지는 사례도
        있었다 - 언제 성공/실패할지 예측하는 안전한 규칙을 못 찾았다(표본
        긍정/부정 비율 등 시도해본 가설은 반증 사례로 기각됨). 그래서 이
        함수는 결과를 조용히 바꿔치기하는 용도가 아니라, 호출자가 원본
        결과와 나란히 보여주고 사람이 어느 쪽이 나은지 판단하게 하는 용도로
        설계됐다(run_search_pilot.py --feedback 참고)."""
        if not positive_sentences or not negative_sentences:
            raise ValueError("Rocchio 방향을 계산하려면 관련있음/없음 예시가 최소 1개씩 필요합니다.")
        pos_embeds = self._embed_texts(positive_sentences)
        neg_embeds = self._embed_texts(negative_sentences)
        direction = pos_embeds.mean(dim=0) - neg_embeds.mean(dim=0)
        return q_feat + alpha * direction

    def search_by_embedding(self, q_feat: torch.Tensor, top_k: int | None = None, use_csls: bool = False, csls_k: int = 10) -> list[SearchResult]:
        """이미 계산됐거나(embed_query) 피드백으로 조정된
        (refine_embedding_with_feedback) 쿼리 임베딩으로 바로 검색한다.
        원본 검색어 문자열이 없으므로 부정문 필터링(penalize_negation)과
        컷오프(is_relevant) 조회는 이 경로에 적용되지 않는다 - 둘 다
        "정확히 이 문자열이 검색어다"라는 전제가 있어야 하는 기능이다."""
        results = self._rank_by_embedding(q_feat, use_csls=use_csls, csls_k=csls_k)
        return results[:top_k] if top_k is not None else results


def _fingerprint_clips(clip_dirs: list[str], caption_filename: str, anchor_queries: list[str] | None, csls_k: int | None) -> str:
    """이 클립 목록 + 캡션 파일들의 현재 상태를 해시 하나로 요약한다 -
    build_cached_index()가 "이 조합을 전에 임베딩해본 적 있는지" 판단하는
    캐시 키다. 각 caption.txt의 수정시각/크기까지 포함하므로, 클립이 그대로여도
    캡션 내용이 바뀌면(재생성 등) 자동으로 다른 해시가 나와 캐시가 무효화된다
    - 새 데이터가 들어오면(클립 목록이 달라지거나 caption.txt가 새로 생기면)
    자연스럽게 캐시 미스가 나서 처음 한 번은 다시 빌드하고, 그 이후부터는
    다시 캐싱 효과를 본다."""
    h = hashlib.sha256()
    for clip_dir in sorted(clip_dirs):
        path = Path(clip_dir) / caption_filename
        try:
            stat = path.stat()
            h.update(f"{clip_dir}|{stat.st_mtime_ns}|{stat.st_size}".encode())
        except FileNotFoundError:
            h.update(f"{clip_dir}|MISSING".encode())
    h.update(caption_filename.encode())
    h.update(repr(sorted(anchor_queries or [])).encode())
    h.update(repr(csls_k).encode())
    return h.hexdigest()[:16]


def _prune_index_cache(cache_dir: Path, keep: int, verbose: bool = True) -> None:
    """cache_dir에 쌓인 캐시(*.pkl)가 keep개를 넘으면, 가장 오래전에 "쓰인"
    것부터 지운다 - LRU(least-recently-used) 방식. build_cached_index()가
    캐시 적중 시 파일 mtime을 현재 시각으로 갱신해두므로(아래 참고), 최근에
    실제로 조회된 캐시는 만들어진 지 오래됐어도 살아남고, 한동안 안 쓰인
    캐시(예: 이제 안 쓰는 옛날 데이터셋)부터 먼저 지워진다. 서로 다른
    데이터셋을 번갈아 파일럿에 쓰면 캐시 파일이 무한정 쌓이는 걸 막는다."""
    if not cache_dir.exists():
        return
    cache_files = sorted(cache_dir.glob("*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    for stale in cache_files[keep:]:
        stale.unlink()
        if verbose:
            print(f"[인덱스 캐시 정리] 오래 안 쓰인 캐시 삭제: {stale.name}")


def build_cached_index(
    clip_dirs: list[str],
    caption_filename: str = "caption.txt",
    anchor_queries: list[str] | None = None,
    csls_k: int | None = None,
    cache_dir: str | Path = "outputs/.index_cache",
    max_cache_entries: int = 5,
    verbose: bool = True,
) -> CaptionSearchIndex:
    """CaptionSearchIndex(...)와 결과가 동일하지만, 디스크 캐시를 먼저
    확인한다. 같은 클립 목록 + 같은 caption.txt 내용으로 이미 인덱싱해본
    적이 있으면 임베딩 재계산(CLIP 모델 호출, 가장 오래 걸리는 부분)을
    건너뛰고 캐시에서 바로 불러온다. 매번 새로 빌드하는 CaptionSearchIndex(...)
    직접 호출과 달리, "같은 데이터셋에 검색어만 바꿔가며 여러 번 파일럿을
    돌리는" 상황에서 반복되는 임베딩 비용을 없애준다.

    새 데이터가 들어와 클립 목록이나 caption.txt 내용이 바뀌면
    (_fingerprint_clips 참고) 캐시 키가 자동으로 달라져 캐시 미스가 나고,
    그 시점엔 다시 한 번 처음부터 빌드한다 - 이건 캐싱으로도 피할 수 없는
    1회성 비용이다. 그 이후부터는(같은 데이터로 재실행하는 한) 다시
    캐싱 효과를 본다.

    max_cache_entries(기본 5)를 넘는 캐시 파일은 자동으로 정리된다
    (_prune_index_cache 참고) - 여러 데이터셋을 번갈아 테스트해도 캐시
    디렉터리가 무한정 커지지 않는다."""
    cache_dir = Path(cache_dir)
    fingerprint = _fingerprint_clips(clip_dirs, caption_filename, anchor_queries, csls_k)
    cache_path = cache_dir / f"{fingerprint}.pkl"

    if cache_path.exists():
        if verbose:
            print(f"[인덱스 캐시 적중] {cache_path.name} - 임베딩 재계산 생략")
        index = CaptionSearchIndex._load_cache(cache_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        index._embeds = index._embeds.to(device)
        cache_path.touch()  # "최근에 쓰였다"고 기록 - LRU 정리 때 보존되도록
        return index

    if verbose:
        print(f"[인덱스 캐시 없음] 새로 빌드합니다 (완료 후 {fingerprint}.pkl로 저장)")
    index = CaptionSearchIndex(clip_dirs, caption_filename=caption_filename, anchor_queries=anchor_queries, csls_k=csls_k)
    index._save_cache(cache_path)
    _prune_index_cache(cache_dir, keep=max_cache_entries, verbose=verbose)
    return index


# 지금까지 검증한 9개 도메인 검색어 - CSLS anchor_queries 기본값으로 사용.
# 실서비스에서는 더 크고 다양한 anchor 세트로 교체하는 게 이상적이다 (미검증).
VALIDATED_ANCHOR_QUERIES = [
    "The vehicle is turning right.",
    "The vehicle is turning left.",
    "The vehicle is stopped.",
    "The car drives through a tunnel.",
    "The car crosses a bridge.",
    "The scene is at night.",
    "It is raining.",
    "It is snowing.",
    "The vehicle drives on a multi-lane highway.",
]


if __name__ == "__main__":
    import json

    with open("outputs/search_sample_100.json", encoding="utf-8") as f:
        clips = json.load(f)
    clip_dirs = [c["clip_dir"] for c in clips]

    print(f"인덱스 빌드 중... ({len(clip_dirs)}개 클립, caption.txt 기준, CSLS anchor 9개)")
    index = CaptionSearchIndex(clip_dirs, caption_filename="caption.txt", anchor_queries=VALIDATED_ANCHOR_QUERIES)
    print("빌드 완료.\n")

    demo_queries = [
        "The vehicle is stopped.",
        "The vehicle is turning left.",
    ]
    for q in demo_queries:
        print(f'[검색어] "{q}" (원본 코사인)')
        for r in index.search(q, top_k=5):
            clip_id = "/".join(r.clip_dir.rstrip("/").split("/")[-2:])
            print(f"  {r.score:.4f}  {clip_id}  <- \"{r.matched_sentence[:70]}\"")
        print(f'[검색어] "{q}" (CSLS 보정)')
        for r in index.search(q, top_k=5, use_csls=True):
            clip_id = "/".join(r.clip_dir.rstrip("/").split("/")[-2:])
            print(f"  {r.score:.4f}  {clip_id}  <- \"{r.matched_sentence[:70]}\"")
        print()
