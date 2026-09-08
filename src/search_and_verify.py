"""
검색 + 캡션-영상 매칭 검증 통합 파이프라인
================================================

지금까지 두 파이프라인이 서로 연결되지 않은 채 따로 존재했다:
  - search_module.py: ①검색어 <-> 캡션 텍스트 비교 (CLIP 텍스트 인코더 + CSLS)
  - full_inspector.py 등 (1~6단계): ②캡션 <-> 영상 매칭 검증 (BLIP-ITM 멀티프레임)

실제 목표는 "검색어로 찾은 캡션이, 실제로 그 영상 내용과 맞는가"까지
확인하는 것이므로 - 검색만 해서는 "캡션이 그럴듯하게 검색어와 비슷해서
뽑혔을 뿐 실제로는 틀린 내용"인 경우를 못 걸러낸다 - 이 모듈에서 둘을
연결한다.

절차:
  1) CaptionSearchIndex.search()로 top_k 후보를 찾는다 (①)
  2) 각 후보에서 "검색어와 가장 잘 맞았던 문장"(matched_sentence)을 골라,
     실제 영상 프레임과 BLIP-ITM으로 대조한다 (②, stage 6에서 확정한
     "멀티프레임 + 정답점수 기반 최댓값" 방법 그대로 - 오답 없이도
     오라클 대비 90.5% 성능이 나왔던 실전 채택 방법)

주의(중요): full_inspector.py의 PASS/REVIEW/FAIL 판정 임계값
(LOW_THRESHOLD=0.9101, HIGH_THRESHOLD=0.9560 등)은 MSVD/Wikimedia
공개 데이터로 보정된 것이라 회사 실주행 캡션에는 안 맞는다는 게 이미
확인됐다(회사 파일럿 10개 전부 REVIEW로 오판 - "ANY 이상 flag시 REVIEW"
규칙과 44.7단어 평균 캡션 길이 조합이 통계적으로 99.1% 확률로 최소
1개는 오탐하게 만듦). 그래서 여기서는 그 판정 체계를 재사용하지 않고,
match_score(0~1 매칭 확률) 원점수만 그대로 제공한다.

[검증됨] match_score가 이 도메인에서 실제로 의미있는 신호인지 확인함
(validate_match_score.py): 회사 실제 영상 20개에 대해, info.txt의 실제
값으로 "참" 문장을, 다른 값으로 "거짓" 문장을 만들어 같은 영상 프레임과
대조했다. 60개 (참,거짓) 쌍 중 86.7%에서 참 문장이 더 높은 점수를 받았고
(평균 margin +0.125), motion/weather/time_of_day 세 필드 전부 통계적으로
유의함(p<0.005, time_of_day는 p<0.0001로 20/20 완벽 판별). 즉 노이즈가
아니라 실제 캡션 사실 여부를 구분하는 신호로 확인됨.

[해결됨] 절대 컷오프: 검색 precision@K 컷오프와 달리, 여기서는 "거짓"
문장을 직접 합성하므로 희귀 카테고리에 구애받지 않고 표본을 늘릴 수
있었다. 처음 60클립(참/거짓 120점)을 클립 단위 calibration/holdout(30/30)
분리해서 F1-최대화 임계값을 찾았고(calibrate_match_score_threshold.py),
표본이 작아 신뢰구간이 넓다는 우려로 이후 200클립(holdout 100)으로
재검증했다. 재검증 결과, "정지" 옵티컬 플로우 때와 달리 이번엔 임계값과
정확도 둘 다 눈에 띄게 움직였다(특히 weather 임계값이 4배 가까이 이동):

  - 통합 임계값 0.0729: holdout accuracy 73.7%, precision 67.8%, recall 90.0%
  - motion 전용 0.2553: holdout accuracy 68.5% - 가장 약함(60클립 기준 63.3%
    에서 개선됐지만 여전히 최약), 검색어 축(①)의 "상태동사 구분이 CLIP의
    근본 약점"과 동일한 패턴이 BLIP-ITM(②)에서도 재현됨(동작/상태 구분은
    두 모델 모두에게 어렵다)
  - weather 전용 0.0375: holdout accuracy 71.5%
  - time_of_day 전용 0.0957: holdout accuracy 97.0%, precision 97.0% - 가장
    신뢰할 만함 (낮/밤은 시각적으로 뚜렷해서 두 모델 다 잘 구분)
    (수치는 코드리뷰로 n_frames=3/5 불일치를 잡고, 이후 200클립으로 재검증한
    최종본 - 아래 [코드리뷰로 수정] 항목 참고)

[스팟체크로 확인] 실제 search_and_verify() 결과 몇 건을 프레임까지 직접
확인했다:
  - "야간" 검색에서 ①이 실수로 daytime 클립을 상위에 올렸는데(캡션이
    "bright and well-lit"이라고 서술), ②는 이걸 정확히 "캡션은 맞다"고
    판단했다(is_video_verified=True) - 검색 실수와 캡션 정확성을 분리해서
    보여주는 설계 의도대로 작동함을 확인.
  - "다차선고속도로" 검색에서 캡션은 info.txt 기준 정확한데(실제
    multi-lane highway) ②가 False로 판정한 사례 발견 - 프레임을 보니
    우천+근접 정체 차량+렌즈 위 빗방울로 화면이 심하게 가려져 있어서,
    "다차선"이라는 시각적 증거 자체가 프레임에 제대로 안 보였다. 즉
    **악천후/시야 가림 상황에서는 캡션이 맞아도 ②가 false negative를
    낼 수 있다** - 새로 확인된 한계, 아직 대응책 없음.
  - "좌회전" 검색 결과 중, motion=curved lane driving인 클립의 문장이
    "continues straight ahead"라고 서술한 경계 사례를 발견 - 통합
    임계값(당시 0.0379)으로는 통과했지만, 이건 진짜 캡션 오류가
    아니라 "완만한 커브 구간을 순간적으로는 직진처럼 묘사"한 라벨
    모호성에 가까웠다(road_type의 다차선/3차선 모호성과 동일 패턴). 다만
    이 스팟체크 과정에서 **motion 문장에 지나치게 관대한 통합값이
    적용되고 있다는 실제 설계 gap**을 발견함 -> 아래에서 해결.

[해결됨] 필드별 임계값 자동 라우팅: classify_sentence_field()로 문장이
motion/weather/time_of_day 중 무엇을 얘기하는지 키워드로 감지해서, 감지되면
FIELD_MATCH_SCORE_THRESHOLDS의 해당 필드 임계값을(예: motion=0.2553) 통합값
대신 쓴다. 감지 안 되면 통합값(0.0729)으로 fallback. 키워드 매칭이라 완벽한
분류는 아니고(동의어 누락 가능), 여러 필드가 동시에 매칭되면 motion(제일
엄격) -> time_of_day -> weather 순으로 보수적으로 택한다.

[사용자 노출용] match_score를 그대로 %로 사용자에게 보여주면 안 된다 - 필드마다
컷오프 스케일이 6배 이상 차이 나고(weather 0.0375 vs motion 0.2553), BLIP-ITM
원점수는 보정된 확률이 아니다(컷오프가 전부 0.5에서 한참 떨어져 있음). 대신
FIELD_CONFIDENCE_TIER로 "이 판정을 얼마나 믿어도 되는지"를 등급(높음/보통/
낮음)으로 매핑한다 - 새로 지어낸 값이 아니라 calibrate_match_score_threshold.py의
실측 holdout accuracy 그대로(time_of_day 97.0%->높음, weather/통합 71~74%->보통,
motion 68.5%->낮음, 200클립 기준). 등급을 3개로 제한한 이유: 필드당 표본이
100개 안팎이라 그
이상 세분화하면 세분화 자체가 검증 안 된 과잉 정밀도가 된다.
format_verification_label(result)로 "일치/불일치 의심 (신뢰도: N)" 형태의
사람이 읽을 라벨을 바로 얻을 수 있다.

[구현됨] 필드 단위 세부 분석: search_and_verify(..., detailed_claims=True)면
matched_sentence 안에서 motion/weather/time_of_day 각각에 대해 어떤 값을
주장하는지(extract_field_claims) 전부 찾아내고, 원문 그대로가 아니라
calibrate_match_score_threshold.py에서 실제 검증에 쓴 것과 동일한 깨끗한
템플릿 문장으로 바꿔서 각각 독립적으로 BLIP-ITM 검증한다(verify_caption_claims).
"이 캡션이 3가지를 주장하는데 그중 시간대만 틀린 것 같다"는 식으로 필드
단위까지 짚어줄 수 있다. VerifiedSearchResult.claims에 담기고,
format_claims_breakdown()으로 사람이 읽을 형태로 뽑을 수 있다.

원문을 그대로 쓰지 않고 템플릿으로 바꿔치기하는 이유: 우리가 가진 컷오프/
신뢰도 등급은 전부 그 템플릿으로 보정된 것이라, 원문(훨씬 길고 복잡한
문장)을 그대로 넣으면 컷오프가 애초에 다른 종류의 입력에 대해 계산된
것이 되어버린다. 다만 이 방식은 object_check.py류 접근(1~4단계 참고)과
달리 객체/속성 단위(예: "하얀 차")까지는 못 짚는다 - motion/weather/
time_of_day 3개 필드로 범위가 제한된다. 감지된 주장 개수만큼 BLIP-ITM을
추가로 돌리므로 기본값은 꺼져 있다(detailed_claims=False).

[검토 후 보류] road_context/road_type 필드 추가: 실제 캡션 문장의 67%가
현재 3개 필드 중 아무것도 안 걸려서(커버리지가 좁음), road_context/
road_type도 같은 방식으로 추가해볼지 검증했다(calibrate_road_fields_threshold.py,
60클립). 결과가 갈렸다:
  - road_type: 완전 실패 - margin 거의 0(참>거짓 45%, p=0.85, 노이즈),
    holdout accuracy 50%(=전부 True로 찍는 것과 동일). ①검색 단계에서
    이미 확인한 "다차선고속도로 vs 3차선 경계 모호성"이 ②에서도 재현됨.
    -> 추가 안 함.
  - road_context: 수치상으론 매우 강함(holdout accuracy 91.7%) - 하지만
    road_context는 캡션 생성 프롬프트에 정답(info.txt)이 그대로 주입되는
    필드라(①검색 단계에서 이미 확인, 85~97% 그대로 등장), 실제 프로덕션
    캡션에서 이 값이 틀릴 일이 구조적으로 거의 없다. 즉 검증하면 "당연히
    일치"만 반복해서 나올 뿐, "이 캡션 중 어디가 틀렸는지 짚어주기"라는
    본래 목적에는 거의 도움이 안 된다(커버리지는 33%->41.5%로 늘지만 그
    늘어난 부분 대부분이 저정보량). 드물게 "불일치 의심"이 나와도 캡션
    파이프라인 버그/info.txt 라벨 오류/도구 오차 중 뭔지 구분이 안 돼서
    해석이 어렵다. -> 추가 안 함, 지금 3개 필드(motion/weather/time_of_day)
    범위 그대로 유지하기로 결정.

[구현됨] "정지" 전용 옵티컬 플로우 보조 신호: search_and_verify()를 전수
감사(여러 검색어에 걸쳐 30개 결과를 직접 프레임까지 확인)하다가, "정지"
관련 문장이 실제로 정확한데도(배경 건물이 고정, 지나가는 차량만 바뀜)
BLIP-ITM 기반 판정이 틀린 사례를 발견했다. 원인은 구조적이다 - 지금
방식은 프레임 여러 장을 보긴 해도 서로 비교 없이 독립적으로 채점하고
최댓값만 쓰기 때문에, "배경이 프레임 사이에 안 움직였다"는 정지의 핵심
증거 자체를 볼 방법이 없다. 단순 픽셀 차이 비교는 시도해봤지만 조명
변화 등 노이즈에 취약해 실패했고, 옵티컬 플로우(프레임 간 픽셀 이동을
정식으로 추정하는 컴퓨터비전 기법)로 재시도해서 성공했다 - 프레임 상단
(배경일 가능성 높은 영역)의 평균 흐름 크기가 작을수록 정지, 크면 이동.
calibrate_optical_flow_threshold.py로 60클립(정지 30 + 그 외 30)
calibration/holdout(30/30)으로 1차 검증한 뒤(holdout accuracy 83.3%),
holdout이 작아 신뢰구간이 넓다는 우려로 200클립(정지 100 + 그 외 100)
calibration/holdout(100/100)으로 재검증했다: holdout accuracy 85.0%,
precision 86.0%, recall 84.3% - 결론은 그대로 유지되고 수치도 소폭
개선됐다(정밀도가 오르고 신뢰구간도 좁아짐). 기존 motion 임계값(BLIP-ITM
방식, 200클립 재검증 후 68.5%)보다 "정지" 판별에 한해 뚜렷이 낫다.
motion="stop" 주장(전체 문장 판정과
detailed_claims 세부
분석 둘 다)에서 BLIP-ITM 대신 이 신호를 쓰도록 교체했고, 실제로 발견한
오류 사례가 고쳐지는 것까지 확인했다. 나머지 motion 값(좌회전/우회전/
직진/커브)은 이 신호로 검증한 적 없어서 기존 방식 그대로 둔다.

[구현됨] 새 데이터 어휘 사전 점검: 여기의 컷오프/신뢰도 등급은 특정 어휘로
(motion 5종/weather 4종/time_of_day 2종) 검증됐다 - 완전히 다른 데이터(다른
카메라/다른 캡션 방식/다른 라벨링 체계)를 붙이면 코드는 그대로 돌아가지만
이 숫자들이 유효하다는 보장은 없다. check_data_vocabulary(clips)로 새
데이터의 info.txt 값들이 검증된 어휘 범위 안에 있는지 미리 확인할 수 있다
(빈 dict가 나오면 최소한 "완전히 낯선 데이터"는 아니라는 뜻 - 컷오프가
그대로 맞는다는 보장까지는 아니고, 그러려면 calibration/holdout을 실제로
다시 돌려야 한다). 이 점검 도구를 만드는 과정에서 실제로 자체 버그도
하나 찾았다 - motion 필드에 원래 5개 값(직진/좌회전/우회전/정지/커브주행)이
있는데 _VALUE_TEMPLATES에 "커브주행"을 빠뜨리고 4개만 넣어뒀던 걸 발견해서
같이 고침(_VALUE_KEYWORDS/_FIELD_KEYWORDS에도 curve 관련 키워드 추가).
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from pathlib import Path

from frame_extract import extract_frames
from blip_itm import compute_itm_score
from search_module import CaptionSearchIndex
from experiment_optical_flow_motion import background_flow_magnitude, flow_magnitude_from_frames

# ["stop" 전용 보조 신호] BLIP-ITM 프레임 독립채점의 구조적 한계 - 여러
# 프레임을 봐도 서로 비교하지 않고 각자 채점하기 때문에 "배경이 프레임
# 사이에 안 움직였다"(=정지 증거)를 볼 방법이 없다. 실제 전수 감사(30개)에서
# "정지" 관련 문장이 정확한데도 틀리게 판정된 사례를 발견한 뒤(홀드아웃
# 정확도 63.3%로 이미 알려진 약점 - 이후 match_score 자체를 200클립으로
# 재검증해서 68.5%로 갱신됨, 아래 FIELD_MATCH_SCORE_THRESHOLDS 참고), 옵티컬
# 플로우(프레임 간 픽셀 이동 추정)로 "배경이 실제로 안 움직였는지"를 직접
# 재보는 보조 신호를 추가했다. calibrate_optical_flow_threshold.py로 처음엔
# 60클립(정지 30 + 그 외 30) calibration/holdout(30/30)으로 검증(holdout
# accuracy 83.3%)했지만 표본이 작아 신뢰구간이 넓었다(대략 66~93%) - 200클립
# (정지 100 + 그 외 100) calibration/holdout(100/100)으로 재검증: holdout
# accuracy 85.0%, precision 86.0%, recall 84.3% - 결론 그대로에 수치도 소폭
# 개선(신뢰구간도 좁아짐). 기존 motion 전체 임계값(BLIP-ITM 방식, 68.5%)보다
# "정지" 판별에 한해 뚜렷이 낫다. 그래서
# motion="stop" 주장에 한해서만 이 신호로 대체하고, 나머지 motion 값
# (좌회전/우회전/직진/커브)은 기존 방식 그대로 둔다(이 신호는 "정지 여부"만
# 검증했지 회전 방향 등은 검증한 적 없음 - 좌/우회전 확장은 시도했으나
# holdout accuracy 64.3~70.6%로 미채택, KNOWN_LIMITATIONS.md 17번 참고).
OPTICAL_FLOW_STOP_THRESHOLD = 3.1960  # 이 값 이하면 "정지"로 예측 (holdout accuracy 85.0%, n=200)


# calibrate_match_score_threshold.py로 계산한 임계값들. 통합값은 필드를 특정
# 못 하는 문장에 쓰는 fallback이고, 필드별 값은 classify_sentence_field()로
# 문장이 어느 필드를 얘기하는지 감지되면 그쪽을 우선 적용한다 (motion 서술
# 문장에 통합값을 쓰면 너무 관대해서, 경계선 사례를 잘못 통과시키는 걸 실제
# 스팟체크로 확인한 뒤 이 라우팅을 추가함).
# [코드리뷰로 수정, 2차 보정] 원래 n_frames=3으로 계산했는데 search_and_verify()
# 프로덕션 기본값(n_frames=5)과 달랐다 - match_score가 "여러 프레임 중 최댓값"이라
# 프레임을 더 뽑을수록 점수가 체계적으로 올라간다(실측: 15개 샘플 중 2개 판정이
# "불일치"->"일치"로 뒤집힘, 반대 방향 0건). n_frames=5로 다시 계산한 값으로 교체함.
# [3차 보정] 60클립(holdout 30)은 표본이 작아 신뢰구간이 넓다는 우려로, 200클립
# (holdout 100)으로 재검증했다. "정지" 옵티컬 플로우 때(83.3%->85.0%, 거의 그대로)와
# 달리 이번엔 임계값과 정확도 둘 다 눈에 띄게 움직였다 - 특히 weather 임계값이
# 0.0102->0.0375로 거의 4배, motion accuracy가 63.3%->68.5%로 5.2%p 상승. 60클립
# 기준값이 예상보다 더 불안정했다는 뜻으로, 이번 재검증이 실제로 값어치가 있었다.
MATCH_SCORE_THRESHOLD = 0.0729  # fallback (holdout accuracy 73.7%, precision 67.8%, recall 90.0%, n=200)
FIELD_MATCH_SCORE_THRESHOLDS = {
    "motion": 0.2553,        # holdout accuracy 68.5%, n=200 - 가장 약한 필드, 그래도 통합값보다는 엄격하게
    "weather": 0.0375,       # holdout accuracy 71.5%, n=200
    "time_of_day": 0.0957,   # holdout accuracy 97.0%, n=200 - 가장 신뢰할 만함
}

# is_video_verified를 그대로 %로 보여주면 오해를 만든다 - 필드마다 컷오프
# 스케일이 다르고(0.0375~0.2553, 6배 이상 차이), BLIP-ITM 원점수는 보정된 확률이
# 아니다(컷오프가 전부 0.5에서 한참 떨어져 있음). 대신 실측 holdout accuracy를
# 그대로 등급 경계로 써서 "이 판정을 얼마나 믿어도 되는지"를 등급으로 보여준다
# (숫자를 새로 지어내지 않음 - calibrate_match_score_threshold.py의 실측치
# 그대로, n=200 재검증 반영). 등급 3개(높음/보통/낮음)만 쓰는 이유: 표본이
# 필드당 100개 안팎(재검증 후)이라 그 이상 세분화하면(예: 4~5단계) 그 세분화
# 자체가 검증 안 된 과잉 정밀도가 된다. motion이 68.5%로 올라 weather/통합
# (71.5~73.7%)과의 격차가 좁아졌지만, 여전히 뚜렷한 차이가 있어 "낮음"을 유지.
FIELD_CONFIDENCE_TIER = {
    "time_of_day": "높음",   # holdout accuracy 97.0%, n=200
    "weather": "보통",       # holdout accuracy 71.5%, n=200
    None: "보통",            # 통합/fallback, holdout accuracy 73.7%, n=200 (weather와 비슷한 수준)
    "motion": "낮음",        # holdout accuracy 68.5%, n=200 - 개선됐지만 여전히 가장 약한 필드
}

# classify_sentence_field()가 문장에서 필드를 감지하는 데 쓰는 키워드.
# 완벽한 분류기가 아니라 단순 키워드 매칭이라 동의어를 다 못 잡을 수 있다.
# 여러 필드 키워드가 동시에 매칭되면 motion(가장 엄격) -> time_of_day ->
# weather 순으로 우선 적용한다 (보수적으로 더 엄격한 쪽을 택함).
_FIELD_KEYWORDS = {
    "motion": ["stopped", "stationary", "halted", "standstill", "at rest",
               "turning left", "turns left", "turning right", "turns right",
               "moving straight", "moving forward", "driving straight", "straight ahead", "straight path",
               "curve", "curving", "curved lane", "bend in the road"],
    "time_of_day": ["night", "nighttime", "daytime", "during the day"],
    "weather": ["sunny", "clear sky", "rain", "rainy", "raining", "cloudy", "overcast", "snow", "snowy", "snowing"],
}


def classify_sentence_field(sentence: str) -> str | None:
    """문장이 motion/time_of_day/weather 중 어느 필드를 얘기하는지 키워드로
    추정한다. 못 찾으면 None (호출자는 MATCH_SCORE_THRESHOLD로 fallback)."""
    sentence_lower = sentence.lower()
    for field in ("motion", "time_of_day", "weather"):  # 엄격한 순서
        if any(kw in sentence_lower for kw in _FIELD_KEYWORDS[field]):
            return field
    return None


# extract_field_claims()용 - "필드가 뭔지"뿐 아니라 "어느 값을 주장하는지"까지
# 잡아서, calibrate_match_score_threshold.py에서 실제로 검증에 썼던 것과
# 똑같은 깨끗한 템플릿 문장으로 바꿔치기한다. 원문 그대로("as it approaches
# an intersection where traffic signals are visible..." 같은 긴 문장)를
# BLIP-ITM에 넣는 대신 이 템플릿을 쓰는 이유: 우리가 가진 컷오프/신뢰도 등급은
# 전부 이 템플릿으로 보정된 것이라, 원문을 그대로 쓰면 컷오프가 애초에 다른
# 종류의 입력에 대해 계산된 것이 되어 버린다 - 템플릿을 써야 "검증된 그 조건
# 그대로" 재현된다.
_VALUE_KEYWORDS = {
    "motion": {
        "stop": ["stopped", "stationary", "halted", "standstill", "at rest"],
        "turning left": ["turning left", "turns left"],
        "turning right": ["turning right", "turns right"],
        "moving straight": ["moving straight", "moving forward", "driving straight", "straight ahead", "straight path"],
        "curved lane driving": ["curve", "curving", "curved lane", "bend in the road"],
    },
    "weather": {
        "sunny": ["sunny", "clear sky"],
        "rainy": ["rain", "rainy", "raining"],
        "cloudy": ["cloudy", "overcast"],
        "snowy": ["snow", "snowy", "snowing"],
    },
    "time_of_day": {
        "nighttime": ["night", "nighttime"],
        "daytime": ["daytime", "during the day"],
    },
}

# calibrate_match_score_threshold.py / validate_match_score.py에서 실제
# 검증에 쓴 것과 동일한 템플릿 문장 (값 하나당 문장 하나).
_VALUE_TEMPLATES = {
    "motion": {
        "moving straight": "The vehicle is moving straight.",
        "turning right": "The vehicle is turning right.",
        "turning left": "The vehicle is turning left.",
        "stop": "The vehicle is stopped.",
        "curved lane driving": "The vehicle is driving through a curve.",
    },
    "weather": {
        "sunny": "The weather is sunny.",
        "rainy": "The weather is rainy.",
        "cloudy": "The weather is cloudy.",
        "snowy": "It is snowing.",
    },
    "time_of_day": {
        "daytime": "It is daytime.",
        "nighttime": "It is nighttime.",
    },
}


def check_data_vocabulary(clips: list[dict]) -> dict[str, set[str]]:
    """새 데이터셋을 이 파이프라인에 붙이기 전에 실행하는 사전 점검 도구.
    이 모듈의 검증된 컷오프/신뢰도 등급은 특정 어휘(motion 4종, weather 4종,
    time_of_day 2종)로 calibration/holdout 검증된 것이다 - 새 데이터의
    info.txt에 이 어휘에 없는 값(예: motion="reversing")이 있으면, 그
    필드의 컷오프/등급이 새 데이터에도 그대로 맞는다는 보장이 없다.

    프로덕션 검색 경로(search_and_verify())는 info.txt를 아예 안 읽으므로
    이 검사에 관여하지 않는다 - 새 데이터를 들여올 때 사람이 한 번
    실행해보는 용도다. clips는 outputs/search_full_dataset.json 형식
    (각 원소가 motion/weather/time_of_day 키를 가진 dict)을 가정한다.

    반환값이 빈 dict면 "이 3개 필드에 한해서는" 지금까지 검증한 어휘
    범위 안에 있다는 뜻 - 컷오프가 그대로 맞는다는 걸 보장하진 않는다
    (그러려면 실제로 calibration/holdout을 다시 돌려야 한다), 최소한
    "완전히 낯선 데이터"는 아니라는 신호일 뿐이다.
    """
    unknown: dict[str, set[str]] = {}
    for field, templates in _VALUE_TEMPLATES.items():
        known_values = set(templates.keys())
        seen_values = {c[field] for c in clips if field in c}
        new_values = seen_values - known_values
        if new_values:
            unknown[field] = new_values
    return unknown


@dataclass
class FieldClaim:
    field: str  # motion / weather / time_of_day
    value: str  # 예: "stop", "rainy", "nighttime"
    template_sentence: str  # 실제 검증에 쓴 것과 동일한 깨끗한 문장
    match_score: float  # 이 템플릿 문장의 BLIP-ITM 매칭 확률
    is_verified: bool  # match_score >= FIELD_MATCH_SCORE_THRESHOLDS[field]
    confidence_tier: str  # FIELD_CONFIDENCE_TIER[field]


def extract_field_claims(sentence: str) -> list[tuple[str, str]]:
    """문장 안에서 motion/weather/time_of_day 각각에 대해 어떤 값을
    주장하는지 전부 찾아낸다 (한 문장에 여러 필드가 섞여 있어도 전부 감지).
    classify_sentence_field()와 달리 필드 하나만 고르지 않고, 검출된 전부를
    (field, value) 쌍 리스트로 반환한다."""
    sentence_lower = sentence.lower()
    claims = []
    for field, value_kw_map in _VALUE_KEYWORDS.items():
        for value, keywords in value_kw_map.items():
            if any(kw in sentence_lower for kw in keywords):
                claims.append((field, value))
                break  # 이 필드는 값 하나만(먼저 매칭된 것) - 같은 필드에 상충 값이 동시에 매칭되는 경우는 드묾
    return claims


def verify_claims_against_frames(frames: list, sentence: str) -> list[FieldClaim]:
    """verify_caption_claims()와 동일하지만, video_path를 받아 매번 새로
    디코딩하는 대신 이미 추출된 프레임 리스트를 받는다. search_and_verify()가
    match_score용으로 이미 뽑아둔 프레임을 재사용해 중복 디코딩을 피하려고
    분리함 (코드리뷰로 발견된 성능 문제 - 문장 하나에 필드 주장이 3개면
    영상을 4번(match_score 1 + claims 3) 따로 디코딩하고 있었음)."""
    claims = extract_field_claims(sentence)
    results = []
    flow_mag = None  # 필요할 때(=="stop" 주장이 있을 때)만 한 번 계산해서 재사용
    for field, value in claims:
        template = _VALUE_TEMPLATES[field][value]
        score = max(compute_itm_score(f, template) for f in frames)

        if field == "motion" and value == "stop":
            # ["stop" 전용] BLIP-ITM 임계값(motion, holdout 68.5%) 대신
            # 옵티컬 플로우 기반 판정(holdout 85.0%, n=200)을 쓴다 - match_score는
            # 참고용으로 계속 기록하되 최종 판정(is_verified)엔 안 쓴다.
            if flow_mag is None:
                flow_mag = flow_magnitude_from_frames(frames)
            is_verified = flow_mag <= OPTICAL_FLOW_STOP_THRESHOLD
            confidence_tier = "보통"  # 실측 85.0%(n=200) - FIELD_CONFIDENCE_TIER의 "보통"(71~74%)보다 실제론 더 신뢰할 만함
        else:
            threshold = FIELD_MATCH_SCORE_THRESHOLDS[field]
            is_verified = score >= threshold
            confidence_tier = FIELD_CONFIDENCE_TIER[field]

        results.append(FieldClaim(
            field=field, value=value, template_sentence=template, match_score=score,
            is_verified=is_verified, confidence_tier=confidence_tier,
        ))
    return results


def verify_caption_claims(video_path: str, sentence: str, n_frames: int = 5) -> list[FieldClaim]:
    """문장에서 감지된 필드별 주장을 전부 뽑아, 각각을 검증된 템플릿
    문장으로 바꿔서 독립적으로 BLIP-ITM 검증한다. "이 캡션의 어느 부분이
    안 맞는지"를 필드 단위로 짚어줄 수 있다 (전체 문장 하나의 match_score
    만으로는 못 하는 것). video_path만 있고 프레임을 아직 안 뽑은 경우를
    위한 편의 함수 - 프레임을 이미 갖고 있다면 verify_claims_against_frames()를
    직접 쓰는 게 더 빠르다(search_and_verify()가 그렇게 함)."""
    frames = extract_frames(video_path, n_frames=n_frames)
    return verify_claims_against_frames(frames, sentence)


@dataclass
class VerifiedSearchResult:
    clip_dir: str
    search_score: float  # ①: 검색어-캡션 유사도 (use_csls 여부는 호출 시점에 결정됨)
    is_relevant: bool | None  # ①: 검증된 컷오프가 있는 검색어만 채워짐 (search_module.py 참고)
    matched_sentence: str  # 검색어와 가장 잘 맞았던 캡션 문장
    match_score: float  # ②: 그 문장이 실제 영상과 얼마나 일치하는지 (BLIP-ITM 멀티프레임 최댓값, 0~1) - 필드마다 스케일이 달라 그대로 %로 보여주면 안 됨, confidence_tier 참고
    is_video_verified: bool | None  # ②: match_score >= MATCH_SCORE_THRESHOLD (motion 서술 문장은 신뢰도 낮음, 위 참고)
    confidence_tier: str | None  # ②: 이 is_video_verified 판정을 얼마나 믿어도 되는지 ("높음"/"보통"/"낮음"/None=판단불가) - FIELD_CONFIDENCE_TIER의 실측 holdout accuracy 기반
    claims: list[FieldClaim] = dataclass_field(default_factory=list)  # detailed_claims=True일 때만 채워짐 - matched_sentence 안의 필드별 주장을 쪼개 각각 검증한 결과
    video_path: str = ""


def format_verification_label(result: "VerifiedSearchResult") -> str:
    """is_video_verified + confidence_tier를 사람이 읽을 라벨 하나로 합친다.
    match_score를 그대로 %로 노출하지 않는 이유는 모듈 docstring 참고."""
    if result.is_video_verified is None:
        return "판단불가 (영상 없음)"
    verdict = "일치" if result.is_video_verified else "불일치 의심"
    return f"{verdict} (신뢰도: {result.confidence_tier})"


def format_claims_breakdown(result: "VerifiedSearchResult") -> str:
    """detailed_claims=True로 얻은 claims를 사람이 읽을 세부 내역으로
    포맷한다 (예: 'weather=rainy: 일치(보통) / time_of_day=nighttime: 불일치 의심(높음)')."""
    if not result.claims:
        return "(세부 주장 없음 - detailed_claims=True로 호출했는지, 또는 감지된 필드 키워드가 없는지 확인)"
    parts = []
    for c in result.claims:
        verdict = "일치" if c.is_verified else "불일치 의심"
        parts.append(f"{c.field}={c.value}: {verdict}({c.confidence_tier})")
    return " / ".join(parts)


def verify_caption_matches_video(video_path: str, sentence: str, n_frames: int = 5) -> float:
    """stage 6에서 확정한 방법 그대로: 영상에서 N개 프레임을 뽑아, 이 문장과
    가장 잘 맞는 프레임의 BLIP-ITM 매칭 확률을 "영상 일치도"로 삼는다.
    (오답 캡션 없이도 계산 가능 - 정답점수만으로 최댓값을 고르는 실전 방법)"""
    frames = extract_frames(video_path, n_frames=n_frames)
    return max(compute_itm_score(f, sentence) for f in frames)


def score_sentence_against_frames(frames: list, sentence: str) -> float:
    """verify_caption_matches_video()와 동일한 계산이지만, 이미 추출된
    프레임 리스트를 받아 재디코딩을 피한다 (search_and_verify()가
    match_score/claims 계산에서 프레임을 공유하려고 분리함)."""
    return max(compute_itm_score(f, sentence) for f in frames)


def search_and_verify(
    index: CaptionSearchIndex,
    query: str,
    top_k: int = 5,
    use_csls: bool = True,
    lang: str = "en",
    video_filename: str = "1_clip/5.mp4",
    n_frames: int = 5,
    detailed_claims: bool = False,
) -> list[VerifiedSearchResult]:
    """①검색으로 top_k 후보를 찾고, ②각 후보의 matched_sentence가 실제
    영상과 맞는지 검증해서 하나로 합쳐 반환한다.

    detailed_claims=True면 matched_sentence 안에서 감지되는 필드별 주장을
    전부 쪼개 각각 독립적으로 검증한 결과(claims)도 채워준다 - "이 캡션의
    어느 부분이 안 맞는지"까지 짚어줄 수 있다(extract_field_claims 참고).
    단, 감지된 주장 개수만큼 BLIP-ITM을 추가로 돌리므로 기본은 꺼져 있다
    (문장 하나에 필드 주장이 0~3개 정도 섞여 있는 게 보통이라, 켜면 결과당
    BLIP-ITM 호출 자체는 최대 3~4배 늘어남 - 다만 영상 프레임 디코딩은
    match_score/claims가 공유해서 한 번만 하도록 고쳐뒀다(코드리뷰로 발견된
    중복 디코딩 문제 수정), 그래서 실제 체감 배율은 이보다 낮다)."""
    search_results = index.search(query, top_k=top_k, use_csls=use_csls, lang=lang)

    verified = []
    for r in search_results:
        video_path = Path(r.clip_dir) / video_filename
        if not video_path.exists():
            match_score = float("nan")
            is_video_verified = None
            confidence_tier = None
            claims = []
        else:
            # 프레임을 한 번만 뽑아서 match_score와 claims 계산에 재사용한다
            # (코드리뷰로 발견 - 예전엔 match_score용 1번 + claims용 문장당 1번씩
            # 따로 디코딩해서, detailed_claims=True일 때 같은 영상을 최대 4번
            # 중복 디코딩하고 있었음).
            frames = extract_frames(str(video_path), n_frames=n_frames)
            match_score = score_sentence_against_frames(frames, r.matched_sentence)
            sentence_field = classify_sentence_field(r.matched_sentence)

            if sentence_field == "motion" and any(kw in r.matched_sentence.lower() for kw in _VALUE_KEYWORDS["motion"]["stop"]):
                # ["stop" 전용] 실제 전수 감사(30개)에서 "정지" 관련 문장이
                # BLIP-ITM 임계값(holdout 68.5%)으로 잘못 판정된 사례를 발견한
                # 뒤 추가 - 옵티컬 플로우 기반 판정(holdout 85.0%, n=200)을 대신
                # 쓴다. match_score는 참고용으로 계속 기록.
                is_video_verified = flow_magnitude_from_frames(frames) <= OPTICAL_FLOW_STOP_THRESHOLD
                confidence_tier = "보통"  # 실측 85.0%(n=200) - "보통"(71~74%)보다 실제론 더 신뢰할 만함
            else:
                threshold = FIELD_MATCH_SCORE_THRESHOLDS.get(sentence_field, MATCH_SCORE_THRESHOLD)
                is_video_verified = match_score >= threshold
                # FIELD_CONFIDENCE_TIER의 키(motion/weather/time_of_day/None)가
                # classify_sentence_field()의 모든 가능한 반환값을 이미 다 커버하므로
                # else 분기는 불필요 - 코드리뷰로 발견된 죽은 코드 정리.
                confidence_tier = FIELD_CONFIDENCE_TIER[sentence_field]

            claims = verify_claims_against_frames(frames, r.matched_sentence) if detailed_claims else []
        verified.append(VerifiedSearchResult(
            clip_dir=r.clip_dir, search_score=r.score, is_relevant=r.is_relevant,
            matched_sentence=r.matched_sentence, match_score=match_score,
            is_video_verified=is_video_verified, confidence_tier=confidence_tier,
            claims=claims, video_path=str(video_path),
        ))
    return verified


if __name__ == "__main__":
    import json
    from search_module import VALIDATED_ANCHOR_QUERIES

    with open("outputs/search_sample_100.json", encoding="utf-8") as f:
        clips = json.load(f)
    clip_dirs = [c["clip_dir"] for c in clips]

    print(f"검색 인덱스 빌드 중... ({len(clip_dirs)}개 클립)")
    index = CaptionSearchIndex(clip_dirs, anchor_queries=VALIDATED_ANCHOR_QUERIES)
    print("빌드 완료.\n")

    for query in ["The car drives through a tunnel.", "The vehicle is stopped."]:
        print(f'[검색어] "{query}"')
        results = search_and_verify(index, query, top_k=5)
        for r in results:
            clip_id = "/".join(r.clip_dir.rstrip("/").split("/")[-2:])
            print(f"  검색점수={r.search_score:.4f}  {format_verification_label(r)}  {clip_id}")
            print(f"    매칭문장: \"{r.matched_sentence[:80]}\"")
        print()
