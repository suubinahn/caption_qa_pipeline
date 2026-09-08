# 검색 후 검증 — 주행영상 캡션 검색+검증 파이프라인

회사 주행영상 데이터셋에서 자연어로 원하는 장면을 검색하고(①), 찾아낸 캡션 문장이
실제 영상과 맞는지 별도로 검증해서(②) "검색 관련성"과 "캡션 신뢰도"를 독립적인
정보로 함께 보여주는 파이프라인입니다.

전체 배경/설계 이유/실험 과정은 [`results/final_report.md`](results/final_report.md)에,
알려진 한계는 [`KNOWN_LIMITATIONS.md`](KNOWN_LIMITATIONS.md)에 정리돼 있습니다.
이 문서는 "설치하고 실행해보는 법"만 다룹니다.

## 구조

```
검색어 → [①검색 src/search_module.py] → 후보 클립 + 매칭 문장
       → [②검증 src/search_and_verify.py] → 관련성 + 신뢰도 등급
```

- **①검색**: 캡션을 문장 단위로 쪼개 CLIP 텍스트 인코더로 임베딩, 코사인 유사도 +
  CSLS 허브 보정으로 순위를 매깁니다.
- **②검증**: ①이 찾은 문장이 실제 영상과 맞는지 BLIP-ITM으로 대조합니다
  (motion="정지" 판정만 옵티컬 플로우로 대체 — `KNOWN_LIMITATIONS.md` 16번 참고).

## 설치

```bash
# 1) 가상환경 생성 (Python 3.9 기준)
python -m venv .venv
source .venv/bin/activate

# 2) PyTorch — GPU(CUDA) 환경이면 공식 인덱스에서 설치해야 CUDA 가속 wheel이 잡힙니다
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
# CPU만 있다면: pip install torch==2.5.1 torchvision==0.20.1 (속도는 느림)

# 3) 나머지 의존성
pip install -r requirements.txt
```

GPU 없이도 동작하지만, BLIP-ITM 검증(②) 단계가 상당히 느려집니다.

## 새 데이터셋 연결

이 파이프라인의 모든 컷오프/신뢰도 등급은 **지금 회사 데이터셋(740클립)에 맞춰
calibration/holdout으로 검증된 값**입니다 — 새 데이터셋을 붙이면 코드는 그대로
동작하지만, 임계값이 그대로 맞는다는 보장은 없습니다(`KNOWN_LIMITATIONS.md` 참고).

1. **매니페스트 생성** — 클립 목록 + info.txt 값을 스캔해서 JSON으로 만듭니다:

   ```bash
   python src/scan_dataset.py <데이터 루트 경로> --out outputs/search_full_dataset.json
   ```

   `caption.txt`와 `info.txt`(motion/road_context/road_type/time_of_day/surface/weather
   6줄 고정 순서)가 같이 있는 폴더를 클립으로 자동 인식합니다.

2. **어휘 확인** (선택) — `search_and_verify.check_data_vocabulary(clips)`로 새
   데이터의 info.txt 값이 검증된 어휘 범위(motion 5종/weather 4종/time_of_day 2종)
   안에 있는지 미리 확인할 수 있습니다.

3. **재보정 필요 여부 판단** — 어휘가 완전히 다르거나(새로운 motion 값 등), 그냥
   "숫자가 이 새 데이터에서도 맞는지 불안하다"면 `calibrate_match_score_threshold.py`
   / `calibrate_optical_flow_threshold.py`를 다시 돌려서 임계값을 갱신하세요.

## 실행

```bash
python src/run_search_pilot.py "It is raining."
python src/run_search_pilot.py "The vehicle is stopped." --top_k 10 --detailed_claims
python src/run_search_pilot.py "정체가 심하다" --lang ko
python src/run_search_pilot.py "It is raining." --out outputs/pilot_rain.csv
python src/run_search_pilot.py "It is raining." --feedback   # 관련성 피드백 대화형 모드 (실험적)
```

인덱스(캡션 임베딩)는 `outputs/.index_cache/`에 자동으로 디스크 캐싱됩니다 — 같은
데이터셋에 검색어만 바꿔가며 여러 번 실행하면 두 번째 실행부터 훨씬 빠릅니다.

전체 옵션은 `python src/run_search_pilot.py --help` 참고.

## 알아둘 점

- 검증된 검색어 9개 중 **터널·비 2개만** 관련성 컷오프(`is_relevant`)가 있습니다.
  나머지는 순위만 제공됩니다 — 데이터가 희소해서 컷오프를 못 정한 것이지 버그가
  아닙니다.
- motion 판정은 "정지"를 제외하면 holdout accuracy 68.5%로 가장 약합니다(동전
  던지기보다 약간 나은 수준) — `confidence_tier` 필드로 반드시 등급을 같이
  확인하세요.
- `run_pilot.py`는 이 파이프라인과 무관한 **1단계(레거시) 캡션 사실검증**
  (`full_inspector.py`)용 CLI입니다 — 헷갈리지 않도록 주의.

## 문서 지도

| 파일 | 용도 |
|---|---|
| `results/final_report.md` | 전체 실험/설계 서술형 리포트 |
| `KNOWN_LIMITATIONS.md` | 확인된 한계 전체 목록(체크리스트형) |
| `src/search_module.py`, `src/search_and_verify.py` | 각 파일 상단 docstring에 해당 모듈의 설계 이유·검증 근거가 정리돼 있음 |
