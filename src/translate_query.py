"""
한국어 검색어 -> 영어 번역 전처리
====================================

search_module.py는 openai/clip-vit-base-patch32(영어 위주 학습)를 쓰기 때문에
한국어 검색어를 그대로 넣으면 사실상 작동하지 않는다 (실측: "터널 통과" 검색
81.2% -> 6.2%로 붕괴, 영-한 번역쌍 코사인 유사도도 0.57~0.71에 불과함).

다국어 CLIP으로 모델 자체를 바꾸는 대신, 검색어를 영어로 번역한 뒤 기존에
검증해둔 영어 파이프라인(CLIP 텍스트 인코더, CSLS, 컷오프)을 그대로 재사용
하는 방식을 택했다 - 캡션 인덱스, CSLS anchor, VALIDATED_QUERY_THRESHOLDS를
전부 다시 검증할 필요가 없기 때문.

Helsinki-NLP/opus-mt-ko-en (MarianMT)로 실측한 결과, 9개 도메인 검색어 중
7개가 영어 원본과 거의 같거나 더 나은 precision을 회복했다(예: 터널
6.2%->81.2%, 다리 0%->12.5%, 눈 12.5%->25.0%). "정지"만 회복 안 됐는데
(0.0%->0.0%), 이건 번역 문제가 아니라 영어 원본 자체가 이미 약했던
쿼리라서다(9.1%, KNOWN_LIMITATIONS의 상태동사 문제와 동일 원인).

한계: 번역 품질이 완벽하지 않다 ("정지해 있는 장면" -> "Car's still on."
같은 어색한 오역 존재). 그리고 VALIDATED_QUERY_THRESHOLDS는 정확한 영어
문자열로 매칭되므로, 번역된 문장이 등록된 영어 쿼리와 토씨 하나 안 틀리고
같지 않으면 컷오프가 적용되지 않고 순위만 나온다 (예: "차가 터널을
통과하는 장면"을 번역하면 "Car passing through the tunnel."이 되는데,
등록된 건 "The car drives through a tunnel."이라 문자열이 다름 - 컷오프
미적용, 순위 결과 자체는 정상).
"""

from __future__ import annotations

MODEL_NAME = "Helsinki-NLP/opus-mt-ko-en"
_model = None
_tokenizer = None


def _load_translator():
    global _model, _tokenizer
    if _model is None:
        from transformers import MarianMTModel, MarianTokenizer
        print(f"[translate_query] 번역 모델 로딩 중... ({MODEL_NAME})")
        _tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
        _model = MarianMTModel.from_pretrained(MODEL_NAME, use_safetensors=True)
    return _model, _tokenizer


def translate_ko_to_en(texts: list[str]) -> list[str]:
    """한국어 문장 리스트를 영어로 번역한다 (배치 처리)."""
    model, tokenizer = _load_translator()
    inputs = tokenizer(texts, return_tensors="pt", padding=True)
    translated_ids = model.generate(**inputs)
    return [tokenizer.decode(o, skip_special_tokens=True) for o in translated_ids]


if __name__ == "__main__":
    demo = ["차량이 우회전하는 장면", "비가 오는 날 주행하는 장면"]
    for ko, en in zip(demo, translate_ko_to_en(demo)):
        print(f"{ko} -> {en}")
