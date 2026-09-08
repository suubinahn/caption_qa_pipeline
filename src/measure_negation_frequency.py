"""
암묵적 부정문(implicit negation) 빈도 측정
=================================================

search_module.py의 penalize_negation은 "not"/"no" 같은 명시적 부정어만
잡는다(한계 10번). "정체 없이 원활하다"처럼 부정어 없이 반대 의미를
나타내는 암묵적 부정 표현은 여전히 못 잡는데, 이게 실제 데이터에서
얼마나 자주 나오는지 측정된 적이 없었다. NLI 같은 무거운 걸 도입할
가치가 있는지 판단하려면 먼저 빈도부터 싸게 재보는 게 순서다.

측정 대상 두 그룹:
  1) 명시적 부정어(not/no/never/without 등 "no"류) - 이미 완화책이 커버
  2) 암묵적 부정 표현 - "free of", "clear of", "devoid of", "lacking",
     "absent", "none" 등 부정어(not/no)를 안 쓰고 "없음/부재"를
     표현하는 어휘 - 완화책이 못 잡는 그룹

이건 정확한 분류기가 아니라 빈도의 "하한선" 추정을 위한 저비용 키워드
스캔이다. 실제 문장 몇 개를 무작위로 뽑아 사람이 직접 확인하는 것도
같이 한다.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from validate_sentence_split_recall import split_sentences

FULL_DATASET_JSON = Path("outputs/search_full_dataset.json")

EXPLICIT_NEGATION = re.compile(r"\b(not|no|never|n't|without|nobody|nothing|none)\b", re.IGNORECASE)

# "not/no/without" 없이 "없음/부재"를 나타내는 어휘 - 암묵적 부정 후보
IMPLICIT_NEGATION_PHRASES = re.compile(
    r"\b(free of|clear of|devoid of|lacking|absent|empty of|"
    r"unobstructed|light traffic|moderate traffic)\b",
    re.IGNORECASE,
)


def main():
    with open(FULL_DATASET_JSON, encoding="utf-8") as f:
        clips = json.load(f)

    all_sentences = []
    for c in clips:
        path = Path(c["clip_dir"]) / "caption.txt"
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        all_sentences.extend(split_sentences(text))

    n_total = len(all_sentences)
    explicit_hits = [s for s in all_sentences if EXPLICIT_NEGATION.search(s)]
    implicit_hits = [s for s in all_sentences if IMPLICIT_NEGATION_PHRASES.search(s) and not EXPLICIT_NEGATION.search(s)]

    print(f"전체 문장 수: {n_total}")
    print(f"명시적 부정어(not/no/never/without 등) 포함: {len(explicit_hits)}개 ({len(explicit_hits)/n_total*100:.2f}%)")
    print(f"암묵적 부정 후보(free of/clear/smooth 등, 명시적 부정어 없이): {len(implicit_hits)}개 ({len(implicit_hits)/n_total*100:.2f}%)")

    random.seed(3)
    sample = random.sample(implicit_hits, min(15, len(implicit_hits)))
    print(f"\n암묵적 부정 후보 무작위 표본 {len(sample)}개 (실제로 반대 의미를 뜻하는지 육안 확인용):")
    for s in sample:
        print(f"  - {s.strip()}")


if __name__ == "__main__":
    main()
