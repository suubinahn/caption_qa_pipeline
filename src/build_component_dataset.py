"""
객체/행동 단위 임계값 재보정 - 1~2단계: 라벨링된 데이터셋 구축
==================================================================

[라벨링 방법 - 왜 사람이 일일이 라벨을 달지 않는가]
각 샘플의 정답 캡션(correct_caption)은 이미 우리가 실제 프레임을 보고
검증해서 쓴 "그 영상에 대한 정답"이다. 오답 캡션들은 전부 정답 캡션에서
"정확히 무엇을 바꿨는지"를 우리가 이미 알고 만든 것이다(3단계: 행동만
변경 / 속성만 변경 / 다른 샘플 재활용). 그래서 사람이 다시 눈으로
확인하며 라벨을 달 필요 없이, 다음 규칙만으로 자동으로 정확한 라벨을
매길 수 있다:

    어떤 단어가 "정답 캡션에서 추출된 단어 집합"에 들어있다
    -> 그 단어는 이 영상에 실제로 있는 진짜 객체/행동이다 (TRUE)

    어떤 단어가 오답 캡션에만 등장하고 정답 캡션 단어 집합에는 없다
    -> 그 단어는 이 영상에 없는 가짜 객체/행동이다 (FALSE)

이 규칙은 wrong_caption(다른 샘플 재활용), wrong_caption_action(동사만
교체), wrong_caption_attribute(명사만 교체) 세 종류 오답 모두에 예외
없이 그대로 적용된다 - 그 오답들이 정답과 정확히 어디서 갈라지는지는
이미 우리가 설계 단계에서 알고 만들었기 때문이다.

[중복 계산 방지]
같은 단어가 같은 영상에 대해 여러 캡션(정답/오답1/오답2/오답3)에
동시에 등장할 수 있다 (예: "outdoors"는 정답에도 오답에도 다 있음).
이런 경우 (영상, 단어) 쌍은 한 번만 채점하면 된다 - 같은 영상에 같은
질문("a photo of outdoors 있나요?")을 두 번 물을 필요가 없다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from object_check import extract_nouns
from action_check import extract_actions

ROOT = Path(__file__).resolve().parent.parent
CAPTIONS_JSON = ROOT / "data" / "captions.json"
OUTPUT_CSV = ROOT / "outputs" / "component_dataset.csv"


def extract_words(caption: str) -> set[tuple[str, str]]:
    """캡션에서 명사+동사를 전부 뽑아 (단어, 품사태그) 집합으로 반환."""
    nouns = set(extract_nouns(caption))
    actions = set(extract_actions(caption))
    return nouns | actions


def build_dataset() -> pd.DataFrame:
    with open(CAPTIONS_JSON, "r", encoding="utf-8") as f:
        samples = json.load(f)["samples"]

    rows = []
    for s in samples:
        sid = s["id"]
        video_file = s["video_file"]

        # 정답 캡션에서 나온 단어 집합 = "이 영상의 진짜 객체/행동" 정답 기준
        true_words = extract_words(s["correct_caption"])
        true_word_set = {w for w, _tag in true_words}

        # 4개 캡션(정답+오답3) 전체에서 나오는 단어를 다 모으되, 같은
        # (영상, 단어) 쌍은 한 번만 남긴다.
        all_words: dict[str, str] = {}  # word -> tag (같은 단어면 태그도 같다고 가정)
        for caption_field in ["correct_caption", "wrong_caption", "wrong_caption_action", "wrong_caption_attribute"]:
            for word, tag in extract_words(s[caption_field]):
                all_words[word] = tag

        for word, tag in all_words.items():
            is_true = word in true_word_set
            component_type = "action" if tag == "VBG" else "object"
            rows.append({
                "id": sid,
                "video_file": video_file,
                "word": word,
                "tag": tag,
                "type": component_type,
                "is_true": is_true,
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = build_dataset()
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"총 {len(df)}개의 (영상, 단어) 쌍 생성")
    print(f"  - object: {(df['type']=='object').sum()}개 (진짜 {((df['type']=='object')&(df['is_true'])).sum()} / 가짜 {((df['type']=='object')&(~df['is_true'])).sum()})")
    print(f"  - action: {(df['type']=='action').sum()}개 (진짜 {((df['type']=='action')&(df['is_true'])).sum()} / 가짜 {((df['type']=='action')&(~df['is_true'])).sum()})")
    print(f"\n샘플별 개수:")
    print(df.groupby(["id"]).size())
    print(f"\n저장 위치: {OUTPUT_CSV}")

    print("\n전체 데이터:")
    print(df.to_string(index=False))
