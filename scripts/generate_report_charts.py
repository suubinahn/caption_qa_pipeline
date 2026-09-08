"""
최종 리포트(results/final_report.md)에 들어갈 4개의 차트를 생성한다.
모든 수치는 outputs/stage*.csv에 이미 저장된 실제 실험 결과에서 그대로
가져온 것이며, 여기서 새로 계산하지 않는다 (재현성/일관성을 위해 CSV를
단일 진실 공급원(source of truth)으로 유지).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "results" / "images"
OUT.mkdir(parents=True, exist_ok=True)

# 한글 텍스트(그래프 제목/축/범례)가 깨지지 않도록, 시스템에 설치된
# Noto Sans CJK KR 폰트를 matplotlib에 명시적으로 등록해서 사용한다.
# (기본 폰트인 DejaVu Sans는 한글 글리프가 없어 네모 박스로 깨져 보인다)
_NOTO_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(_NOTO_PATH)
_KO_FONT_NAME = fm.FontProperties(fname=_NOTO_PATH).get_name()

plt.rcParams.update({
    "font.family": _KO_FONT_NAME,
    "axes.unicode_minus": False,  # 한글 폰트 사용 시 마이너스 기호가 깨지는 것 방지
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "figure.dpi": 150,
})

# 색상 팔레트 (일관되게 재사용)
C_BLUE = "#3B82C4"
C_ORANGE = "#E08E45"
C_GREEN = "#4E9A6A"
C_RED = "#C0504D"
C_GRAY = "#8A8A8A"
C_PURPLE = "#8064A2"


# ---------------------------------------------------------------------------
# Fig 1. 전체 진행 흐름: 방법을 바꿀 때마다 평균 margin이 어떻게 변했는가
# (모두 "정답 - 오답3개 평균" 이라는 동일한 정의로 8개 샘플 평균한 값)
# ---------------------------------------------------------------------------
stages = [
    "CLIP\n(원본 코사인)",
    "CLIP\n+CSLS",
    "BLIP-ITM\n(단일 프레임)",
    "BLIP-ITM\n다중프레임\n(독립 pooling)",
    "BLIP-ITM\n다중프레임\n(프레임단위, 오라클)",
    "BLIP-ITM\n실전 최적\n(정답점수 max)",
]
values = [0.0748, 0.1475, 0.5732, 0.6622, 0.8239, 0.7454]
colors = [C_GRAY, C_GRAY, C_BLUE, C_ORANGE, C_GREEN, C_PURPLE]

fig, ax = plt.subplots(figsize=(10, 5.5))
bars = ax.bar(stages, values, color=colors, width=0.6)
for b, v in zip(bars, values):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
ax.set_ylabel("평균 margin (정답 점수 − 오답 3개 평균 점수, 8개 샘플 평균)")
ax.set_title("전체 실험 여정: 방법을 바꿀 때마다 평균 판별력이 어떻게 개선됐는가", fontsize=12, fontweight="bold")
ax.set_ylim(0, max(values) * 1.2)
plt.tight_layout()
plt.savefig(OUT / "fig1_stage_progression.png")
plt.close()


# ---------------------------------------------------------------------------
# Fig 2. 문제 샘플 3개(dog_park, horse_herd, people_dancing)의 구제 과정
# ---------------------------------------------------------------------------
samples = ["dog_park", "horse_herd", "people_dancing"]
method_labels = ["단일 프레임", "다중(독립 pooling)", "다중(프레임단위, 오라클)", "실전 최적(정답점수 max)"]
data = {
    "dog_park":       [0.0961, 0.3652, 0.6437, 0.3652],
    "horse_herd":     [-0.1054, 0.5078, 0.9778, 0.9778],
    "people_dancing": [0.7171, 0.5414, 0.8881, 0.7156],
}
colors2 = [C_BLUE, C_ORANGE, C_GREEN, C_PURPLE]

x = np.arange(len(samples))
width = 0.2
fig, ax = plt.subplots(figsize=(10, 5.5))
for i, (label, color) in enumerate(zip(method_labels, colors2)):
    vals = [data[s][i] for s in samples]
    bars = ax.bar(x + (i - 1.5) * width, vals, width, label=label, color=color)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + (0.02 if v >= 0 else -0.05),
                 f"{v:.2f}", ha="center", fontsize=8, rotation=0)

ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(samples)
ax.set_ylabel("BLIP-ITM margin")
ax.set_title("문제 샘플 3개의 구제 과정: 독립 pooling은 people_dancing을 오히려 악화시켰다", fontsize=12, fontweight="bold")
ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
plt.tight_layout()
plt.savefig(OUT / "fig2_hero_samples_rescue.png")
plt.close()


# ---------------------------------------------------------------------------
# Fig 3. 행동 변경 vs 속성 변경 오답 - CLIP이 어느 쪽을 더 못 구분하는가 (3단계)
# ---------------------------------------------------------------------------
groups = ["원본 코사인", "CSLS 보정"]
action_vals = [0.0198, 0.0398]
attribute_vals = [0.0408, 0.0749]

x = np.arange(len(groups))
width = 0.32
fig, ax = plt.subplots(figsize=(7, 5.5))
b1 = ax.bar(x - width / 2, action_vals, width, label="행동(action) 변경 오답", color=C_RED)
b2 = ax.bar(x + width / 2, attribute_vals, width, label="속성(attribute) 변경 오답", color=C_BLUE)
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.003, f"{b.get_height():.3f}",
                 ha="center", fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels(groups)
ax.set_ylabel("평균 margin (8개 샘플)")
ax.set_title("CLIP은 속성 변경보다 행동 변경 오답을 구분하는 데 더 취약하다", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(OUT / "fig3_action_vs_attribute.png")
plt.close()


# ---------------------------------------------------------------------------
# Fig 4. 6단계: 오답 없이 좋은 프레임을 고르는 실전 방법들, 오라클 대비 도달률(%)
# ---------------------------------------------------------------------------
methods = ["단일\n(중간1장)", "선명도\n필터링", "정답점수만\n(max)", "평균\npooling", "오라클\n(오답 활용)"]
means = [0.5741, 0.4601, 0.7454, 0.5025, 0.8239]
pct_of_oracle = [v / means[-1] * 100 for v in means]
colors4 = [C_GRAY, C_RED, C_PURPLE, C_ORANGE, C_GREEN]

fig, ax = plt.subplots(figsize=(9, 5.5))
bars = ax.bar(methods, means, color=colors4, width=0.55)
for b, v, p in zip(bars, means, pct_of_oracle):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}\n({p:.0f}%)", ha="center", fontsize=9.5)
ax.axhline(means[-1], color=C_GREEN, linestyle="--", linewidth=1, alpha=0.6)
ax.set_ylabel("평균 BLIP-ITM margin (8개 샘플)")
ax.set_title("실전 프레임 선택 방법 비교: '정답점수만으로 max 선택'이 오답 없이도 오라클의 90.5%에 도달", fontsize=11.5, fontweight="bold")
ax.set_ylim(0, max(means) * 1.25)
plt.tight_layout()
plt.savefig(OUT / "fig4_practical_methods.png")
plt.close()

print("차트 4개 생성 완료:")
for p in sorted(OUT.glob("*.png")):
    print(" -", p)
