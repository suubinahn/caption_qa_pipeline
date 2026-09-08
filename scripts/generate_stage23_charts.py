"""
2~3단계(검색+검증 통합) 리포트에 들어갈 차트 생성.
results/images/fig1~5는 1단계용이고, 이 스크립트는 fig6부터 이어서 생성한다.
스타일(폰트/색상/dpi)은 scripts/generate_report_charts.py와 동일하게 맞춘다.
모든 수치는 final_report.md에 이미 기록된 실제 실험 결과에서 그대로 가져온다.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "results" / "images"
OUT.mkdir(parents=True, exist_ok=True)

_NOTO_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(_NOTO_PATH)
_KO_FONT_NAME = fm.FontProperties(fname=_NOTO_PATH).get_name()

plt.rcParams.update({
    "font.family": _KO_FONT_NAME,
    "axes.unicode_minus": False,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "figure.dpi": 150,
})

C_BLUE = "#3B82C4"
C_ORANGE = "#E08E45"
C_GREEN = "#4E9A6A"
C_RED = "#C0504D"
C_GRAY = "#8A8A8A"
C_PURPLE = "#8064A2"

QUERY_LABELS = ["우회전", "좌회전", "정지", "터널", "다리", "야간", "비", "눈", "고속도로"]

# ---------------------------------------------------------------------------
# Fig 6. CSLS 보정 전/후 (텍스트-텍스트 검색, leave-one-out 검증)
# ---------------------------------------------------------------------------
raw = [20.0, 25.0, 9.1, 81.2, 12.5, 33.3, 42.9, 25.0, 53.7]
csls = [26.7, 37.5, 18.2, 81.2, 25.0, 58.3, 48.6, 37.5, 58.5]

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(QUERY_LABELS))
w = 0.35
ax.bar(x - w/2, raw, w, label="원본 코사인", color=C_GRAY)
ax.bar(x + w/2, csls, w, label="CSLS 보정", color=C_BLUE)
ax.set_xticks(x)
ax.set_xticklabels(QUERY_LABELS)
ax.set_ylabel("Precision@K (%)")
ax.set_title("①검색 - CSLS 허브 보정 전/후 (9개 검색어, leave-one-out)")
ax.legend()
for i, (r, c) in enumerate(zip(raw, csls)):
    if c > r:
        ax.annotate("", xy=(i+w/2, c+1), xytext=(i-w/2, r+1),
                     arrowprops=dict(arrowstyle="->", color=C_GREEN, alpha=0.5))
fig.tight_layout()
fig.savefig(OUT / "fig6_csls_before_after.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 7. 100클립 vs 740클립 lift(정밀도/랜덤베이스라인) 비교
# ---------------------------------------------------------------------------
lift100 = [1.78, 4.69, 1.65, 5.08, 3.12, 2.43, 1.39, 4.69, 1.43]
lift740 = [6.65, 22.73, 1.49, 31.27, 11.36, 5.22, 1.99, 11.36, 1.62]

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - w/2, lift100, w, label="100클립 표본", color=C_ORANGE)
ax.bar(x + w/2, lift740, w, label="740클립 전체", color=C_PURPLE)
ax.axhline(1.0, color=C_RED, linestyle="--", linewidth=1, alpha=0.6, label="랜덤 기준선(1.0x)")
ax.set_xticks(x)
ax.set_xticklabels(QUERY_LABELS)
ax.set_ylabel("Lift (precision / 랜덤 베이스라인)")
ax.set_yscale("log")
ax.set_title("①검색 - 100클립 표본 vs 740클립 전체, 랜덤 대비 lift")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "fig7_scale_lift_comparison.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 8. 컷오프(임계값) calibration/holdout 성능 - ①검색 + ②캡션-영상 검증
# ---------------------------------------------------------------------------
labels8 = ["①터널\n(검색)", "①비\n(검색)", "②motion\n(검증)", "②weather\n(검증)", "②time_of_day\n(검증)", "②통합\n(fallback)"]
holdout_acc = [96.0, 79.5, 68.5, 71.5, 97.0, 73.7]
holdout_prec = [100.0, 68.3, 66.7, 65.5, 97.0, 67.8]
colors8 = [C_BLUE, C_BLUE, C_ORANGE, C_ORANGE, C_ORANGE, C_GRAY]

fig, ax = plt.subplots(figsize=(9, 5))
xb = np.arange(len(labels8))
ax.bar(xb - w/2, holdout_acc, w, label="holdout accuracy", color=colors8)
ax.bar(xb + w/2, holdout_prec, w, label="holdout precision", color=colors8, alpha=0.55)
ax.axhline(50, color=C_RED, linestyle="--", linewidth=1, alpha=0.5, label="반반(50%) 기준")
ax.set_xticks(xb)
ax.set_xticklabels(labels8)
ax.set_ylabel("%")
ax.set_title("컷오프 calibration/holdout 성능 (①검색 vs ②캡션-영상 검증)")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "fig8_cutoff_holdout_performance.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 9. match_score 참/거짓 margin 검증 (실제 회사 영상 60클립)
# ---------------------------------------------------------------------------
fields9 = ["motion", "weather", "time_of_day"]
margins9 = [0.153, 0.083, 0.140]
pos_rate9 = [80.0, 80.0, 100.0]

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
axes[0].bar(fields9, margins9, color=[C_BLUE, C_ORANGE, C_GREEN])
axes[0].set_ylabel("평균 margin (참-거짓)")
axes[0].set_title("참/거짓 문장 점수 차이")
axes[0].axhline(0, color=C_GRAY, linewidth=0.8)

axes[1].bar(fields9, pos_rate9, color=[C_BLUE, C_ORANGE, C_GREEN])
axes[1].axhline(50, color=C_RED, linestyle="--", linewidth=1, alpha=0.6, label="랜덤(50%)")
axes[1].set_ylabel("%")
axes[1].set_title("참 > 거짓인 쌍의 비율")
axes[1].legend(fontsize=9)
axes[1].set_ylim(0, 105)

fig.suptitle("②match_score 검증 - 참/거짓 합성 문장 60쌍 (전부 p<0.005로 유의)")
fig.tight_layout()
fig.savefig(OUT / "fig9_match_score_margin_validation.png")
plt.close(fig)

print("fig6~9 저장 완료:", OUT)
