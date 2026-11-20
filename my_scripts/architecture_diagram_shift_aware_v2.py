"""Architecture diagram of the redesigned classifier , version 2 (shift_aware_classifier_v2_english.py:
PCA 128 + AE_1024 verdict block, output margin as error detector).
Outputs: results/architecture_shift_aware_classifier_v2.png and .pdf"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
TINT = {BLUE: "#e6f0fb", ORANGE: "#fdebe3", AQUA: "#e2f5ee", None: "#f1f0ec"}
INK, INK2, MUTED, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK})
fig, ax = plt.subplots(figsize=(15, 7.6), facecolor=SURF); ax.set_facecolor(SURF)
ax.set_xlim(0, 124); ax.set_ylim(-7, 66); ax.axis("off")

def box(x, y, w, h, title, sub="", c=None, bold=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.1", fc=TINT[c], ec=c or AXIS, lw=1.6))
    if sub:
        ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=10.5, fontweight="bold" if bold else "normal", color=INK)
        ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center", fontsize=8.8, color=INK2, linespacing=1.25)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=10.5, fontweight="bold" if bold else "normal", color=INK, linespacing=1.3)
def arrow(x0, y0, x1, y1, c=MUTED, rad=0.0):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=c, lw=1.5, shrinkA=0, shrinkB=0, mutation_scale=13,
                                                        connectionstyle=f"arc3,rad={rad}"))

H = 9.5; XA, WA, XB, WB, XC, WC = 1, 17, 25, 23, 55, 18
Y = {"claim": 49, "trace": 36.5, "vote": 21, "gate": 5}
# column headers
for x, w, t in [(XA, WA, "1  Inputs"), (XB, WB, "2  Extract"), (XC, WC, "3  Compress / combine"), (80, 11, "4  Join"), (97, 26, "5  Classify")]:
    ax.text(x, 62.5, t, fontsize=10, fontweight="bold", color=MUTED, ha="left")
ax.plot([1, 123], [61, 61], color=AXIS, lw=0.8)

# inputs
box(XA, Y["claim"], WA, H, "Claim", "text")
box(XA, Y["trace"], WA, H, "Reasoning traces", "up to 20 texts")
box(XA, Y["vote"], WA, H, "Trace verdicts", "one label per trace")
box(XA, Y["gate"], WA, H, "Evidence", "retrieved snippets")
# extract
box(XB, Y["claim"], WB, H, "Qwen3-Embedding-8B", "frozen · claim vector, 4096-d", AQUA)
box(XB, Y["trace"], WB, H, "Qwen3-Embedding-8B", "frozen · mean of trace vectors, 4096-d", AQUA)
box(XB, Y["vote"], WB, H, "Vote features, 5-d", "3 vote shares · entropy · top share", BLUE)
box(XB, Y["gate"], WB, H, "Evidence length", "z = standardised log(word count)", ORANGE)
# compress / combine
box(XC, Y["claim"], WC, H, "PCA", "4096-d → 128-d", AQUA)
box(XC, Y["trace"], WC, H, "PCA", "4096-d → 128-d", AQUA)
box(XC, Y["vote"], WC, H, "Autoencoder (AE_1024)", "5-d → 1024-d code", BLUE)
box(XC, Y["gate"], WC, H, "Gate, 5-d", "z   and   z × (shares, entropy)", ORANGE)
# join + classify
box(80, Y["gate"], 11, Y["claim"] + H - Y["gate"], "Concatenate\n\n1,285\nfeatures\n\nthen\nstandardise")
box(97, 27, 15, 14, "Logistic\nregression", "", None)
ax.text(104.5, 42.2, "class-balanced · strong L2", ha="center", va="bottom", fontsize=8.8, color=INK2)
box(97, 5, 26.8, H, "Margin = p_top − p_second", "error detector: low margin flags a doubtful verdict", None)
arrow(104.5, 27, 104.5, 5 + H)
ax.text(105.8, 20.8, "class probabilities", fontsize=8.8, color=INK2, ha="left", va="center", style="italic")
box(114.5, 27, 9.3, 14, "False\nTrue\nConflicting", "", None, bold=False)
ax.text(119.15, 42.2, "Verdict", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color=INK2)

# arrows
for k in Y: arrow(XA + WA, Y[k] + H / 2, XB, Y[k] + H / 2)
for k, c in [("claim", AQUA), ("trace", AQUA), ("gate", ORANGE)]:
    arrow(XB + WB, Y[k] + H / 2, XC, Y[k] + H / 2, c); arrow(XC + WC, Y[k] + H / 2, 80, Y[k] + H / 2, c)
arrow(XB + WB, Y["vote"] + H / 2, XC, Y["vote"] + H / 2, BLUE); arrow(XC + WC, Y["vote"] + H / 2, 80, Y["vote"] + H / 2, BLUE)                       # votes go straight to the join
arrow(XB + WB - 5, Y["vote"], XC + 4, Y["gate"] + H, BLUE, rad=-0.18)                # votes also feed the gate
ax.text(60.5, 17.2, "votes are multiplied by z", fontsize=8.8, color=INK2, ha="left", va="center", style="italic")
for x, w, t in [(73.5, 0, "64"), (73.5, 0, "64")]: pass
for k, n in [("claim", "128"), ("trace", "128"), ("vote", "1024"), ("gate", "5")]:
    ax.text(76.5, Y[k] + H / 2 + 1.1, n, fontsize=8.8, color=MUTED, ha="center", va="bottom")
arrow(91, 34, 97, 34); arrow(112, 34, 114.5, 34)

# legend: what each part is for
for i, (c, head, txt) in enumerate([(AQUA, "Content", "what the claim and the traces say"), (BLUE, "Votes", "what the traces conclude"),
                                    (ORANGE, "Gate", "how far to trust the votes: longer evidence, more reliable traces")]):
    x = [1, 40, 72][i]
    ax.add_patch(FancyBboxPatch((x, -4.6), 2.2, 2.2, boxstyle="round,pad=0,rounding_size=0.4", fc=TINT[c], ec=c, lw=1.6))
    ax.text(x + 3.4, -3.5, head, fontsize=9.5, fontweight="bold", color=INK, va="center")
    ax.text(x + 3.4 + {0: 7.6, 1: 5.8, 2: 5.0}[i], -3.5, txt, fontsize=9.5, color=INK2, va="center")
fig.suptitle("Shift-aware verdict classifier, version 2: the votes decide, the content corrects, evidence length sets the trust in the votes",
             x=0.012, y=0.975, ha="left", fontsize=13, fontweight="bold", color=INK)
fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.02)
for ext in ["png", "pdf"]:
    out = os.path.join(P, "results", f"architecture_shift_aware_classifier_v2.{ext}"); fig.savefig(out, dpi=170, facecolor=SURF); print(out)
