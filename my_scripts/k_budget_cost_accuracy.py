"""Step 1 of the trace-budget study: cost-accuracy curve on the provided English traces.

Traces are always read in dataset order (first K). No shuffling anywhere.
Compared stopping rules (the verdict is always the plurality of the traces read so far):
  fixed      read exactly K traces, K = 1..20
  beta       Adaptive-Consistency (Aggarwal et al., EMNLP 2023): stop when P(leader beats runner-up) > threshold
  lead       stop when the leader is m votes ahead of the runner-up
  lossless   stop when the remaining traces can no longer change the 20-trace verdict
  oracle     stop at the first K after which the vote never changes again (needs all 20 traces: a lower bound on cost)
Outputs: results/k_budget_cost_accuracy_english.csv and .png
"""
import json, os, numpy as np, pandas as pd, matplotlib
from scipy.special import betainc
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = ["False", "True", "Conflicting"]; L = {c: i for i, c in enumerate(C)}
T = 20; B = 2000; rng = np.random.default_rng(0)
BETA_T = [0.80, 0.90, 0.95, 0.97, 0.99, 0.995, 0.999]
LEAD_M = [2, 3, 4, 5, 6, 8, 10]

def mf1(y, p):
    cm = np.bincount(y * 3 + p, minlength=9).reshape(3, 3); tp = np.diag(cm); d = cm.sum(0) + cm.sum(1)
    return float(np.mean(np.where(d > 0, 2 * tp / np.maximum(d, 1), 0.0)))

def load(split):
    d = json.load(open(os.path.join(P, "task 2", "data", "English", f"{split}.json"), encoding="utf-8"))
    keep = [it for it in d if len(it["Verdict_list"]) == T and all(v in L for v in it["Verdict_list"])]
    return np.array([[L[v] for v in it["Verdict_list"]] for it in keep]), np.array([L[it.get("label", it.get("Label"))] for it in keep])

def rules(V):
    """-> list of (method, param, traces_used (N,), prediction (N,))"""
    N = len(V); cnt = np.cumsum(V[:, :, None] == np.arange(3), 1)          # (N,20,3) votes after k traces
    lead = cnt.argmax(2); s = np.sort(cnt, 2); a, b = s[:, :, 2], s[:, :, 1]   # leader / runner-up counts
    full = lead[:, -1]; rows = np.arange(N); out = []
    def stop(cond):                                                        # first k where cond holds, else all 20
        k = np.where(cond.any(1), cond.argmax(1), T - 1); return k + 1, lead[rows, k]
    for K in range(1, T + 1): out.append(("fixed", K, np.full(N, K), lead[:, K - 1]))
    conf = 1.0 - betainc(a + 1.0, b + 1.0, 0.5)                            # P(p_leader > p_runner-up), Beta posterior
    for t in BETA_T: out.append(("beta", t, *stop(conf > t)))
    for m in LEAD_M: out.append(("lead", m, *stop((a - b) >= m)))
    used, pred = stop((a - b) > (T - 1 - np.arange(T))[None, :]); assert (pred == full).all()
    out.append(("lossless", 0, used, pred))
    need = np.where(lead != full[:, None], np.arange(1, T + 1), 0).max(1) + 1
    out.append(("oracle", 0, need, full))
    return out

RES = []
for split in ["test", "train"]:
    V, y = load(split); N = len(y); R = rules(V); ref = R[T - 1][3]      # fixed K = 20
    idx = rng.integers(0, N, (B, N))
    refb = np.array([mf1(y[i], ref[i]) for i in idx])
    for method, param, used, pred in R:
        fb = np.array([mf1(y[i], pred[i]) for i in idx]); db = fb - refb
        RES.append({"split": split, "n_claims": N, "method": method, "param": param, "mean_traces": used.mean(),
                    "saved_pct": 100 * (1 - used.mean() / T), "macro_f1": mf1(y, pred), "f1_lo": np.quantile(fb, .025), "f1_hi": np.quantile(fb, .975),
                    "diff_vs_20": mf1(y, pred) - mf1(y, ref), "diff_lo": np.quantile(db, .025), "diff_hi": np.quantile(db, .975),
                    "agree_with_20": float(np.mean(pred == ref))})
R = pd.DataFrame(RES); R.to_csv(os.path.join(P, "results", "k_budget_cost_accuracy_english.csv"), index=False)
pd.set_option("display.width", 250)
print(R.round(4).to_string(index=False))

# ---------------- figure ----------------
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.6), facecolor=SURF)
for ax, split in zip(axes, ["test", "train"]):
    r = R[R.split == split]; f = r[r.method == "fixed"].sort_values("mean_traces")
    ax.set_facecolor(SURF); ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(length=0, which="both")
    ax.fill_between(f.mean_traces, f.f1_lo, f.f1_hi, color=BLUE, alpha=0.12, lw=0)
    ax.plot(f.mean_traces, f.macro_f1, color=BLUE, lw=2, marker="o", ms=4, label="Fixed K (band = 95% bootstrap interval)")
    for m, c, mk, lab in [("beta", ORANGE, "D", "Adaptive-Consistency (Beta stopping)"), ("lead", AQUA, "s", "Stop when leader is m votes ahead")]:
        a = r[r.method == m].sort_values("mean_traces")
        ax.plot(a.mean_traces, a.macro_f1, color=c, lw=2, marker=mk, ms=5, label=lab)
    lo, ora = r[r.method == "lossless"].iloc[0], r[r.method == "oracle"].iloc[0]
    ax.scatter([lo.mean_traces], [lo.macro_f1], s=70, marker="^", color=INK2, zorder=5, label="Lossless early stop (same verdict as 20 traces)")
    ax.scatter([ora.mean_traces], [ora.macro_f1], s=90, marker="*", color=INK, zorder=5, label="Oracle budget (lower bound on cost)")
    ax.set_xscale("log"); ax.set_xticks([1, 2, 3, 5, 10, 20]); ax.set_xticklabels(["1", "2", "3", "5", "10", "20"]); ax.minorticks_off(); ax.set_xlim(0.9, 22)
    ax.set_ylim(np.floor((f.f1_lo.min() - 0.005) * 100) / 100, np.ceil((f.f1_hi.max() + 0.005) * 100) / 100)
    ax.set_xlabel("Average number of traces read per claim (log scale)"); ax.set_ylabel("Macro-F1")
    ax.set_title(f"{split.capitalize()}: {int(r.n_claims.iloc[0]):,} claims", loc="left", fontsize=11.5, fontweight="bold", color=INK, pad=10)
axes[0].legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="lower right")
fig.suptitle("Cost against accuracy: every stopping rule stays inside the uncertainty of reading all 20 traces", x=0.012, ha="left", fontsize=12.5, fontweight="bold", color=INK)
fig.text(0.012, 0.012, "English, CheckThat! 2026 Task 2, claims with 20 traces, traces read in dataset order. Verdict = plurality of the traces read. "
         f"Intervals from {B:,} bootstrap resamples of claims.", fontsize=8.5, color=MUTED)
fig.tight_layout(rect=(0, 0.035, 1, 0.94), w_pad=3)
out = os.path.join(P, "results", "k_budget_cost_accuracy_english.png"); fig.savefig(out, dpi=160, facecolor=SURF); print(out)
