"""K-curve with the version-2 classifier (PCA 128 + AE_1024 verdict block + evidence-length gate), English.

For every K = 1..20 and two trace orders everything that depends on the traces is recomputed from the K selected traces
and the model is refitted: vote features -> autoencoder (retrained) -> 1024-d code, mean trace vector, gate.
  given  first K traces in dataset order
  pn     top K traces of the p/n ranking ('pn_selected'; out-of-fold for train and validation)
The PCA bases are those of shift_aware_classifier_v2_english.py, so K = 20 reproduces its 'verdict = AE_1024' row (seed 0).
The margin of the output (p_top - p_second) is recorded as an error detector (AUROC) for every K.
Outputs: results/k_curve_shift_aware_v2_english.csv and .png
"""
import json, os, time, warnings, numpy as np, pandas as pd, torch, torch.nn as nn, matplotlib
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMB = os.path.join(P, "my data", "qwen3 english embeddings")
RK = os.path.join(P, "my data", "saving Ranked English Traces outputs", "en_trace_rankings", "rankings", "pn_selected")
POOLED = os.path.join(P, "my data", "pooled_qwen3emb8b_en_v1.npz")
TRACE128 = os.path.join(P, "my data", "trace_pca128_v2_qwen3emb8b_en_v1.npz")
OUT = os.path.join(P, "results", "k_curve_shift_aware_v2_english")
C = ["False", "True", "Conflicting"]; L = {c: i for i, c in enumerate(C)}
T = 20; DIM = 128; PCA_FIT = 512; EXP_DIM = 1024; C_GRID = [0.001, 0.01, 0.1]; SEED = 0; B = 2000; rng = np.random.default_rng(0)
def log(*a): print(*a, flush=True)
def f1s(y, p):
    cm = np.bincount(y * 3 + p, minlength=9).reshape(3, 3); tp = np.diag(cm); d = cm.sum(0) + cm.sum(1)
    return np.where(d > 0, 2 * tp / np.maximum(d, 1), 0.0)

# ---------------- data ----------------
D = {}
for split, fn in [("train", "train"), ("val", "validation"), ("test", "test")]:
    d = json.load(open(os.path.join(P, "task 2", "data", "English", f"{fn}.json"), encoding="utf-8"))
    rk = json.load(open(os.path.join(RK, f"{split}.json"), encoding="utf-8")); N = len(d)
    verd = np.full((N, T), -1); og = np.full((N, T), -1); op = np.full((N, T), -1)
    for r, (it, k) in enumerate(zip(d, rk)):
        assert k["claim"] == it["claim"]
        n = min(len(it["Verdict_list"]), len([t for t in it["Reasoning_traces"] if t.strip()]), T)
        verd[r, :n] = [L.get(v, -1) for v in it["Verdict_list"][:n]]; og[r, :n] = np.arange(n)
        o = [j for j in k["order"] if j < n]; op[r, :len(o)] = o
    D[split] = {"ids": [it["query_id"] for it in d] if split == "test" else list(range(N)), "verd": verd, "order": {"given": og, "pn": op},
                "y": np.array([L[it.get("label", it.get("Label"))] for it in d]),
                "evw": np.array([sum(len(e.split()) for e in it["evidences"]) for it in d])}
with np.load(POOLED) as z:
    pca_c = PCA(PCA_FIT, random_state=0).fit(z["train_claim"].astype(np.float32)); pca_t = PCA(PCA_FIT, random_state=0).fit(z["train_tmean"].astype(np.float32))
    for s in D: D[s]["claim"] = pca_c.transform(z[f"{s}_claim"].astype(np.float32))[:, :DIM]
if not os.path.exists(TRACE128):                     # every single trace projected on the first 128 trace components, once
    out = {}; W = pca_t.components_[:DIM].T.astype(np.float32); mu = pca_t.mean_.astype(np.float32)
    for s in D:
        a = np.zeros((len(D[s]["y"]), T, DIM), np.float32)
        with np.load(os.path.join(EMB, f"{s}_qwen3emb8b_en_v1.npz"), allow_pickle=False) as z:
            for r, i in enumerate(D[s]["ids"]):
                x = z[f"id_{i}"][1:1 + T].astype(np.float32)
                if len(x): a[r, :len(x)] = (x - mu) @ W
        out[s] = a; log("projected", s)
    np.savez(TRACE128, **out)
with np.load(TRACE128) as z:
    for s in D: D[s]["tr"] = z[s]
lw_mu, lw_sd = np.log1p(D["train"]["evw"]).mean(), np.log1p(D["train"]["evw"]).std()
for s in D: D[s]["z"] = ((np.log1p(D[s]["evw"]) - lw_mu) / lw_sd)[:, None]

def autoencode(V):                                   # as in the v2 script: 5-d verdict vector -> 1024-d code
    torch.manual_seed(SEED); sc = StandardScaler().fit(V["train"]); X = {s: torch.tensor(sc.transform(V[s]), dtype=torch.float32) for s in V}; d = X["train"].shape[1]
    enc = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, EXP_DIM), nn.ReLU()); dec = nn.Sequential(nn.Linear(EXP_DIM, 256), nn.ReLU(), nn.Linear(256, d))
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-3, weight_decay=1e-4)
    for _ in range(300):
        opt.zero_grad(); ((dec(enc(X["train"])) - X["train"]) ** 2).mean().backward(); opt.step()
    with torch.no_grad(): return {s: enc(X[s]).numpy() for s in V}

def select(s, order, K):
    idx = D[s]["order"][order][:, :K]; m = idx >= 0; safe = np.maximum(idx, 0)
    v = np.where(m, np.take_along_axis(D[s]["verd"], safe, 1), -1)
    cn = np.stack([(v == c).sum(1) for c in range(3)], 1).astype(float); sh = cn / np.maximum(cn.sum(1, keepdims=True), 1)
    votes = np.concatenate([sh, -(sh * np.log(sh + 1e-9)).sum(1, keepdims=True), sh.max(1, keepdims=True)], 1)
    tm = (np.take_along_axis(D[s]["tr"], safe[:, :, None], 1) * m[:, :, None]).sum(1) / np.maximum(m.sum(1, keepdims=True), 1)
    return votes, tm, cn.argmax(1)

# ---------------- K sweep ----------------
yv, yt = D["val"]["y"], D["test"]["y"]; idx = rng.integers(0, len(yt), (B, len(yt))); PRED, VOTE, ROWS = {}, {}, []
for order in ["given", "pn"]:
    for K in range(1, T + 1):
        t0 = time.time(); S = {s: select(s, order, K) for s in D}; E = autoencode({s: S[s][0] for s in D})
        X = {s: np.concatenate([E[s], D[s]["z"], S[s][0][:, :4] * D[s]["z"], D[s]["claim"], S[s][1]], 1) for s in D}
        sc = StandardScaler().fit(X["train"]); X = {s: sc.transform(X[s]) for s in X}; best = None; tests = []
        for Cc in C_GRID:
            lr = LogisticRegression(C=Cc, max_iter=3000, class_weight="balanced").fit(X["train"], D["train"]["y"])
            pv, pt = lr.predict_proba(X["val"]), lr.predict_proba(X["test"]); vf = f1s(yv, pv.argmax(1)).mean(); tests.append(f1s(yt, pt.argmax(1)).mean())
            if best is None or vf > best[0]: best = (vf, pt, Cc)
        vf, pt, Cc = best; p = pt.argmax(1); ft = f1s(yt, p); mt = np.sort(pt, 1); mt = mt[:, -1] - mt[:, -2]
        PRED[order, K], VOTE[order, K] = p, S["test"][2]
        ROWS.append({"order": order, "K": K, "C": Cc, "val_f1": vf, "test_f1": ft.mean(), "test_f1_min_over_C": min(tests), "test_f1_max_over_C": max(tests),
                     "test_acc": float(np.mean(p == yt)), "f1_False": ft[0], "f1_True": ft[1], "f1_Conflicting": ft[2], "margin_error_auroc": roc_auc_score(p != yt, -mt),
                     "vote_val_f1": f1s(yv, S["val"][2]).mean(), "vote_test_f1": f1s(yt, S["test"][2]).mean()})
        log(f"{order:5s} K {K:2d} | C {Cc:5.3f} | val {vf:.4f} | test {ft.mean():.4f} | vote test {ROWS[-1]['vote_test_f1']:.4f} | AUROC {ROWS[-1]['margin_error_auroc']:.3f} | {time.time() - t0:.0f}s")
R = pd.DataFrame(ROWS)
bs = {k: np.array([f1s(yt[i], p[i]).mean() for i in idx]) for k, p in PRED.items()}
bv = {k: np.array([f1s(yt[i], p[i]).mean() for i in idx]) for k, p in VOTE.items()}
q = lambda a: (np.quantile(a, .025), np.quantile(a, .975))
R["ci_lo"] = [q(bs[o, k])[0] for o, k in zip(R.order, R.K)]; R["ci_hi"] = [q(bs[o, k])[1] for o, k in zip(R.order, R.K)]
for name, ref in [("vs_K20", lambda o, k: bs[o, T]), ("vs_vote_sameK", lambda o, k: bv[o, k]), ("vs_vote_all20", lambda o, k: bv["given", T]), ("pn_minus_given", lambda o, k: bs["given", k])]:
    dd = [bs[o, k] - ref(o, k) for o, k in zip(R.order, R.K)]
    R[f"d_{name}"] = [np.mean(d) for d in dd]; R[f"d_{name}_lo"] = [q(d)[0] for d in dd]; R[f"d_{name}_hi"] = [q(d)[1] for d in dd]
R.to_csv(OUT + ".csv", index=False)

# ---------------- figure ----------------
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.6), facecolor=SURF)
for ax, col, vcol, title, kmax in [(axes[0], "test_f1", "vote_test_f1", "Test: 2,558 claims", 20), (axes[1], "val_f1", "vote_val_f1", "Validation: 1,600 claims (15 traces each)", 15)]:
    ax.set_facecolor(SURF); ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)
    lo, hi = 1, 0
    for order, c, lab in [("given", BLUE, "dataset order (first K)"), ("pn", ORANGE, "p/n ranking (top K)")]:
        a = R[(R.order == order) & (R.K <= kmax)].sort_values("K")
        if col == "test_f1": ax.fill_between(a.K, a.ci_lo, a.ci_hi, color=c, alpha=0.10, lw=0); lo, hi = min(lo, a.ci_lo.min()), max(hi, a.ci_hi.max())
        ax.plot(a.K, a[col], color=c, lw=2.2, marker="o", ms=4.5, label=f"Classifier v2, {lab}")
        ax.plot(a.K, a[vcol], color=c, lw=1.4, ls=(0, (3, 2)), label=f"Majority vote, {lab}")
        lo, hi = min(lo, a[col].min(), a[vcol].min()), max(hi, a[col].max(), a[vcol].max())
    ax.set_ylim(np.floor((lo - 0.008) * 50) / 50, np.ceil((hi + 0.008) * 50) / 50)
    ax.set_xlim(0.4, kmax + 0.6); ax.set_xticks([k for k in [1, 3, 5, 10, 15, 20] if k <= kmax])
    ax.set_xlabel("Number of traces given to the classifier (K)"); ax.set_ylabel("Macro-F1")
    ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold", color=INK, pad=10)
axes[0].legend(frameon=False, fontsize=8.8, labelcolor=INK2, loc="lower right", ncol=2)
fig.suptitle("K-curve with classifier version 2: Macro-F1 against the number of traces", x=0.012, ha="left", fontsize=12.5, fontweight="bold", color=INK)
fig.text(0.012, 0.012, "English. Classifier v2 = AE_1024 vote code + evidence-length gate + PCA-128 of claim and mean trace vector, logistic regression, refitted for every K. "
         "Band on test = 95% bootstrap interval.", fontsize=8.5, color=MUTED)
fig.tight_layout(rect=(0, 0.035, 1, 0.94), w_pad=3)
fig.savefig(OUT + ".png", dpi=160, facecolor=SURF); log(OUT + ".png")
