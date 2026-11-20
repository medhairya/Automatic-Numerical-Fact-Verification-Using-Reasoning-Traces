"""Adaptive stopping rules for the trace budget: English, Arabic and Spanish (original and deduplicated training data).

Traces are read in dataset order, one at a time. A stopping rule decides after each trace whether to stop.
Vote family (training-free, so identical for original and dedup data; verdict = plurality of the traces read):
  fixed K        always read K traces
  Beta           Adaptive-Consistency (Aggarwal et al., EMNLP 2023): stop when P(leader beats runner-up) > t
  lead m         stop when the leader is m votes ahead of the runner-up
  lossless       stop when the remaining traces can no longer change the 20-trace vote
  oracle         first K after which the vote never changes (needs all 20: a lower bound on cost)
Classifier family (version-2 classifier refitted for every K on the first K traces; depends on the training data):
  fixed K        classifier on the first K traces
  margin tau     stop at the first K where the classifier margin p_top - p_second >= tau (tau = quantiles of the
                 validation margins, so no test labels are used)
  hybrid t       stop with the Beta rule on the votes, but output the classifier verdict at that K
  oracle         first K after which the classifier verdict never changes
Outputs: results/adaptive_stopping_all_languages.csv and .png
Usage: python adaptive_stopping_all_languages.py [english] [arabic] [spanish]
"""
import json, os, sys, time, warnings, numpy as np, pandas as pd, matplotlib
from scipy.special import betainc
from sklearn.decomposition import PCA
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shift_aware_v2_arabic_spanish import VERD, VL, vote_feats, autoencode, fit_lr, f1s
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = {"english": {"dir": "English", "emb": "qwen3 english embeddings", "npz": "qwen3emb8b_en_v1", "files": ["train.json", "validation.json", "test.json"], "dedup": False},
       "arabic": {"dir": "Arabic", "emb": "qwen3 arabic embeddings", "npz": "qwen3emb8b_ar_v1", "dedup": True,
                  "files": ["clef2026_gpt4_o_mini_train_arabic.json", "clef2026_gpt4_o_mini_val_arabic.json", "clef_arabic_test_gold_labels-final.json"]},
       "spanish": {"dir": "Spanish", "emb": "qwen3 spanish embeddings", "npz": "qwen3emb8b_ar_v1", "dedup": True,   # Spanish files carry the _ar_v1 tag
                   "files": ["spanish_train.json", "spanish_val.json", "clef_spanish_test_final_with_gold_labels.json"]}}
T = 20; DIM = 128; PCA_FIT = 512; B = 2000
BETA_T = [0.8, 0.9, 0.95, 0.99]; LEAD_M = [2, 3, 4, 5]; TAU_Q = [0.2, 0.4, 0.6, 0.8, 0.9]
OUT = os.path.join(P, "results", "adaptive_stopping_all_languages")
def log(*a): print(*a, flush=True)

def load(lang):
    cfg = CFG[lang]; RAW = {}
    for split, fn in zip(["train", "val", "test"], cfg["files"]):
        d = json.load(open(os.path.join(P, "task 2", "data", cfg["dir"], fn), encoding="utf-8")); N = len(d)
        ids = [it.get("query_id", it.get("query_index")) for it in d] if split == "test" else list(range(N))
        cache = os.path.join(P, "my data", f"traces_fp16_{lang}_{split}.npz")
        if not os.path.exists(cache):
            cl = np.zeros((N, 4096), np.float16); tr = np.zeros((N, T, 4096), np.float16); nt = np.zeros(N, np.int64)
            with np.load(os.path.join(P, "my data", cfg["emb"], f"{split}_{cfg['npz']}.npz"), allow_pickle=True) as z:
                for r, i in enumerate(ids):
                    a = z[f"id_{i}"]; n = min(len(a) - 1, T); cl[r] = a[0]; tr[r, :n] = a[1:1 + n]; nt[r] = n
            np.savez(cache, claim=cl, traces=tr, ntr=nt); log(lang, "cached", split)
        with np.load(cache) as z:
            cl, tr = z["claim"].astype(np.float32), z["traces"]; nt = z["ntr"] if "ntr" in z.files else np.full(N, T)
        verd = np.full((N, T), -1); order = np.full((N, T), -1)
        for r, it in enumerate(d):
            n = int(min(nt[r], len(it["Verdict_list"]), T)); verd[r, :n] = [VL.get(x, -1) for x in it["Verdict_list"][:n]]; order[r, :n] = np.arange(n)
        RAW[split] = {"claim_text": [it["claim"].strip() for it in d], "label": [str(it.get("label", it.get("Label"))) for it in d], "verd": verd, "order": order,
                      "evw": np.array([sum(len(str(e).split()) for e in (it.get("evidences") or [])) for it in d]), "claim": cl, "traces": tr}
    keep = {"original": {s: np.arange(len(RAW[s]["label"])) for s in RAW}}
    if cfg["dedup"]:                                   # official test unchanged; train / val without test claims and repeats
        test_keys, seen, kt, kv = set(RAW["test"]["claim_text"]), set(), [], []
        for split, out in [("train", kt), ("val", kv)]:
            for r, c in enumerate(RAW[split]["claim_text"]):
                if c not in test_keys and c not in seen: out.append(r); seen.add(c)
        keep["dedup"] = {"train": np.array(kt), "val": np.array(kv), "test": keep["original"]["test"]}
    return RAW, keep

def classifier_by_K(RAW, keep):
    """Version-2 classifier refitted on the first K traces, K = 1..T. Returns test labels, classes, test/val probabilities per K."""
    D = {s: {k: (v[keep[s]] if isinstance(v, np.ndarray) else [v[i] for i in keep[s]]) for k, v in RAW[s].items()} for s in RAW}
    classes = [c for c in VERD if c in set(D["train"]["label"])]; nc = len(classes); CL = {c: i for i, c in enumerate(classes)}
    y = {s: np.array([CL[l] for l in D[s]["label"]]) for s in D}
    tmean = np.stack([D["train"]["traces"][i, :max(int((D["train"]["order"][i] >= 0).sum()), 1)].astype(np.float32).mean(0) for i in range(len(y["train"]))])
    pc, pt_ = PCA(PCA_FIT, random_state=0).fit(D["train"]["claim"]), PCA(PCA_FIT, random_state=0).fit(tmean); del tmean
    W, mu = pt_.components_[:DIM].T.astype(np.float32), pt_.mean_.astype(np.float32)
    for s in D:
        D[s]["c"] = pc.transform(D[s]["claim"])[:, :DIM].astype(np.float32); N = len(y[s]); D[s]["t"] = np.zeros((N, T, DIM), np.float32)
        for a in range(0, N, 128): D[s]["t"][a:a + 128] = (D[s]["traces"][a:a + 128].astype(np.float32) - mu) @ W
    m_, s_ = np.log1p(D["train"]["evw"]).mean(), max(np.log1p(D["train"]["evw"]).std(), 1e-6)
    for s in D: D[s]["z"] = ((np.log1p(D[s]["evw"]) - m_) / s_)[:, None]
    PT, PV = {}, {}
    for K in range(1, T + 1):
        t0 = time.time(); V, TM = {}, {}
        for s in D:
            idx = D[s]["order"][:, :K]; m = idx >= 0; safe = np.maximum(idx, 0)
            V[s] = vote_feats(np.where(m, np.take_along_axis(D[s]["verd"], safe, 1), -1))
            TM[s] = (np.take_along_axis(D[s]["t"], safe[:, :, None], 1) * m[:, :, None]).sum(1) / np.maximum(m.sum(1, keepdims=True), 1)
        E = autoencode(V, 0)
        X = {s: np.concatenate([E[s], D[s]["z"], V[s][:, :4] * D[s]["z"], D[s]["c"], TM[s]], 1) for s in D}
        vf, PV[K], PT[K], Cc = fit_lr(X, y, nc)
        log(f"    classifier K {K:2d} | val {vf:.4f} | test {f1s(y['test'], PT[K].argmax(1), nc).mean():.4f} | {time.time() - t0:.0f}s")
    return y["test"], classes, PT, PV, len(y["train"])

def first_true(cond):                                  # cond (N,T) -> traces used (1..T) at the first True, else T
    k = np.where(cond.any(1), cond.argmax(1), T - 1); return k + 1

def evaluate(rows, lang, setting, y, nc, idx, family, method, param, used, pred, ref, ref_name, vote20):
    def bs(p): return np.array([f1s(y[i], p[i], nc).mean() for i in idx])
    fb, rb, vb = bs(pred), bs(ref), bs(vote20); q = lambda a: (float(np.quantile(a, .025)), float(np.quantile(a, .975)))
    f = f1s(y, pred, nc).mean()
    rows.append({"language": lang, "setting": setting, "family": family, "method": method, "param": param, "mean_traces": float(used.mean()),
                 "saved_pct": 100 * (1 - used.mean() / T), "macro_f1": f, "ci_lo": q(fb)[0], "ci_hi": q(fb)[1], "reference": ref_name,
                 "diff_vs_ref": f - f1s(y, ref, nc).mean(), "diff_lo": q(fb - rb)[0], "diff_hi": q(fb - rb)[1], "agree_with_ref": float(np.mean(pred == ref)),
                 "diff_vs_vote20": f - f1s(y, vote20, nc).mean(), "dv_lo": q(fb - vb)[0], "dv_hi": q(fb - vb)[1]})

def run(lang, rows):
    RAW, keep = load(lang); rng = np.random.default_rng(0)
    for setting in keep:
        log(f"\n===== {lang} / {setting}")
        y, classes, PT, PV, n_train = classifier_by_K(RAW, keep[setting]); nc = len(classes); cls_v = [VL[c] for c in classes]
        idx = rng.integers(0, len(y), (B, len(y))); V = RAW["test"]["verd"][keep[setting]["test"]]; N = len(y); rws = np.arange(N)
        assert (RAW["test"]["order"] >= 0).all(), "every test claim is expected to have 20 traces"
        # ---- vote family ----
        cnt = np.cumsum(V[:, :, None] == np.array(cls_v)[None, None, :], 1)      # (N,T,nc) votes for the label classes
        lead = cnt.argmax(2); srt = np.sort(cnt, 2); a, b = srt[:, :, -1], srt[:, :, -2]; vote20 = lead[:, -1]
        ev = lambda fam, meth, par, used, pred, ref, rn: evaluate(rows, lang, setting, y, nc, idx, fam, meth, par, used, pred, ref, rn, vote20)
        for K in range(1, T + 1): ev("vote", "fixed K", K, np.full(N, K), lead[:, K - 1], vote20, "vote, 20 traces")
        conf = 1.0 - betainc(a + 1.0, b + 1.0, 0.5)
        for t in BETA_T:
            u = first_true(conf > t); ev("vote", "Beta (Adaptive-Consistency)", t, u, lead[rws, u - 1], vote20, "vote, 20 traces")
        for m in LEAD_M:
            u = first_true((a - b) >= m); ev("vote", "lead m votes", m, u, lead[rws, u - 1], vote20, "vote, 20 traces")
        u = first_true((a - b) > (T - np.arange(1, T + 1))[None, :]); assert (lead[rws, u - 1] == vote20).all()
        ev("vote", "lossless", 0, u, vote20, vote20, "vote, 20 traces")
        need = np.where(lead != vote20[:, None], np.arange(1, T + 1), 0).max(1) + 1; ev("vote", "oracle", 0, need, vote20, vote20, "vote, 20 traces")
        # ---- classifier family ----
        PRED = np.stack([PT[K].argmax(1) for K in range(1, T + 1)], 1); MARG = np.stack([np.diff(np.sort(PT[K], 1)[:, -2:], axis=1)[:, 0] for K in range(1, T + 1)], 1)
        clf20 = PRED[:, -1]
        for K in range(1, T + 1): ev("classifier", "fixed K", K, np.full(N, K), PRED[:, K - 1], clf20, "classifier, 20 traces")
        vm = np.concatenate([np.diff(np.sort(PV[K], 1)[:, -2:], axis=1)[:, 0] for K in range(1, T + 1)])
        for qq in TAU_Q:
            tau = float(np.quantile(vm, qq)); u = first_true(MARG >= tau)
            ev("classifier", "margin stop", round(tau, 3), u, PRED[rws, u - 1], clf20, "classifier, 20 traces")
        for t in BETA_T:
            u = first_true(conf > t); ev("hybrid", "Beta stop, classifier verdict", t, u, PRED[rws, u - 1], clf20, "classifier, 20 traces")
        need = np.where(PRED != clf20[:, None], np.arange(1, T + 1), 0).max(1) + 1; ev("classifier", "oracle", 0, need, clf20, clf20, "classifier, 20 traces")
        for r in rows:
            if r["language"] == lang and r["setting"] == setting: r["n_train"], r["classes"] = n_train, "/".join(classes)
        pd.DataFrame(rows).to_csv(OUT + ".csv", index=False)

def plot(R):
    BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"; INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    panels = [(l, s) for l in ["english", "arabic", "spanish"] for s in ["original", "dedup"] if ((R.language == l) & (R.setting == s)).any()]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.4), facecolor=SURF); axes = axes.ravel()
    for ax, (l, s) in zip(axes, panels):
        r = R[(R.language == l) & (R.setting == s)]; ax.set_facecolor(SURF); ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(length=0, which="both")
        for fam, meth, c, ls, mk, lab in [("vote", "fixed K", BLUE, "-", "o", "Vote, fixed K"), ("vote", "Beta (Adaptive-Consistency)", BLUE, (0, (3, 2)), "D", "Vote, Beta stopping"),
                                          ("classifier", "fixed K", ORANGE, "-", "o", "Classifier v2, fixed K"), ("classifier", "margin stop", ORANGE, (0, (3, 2)), "s", "Classifier v2, margin stopping"),
                                          ("hybrid", "Beta stop, classifier verdict", AQUA, (0, (3, 2)), "D", "Hybrid: Beta stopping, classifier verdict")]:
            a = r[(r.family == fam) & (r.method == meth)].sort_values("mean_traces")
            ax.plot(a.mean_traces, a.macro_f1, color=c, ls=ls, lw=2 if ls == "-" else 1.6, marker=mk, ms=3.5 if ls == "-" else 5, label=lab)
        for fam, c, lab in [("vote", BLUE, "Oracle budget, vote"), ("classifier", ORANGE, "Oracle budget, classifier")]:
            o = r[(r.family == fam) & (r.method == "oracle")].iloc[0]; ax.scatter([o.mean_traces], [o.macro_f1], marker="*", s=110, color=c, edgecolor=INK, lw=0.6, zorder=5, label=lab)
        ax.set_xscale("log"); ax.set_xticks([1, 2, 3, 5, 10, 20]); ax.set_xticklabels(["1", "2", "3", "5", "10", "20"]); ax.minorticks_off(); ax.set_xlim(0.9, 22)
        lo, hi = r.macro_f1.min(), r.macro_f1.max(); pad = max(0.01, (hi - lo) * 0.12); ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel("Average traces read per claim (log scale)"); ax.set_ylabel("Macro-F1")
        ttl = {"original": "original train / val", "dedup": "deduplicated train / val"}[s]
        ax.set_title(f"{l.capitalize()}, {ttl}" if CFG[l]["dedup"] else f"{l.capitalize()}", loc="left", fontsize=11, fontweight="bold", color=INK, pad=8)
    for ax in axes[len(panels):]: ax.axis("off")
    h, lb = axes[0].get_legend_handles_labels(); axes[-1].legend(h, lb, frameon=False, fontsize=10, labelcolor=INK2, loc="center left")
    fig.suptitle("Adaptive stopping: Macro-F1 against the average number of traces read (test)", x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.01, 0.01, "Traces read in dataset order. Vote rules need no training, so they are identical for original and deduplicated data. "
             "Classifier v2 is refitted for every K on the first K traces.", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.025, 1, 0.95), h_pad=2.5, w_pad=2.5); fig.savefig(OUT + ".png", dpi=150, facecolor=SURF); log(OUT + ".png")

if __name__ == "__main__":
    rows = []
    for lang in (sys.argv[1:] or ["english", "arabic", "spanish"]): run(lang, rows)
    R = pd.DataFrame(rows); R.to_csv(OUT + ".csv", index=False); plot(R)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    keep = (R.method != "fixed K") | R.param.isin([1, 3, 5, 10, 20])
    print(R[keep][["language", "setting", "family", "method", "param", "mean_traces", "macro_f1", "diff_vs_ref", "diff_lo", "diff_hi", "agree_with_ref",
                   "diff_vs_vote20", "dv_lo", "dv_hi"]].round(4).to_string(index=False))
