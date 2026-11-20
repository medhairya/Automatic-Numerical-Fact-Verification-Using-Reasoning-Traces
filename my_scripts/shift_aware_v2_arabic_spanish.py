"""Version-2 classifier on Arabic and Spanish (CheckThat! 2026 Task 2): main results + K-curve.

Two data settings per language:
  original  train / validation as released (test claims also occur in train / validation: leakage, as in the paper table)
  dedup     official test unchanged; train / validation without any claim that is in test and without repeated claims
Models (class-balanced logistic regression, C chosen on validation, classes = labels present in train):
  votes     5 vote features
  v1        votes + evidence-length gate + PCA-64 of claim and mean trace vector
  v2        AE_1024 code of the votes + gate + PCA-128 of claim and mean trace vector (3 autoencoder seeds)
K-curve (v2, seed 0): K = 1..20 traces, dataset order (first K) and a p/n ranking (top K). The p/n ranking is computed here
from a vote-count model (out-of-fold on train): verdict groups ordered by p(class) / group size, dataset order inside a group.
Outputs in results/: shift_aware_v2_{lang}.csv, shift_aware_v2_{lang}_{setting}_test_predictions.csv,
                     k_curve_shift_aware_v2_{lang}.csv and .png
Usage: python shift_aware_v2_arabic_spanish.py [arabic] [spanish]
"""
import json, os, sys, time, warnings, numpy as np, pandas as pd, torch, torch.nn as nn, matplotlib
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = {"arabic": {"dir": "Arabic", "emb": "qwen3 arabic embeddings",
                  "files": ["clef2026_gpt4_o_mini_train_arabic.json", "clef2026_gpt4_o_mini_val_arabic.json", "clef_arabic_test_gold_labels-final.json"]},
       "spanish": {"dir": "Spanish", "emb": "qwen3 spanish embeddings",
                   "files": ["spanish_train.json", "spanish_val.json", "clef_spanish_test_final_with_gold_labels.json"]}}
VERD = ["False", "True", "Conflicting"]; VL = {c: i for i, c in enumerate(VERD)}
T = 20; PCA_FIT = 512; EXP_DIM = 1024; C_GRID = [0.001, 0.01, 0.1]; SEEDS = [0, 1, 2]; B = 2000
def log(*a): print(*a, flush=True)

def f1s(y, p, nc):
    cm = np.bincount(y * nc + p, minlength=nc * nc).reshape(nc, nc); tp = np.diag(cm); d = cm.sum(0) + cm.sum(1)
    return np.where(d > 0, 2 * tp / np.maximum(d, 1), 0.0)

def vote_feats(v):                                    # v: (N,K) verdict ids, -1 = none -> 3 shares, entropy, top share
    cn = np.stack([(v == c).sum(1) for c in range(3)], 1).astype(float); sh = cn / np.maximum(cn.sum(1, keepdims=True), 1)
    return np.concatenate([sh, -(sh * np.log(sh + 1e-9)).sum(1, keepdims=True), sh.max(1, keepdims=True)], 1)

def autoencode(V, seed):
    torch.manual_seed(seed); sc = StandardScaler().fit(V["train"]); X = {s: torch.tensor(sc.transform(V[s]), dtype=torch.float32) for s in V}; d = X["train"].shape[1]
    enc = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, EXP_DIM), nn.ReLU()); dec = nn.Sequential(nn.Linear(EXP_DIM, 256), nn.ReLU(), nn.Linear(256, d))
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-3, weight_decay=1e-4)
    for _ in range(300):
        opt.zero_grad(); ((dec(enc(X["train"])) - X["train"]) ** 2).mean().backward(); opt.step()
    with torch.no_grad(): return {s: enc(X[s]).numpy() for s in V}

def fit_lr(X, y, nc):                                 # C chosen on validation; returns (val_f1, val_proba, test_proba, C)
    sc = StandardScaler().fit(X["train"]); X = {s: sc.transform(X[s]) for s in X}; best = None
    for Cc in C_GRID:
        lr = LogisticRegression(C=Cc, max_iter=3000, class_weight="balanced").fit(X["train"], y["train"])
        pv, pt = lr.predict_proba(X["val"]), lr.predict_proba(X["test"]); vf = f1s(y["val"], pv.argmax(1), nc).mean()
        if best is None or vf > best[0]: best = (vf, pv, pt, Cc)
    return best

def run_language(lang):
    cfg = CFG[lang]; rng = np.random.default_rng(0); RAW = {}
    # ---------------- load ----------------
    for split, fn in zip(["train", "val", "test"], cfg["files"]):
        d = json.load(open(os.path.join(P, "task 2", "data", cfg["dir"], fn), encoding="utf-8")); N = len(d)
        ids = [it.get("query_id", it.get("query_index")) for it in d] if split == "test" else list(range(N))
        cache = os.path.join(P, "my data", f"traces_fp16_{lang}_{split}.npz")
        if not os.path.exists(cache):
            cl = np.zeros((N, 4096), np.float16); tr = np.zeros((N, T, 4096), np.float16)
            with np.load(os.path.join(P, "my data", cfg["emb"], f"{split}_qwen3emb8b_ar_v1.npz"), allow_pickle=True) as z:
                for r, i in enumerate(ids):
                    a = z[f"id_{i}"]; n = min(len(a) - 1, T); cl[r] = a[0]; tr[r, :n] = a[1:1 + n]
            np.savez(cache, claim=cl, traces=tr); log(lang, "cached", split)
        with np.load(cache) as z: cl, tr = z["claim"].astype(np.float32), z["traces"]
        verd = np.full((N, T), -1)
        for r, it in enumerate(d):
            v = [VL.get(x, -1) for x in it["Verdict_list"][:T]]; verd[r, :len(v)] = v
        RAW[split] = {"ids": ids, "claim_text": [it["claim"].strip() for it in d], "label": [str(it.get("label", it.get("Label"))) for it in d], "verd": verd,
                      "evw": np.array([sum(len(str(e).split()) for e in (it.get("evidences") or [])) for it in d]), "claim": cl, "traces": tr}
    classes = [c for c in VERD if c in set(RAW["train"]["label"])]; nc = len(classes); CL = {c: i for i, c in enumerate(classes)}; cls_v = [VL[c] for c in classes]
    log(f"\n################ {lang}: classes {classes}")
    # ---------------- settings ----------------
    test_keys = set(RAW["test"]["claim_text"]); KEEP = {"original": {s: np.arange(len(RAW[s]["label"])) for s in RAW}}
    seen, ktr = set(), []
    for r, c in enumerate(RAW["train"]["claim_text"]):
        if c not in test_keys and c not in seen: ktr.append(r); seen.add(c)
    kva = []
    for r, c in enumerate(RAW["val"]["claim_text"]):
        if c not in test_keys and c not in seen: kva.append(r); seen.add(c)
    KEEP["dedup"] = {"train": np.array(ktr), "val": np.array(kva), "test": KEEP["original"]["test"]}
    MAIN, KROWS = [], []
    for setting in ["original", "dedup"]:
        D = {s: {k: (v[KEEP[setting][s]] if isinstance(v, np.ndarray) else [v[i] for i in KEEP[setting][s]]) for k, v in RAW[s].items()} for s in RAW}
        y = {s: np.array([CL[l] for l in D[s]["label"]]) for s in D}; yt = y["test"]
        log(f"\n===== {lang} / {setting}: train {len(y['train'])} | val {len(y['val'])} | test {len(yt)} | train labels {np.bincount(y['train'], minlength=nc).tolist()} "
            f"| test labels {np.bincount(yt, minlength=nc).tolist()} | per-trace accuracy " + " / ".join(f"{s} {np.mean(D[s]['verd'] == np.array(cls_v)[y[s]][:, None]):.3f}" for s in D)
            + f" | evidence words median " + " / ".join(f"{s} {np.median(D[s]['evw']):.0f}" for s in D))
        # PCA bases on this setting's train; project claims and every single trace
        tmean = np.stack([D["train"]["traces"][i].astype(np.float32).mean(0) for i in range(len(y["train"]))])
        pc, pt_ = PCA(PCA_FIT, random_state=0).fit(D["train"]["claim"]), PCA(PCA_FIT, random_state=0).fit(tmean); del tmean
        Wt, mut = pt_.components_[:128].T.astype(np.float32), pt_.mean_.astype(np.float32)
        for s in D:
            D[s]["c"] = pc.transform(D[s]["claim"])[:, :128].astype(np.float32); N = len(y[s]); D[s]["t"] = np.zeros((N, T, 128), np.float32)
            for a in range(0, N, 128): D[s]["t"][a:a + 128] = (D[s]["traces"][a:a + 128].astype(np.float32) - mut) @ Wt
            D[s]["given"] = np.tile(np.arange(T), (N, 1))
        mu, sd = np.log1p(D["train"]["evw"]).mean(), max(np.log1p(D["train"]["evw"]).std(), 1e-6)
        for s in D: D[s]["z"] = ((np.log1p(D[s]["evw"]) - mu) / sd)[:, None]
        # p/n ranking from a vote-count model (out-of-fold on train)
        VF = {s: vote_feats(D[s]["verd"]) for s in D}; sc = StandardScaler().fit(VF["train"]); Pn = {}
        mk = lambda: LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced")
        Pn["train"] = np.zeros((len(y["train"]), nc))
        for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(VF["train"], y["train"]):
            Pn["train"][b] = mk().fit(sc.transform(VF["train"][a]), y["train"][a]).predict_proba(sc.transform(VF["train"][b]))
        m = mk().fit(sc.transform(VF["train"]), y["train"])
        for s in ["val", "test"]: Pn[s] = m.predict_proba(sc.transform(VF[s]))
        for s in D:
            v = D[s]["verd"]; order = np.zeros_like(v)
            for r in range(len(v)):
                n = {c: max((v[r] == c).sum(), 1) for c in range(3)}; prio = np.array([Pn[s][r, cls_v.index(x)] / n[x] if x in cls_v else -1.0 for x in v[r]])
                order[r] = np.argsort(-prio, kind="stable")
            D[s]["pn"] = order

        def build(order, K, dim, ae_seed):            # features from the K selected traces
            X, V, vote = {}, {}, {}
            for s in D:
                idx = D[s][order][:, :K]; V[s] = vote_feats(np.take_along_axis(D[s]["verd"], idx, 1)); vote[s] = V[s][:, cls_v].argmax(1)
            E = autoencode(V, ae_seed) if ae_seed is not None else V
            for s in D:
                tm = np.take_along_axis(D[s]["t"], D[s][order][:, :K, None], 1).mean(1)
                X[s] = np.concatenate([E[s], D[s]["z"], V[s][:, :4] * D[s]["z"], D[s]["c"][:, :dim], tm[:, :dim]], 1)
            return X, V, vote

        idx = rng.integers(0, len(yt), (B, len(yt))); bsf = lambda p: np.array([f1s(yt[i], p[i], nc).mean() for i in idx]); q = lambda a: (np.quantile(a, .025), np.quantile(a, .975))
        # ---------------- main table (all 20 traces) ----------------
        _, V20, vote20 = build("given", T, 64, None); majb = bsf(vote20["test"]); MAJ = f1s(yt, vote20["test"], nc).mean()
        def add(name, preds, vals, extra=None):
            tf = [f1s(yt, p, nc).mean() for p in preds]; p = preds[0]; ft = f1s(yt, p, nc); fb = bsf(p); db = fb - majb
            r = {"language": lang, "setting": setting, "model": name, "n_train": len(y["train"]), "val_f1": float(np.mean(vals)), "test_f1": float(np.mean(tf)), "test_f1_std": float(np.std(tf)),
                 "ci_lo": q(fb)[0], "ci_hi": q(fb)[1], "diff_vs_vote": float(np.mean(tf)) - MAJ, "diff_lo": q(db)[0], "diff_hi": q(db)[1], "test_acc": float(np.mean(p == yt))}
            r.update({f"f1_{c}": ft[i] for i, c in enumerate(classes)}); r.update(extra or {}); MAIN.append(r)
            log(f"  {name:38s} | val {r['val_f1']:.4f} | test {r['test_f1']:.4f} ±{r['test_f1_std']:.4f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] | vs vote {r['diff_vs_vote']:+.4f} "
                f"[{r['diff_lo']:+.3f}, {r['diff_hi']:+.3f}] | acc {r['test_acc']:.4f} | per class {ft.round(3).tolist()}")
        add("Majority vote of the traces", [vote20["test"]], [f1s(y["val"], vote20["val"], nc).mean()])
        b = fit_lr(V20, y, nc); add("Votes (logistic regression)", [b[2].argmax(1)], [b[0]])
        X, _, _ = build("given", T, 64, None); b = fit_lr(X, y, nc); add("v1: votes + gate + PCA-64 content", [b[2].argmax(1)], [b[0]])
        runs = [fit_lr(build("given", T, 128, sd_)[0], y, nc) for sd_ in SEEDS]; pt0 = runs[0][2]; mt = np.sort(pt0, 1); mt = mt[:, -1] - mt[:, -2]
        add("v2: AE_1024 + gate + PCA-128 content", [r[2].argmax(1) for r in runs], [r[0] for r in runs], {"margin_error_auroc": roc_auc_score(pt0.argmax(1) != yt, -mt) if 0 < np.mean(pt0.argmax(1) != yt) < 1 else np.nan})
        pred = pd.DataFrame(pt0, columns=[f"prob_{c}" for c in classes]); pred.insert(0, "query_id", D["test"]["ids"]); pred["margin"] = mt
        pred["pred"] = [classes[i] for i in pt0.argmax(1)]; pred["gold"] = D["test"]["label"]; pred["majority_vote"] = [classes[i] for i in vote20["test"]]
        pred.to_csv(os.path.join(P, "results", f"shift_aware_v2_{lang}_{setting}_test_predictions.csv"), index=False)
        # ---------------- K-curve ----------------
        PRED, VOTE = {}, {}
        for order in ["given", "pn"]:
            for K in range(1, T + 1):
                t0 = time.time(); X, _, vote = build(order, K, 128, 0); vf, pv, pt, Cc = fit_lr(X, y, nc); p = pt.argmax(1); ft = f1s(yt, p, nc)
                PRED[order, K], VOTE[order, K] = p, vote["test"]
                KROWS.append({"language": lang, "setting": setting, "order": order, "K": K, "C": Cc, "val_f1": vf, "test_f1": ft.mean(), "test_acc": float(np.mean(p == yt)),
                              **{f"f1_{c}": ft[i] for i, c in enumerate(classes)}, "vote_test_f1": f1s(yt, vote["test"], nc).mean()})
                log(f"  K-curve {order:5s} K {K:2d} | val {vf:.4f} | test {ft.mean():.4f} | vote {KROWS[-1]['vote_test_f1']:.4f} | {time.time() - t0:.0f}s")
        bs = {k: bsf(p) for k, p in PRED.items()}; bv = {k: bsf(p) for k, p in VOTE.items()}
        for r in KROWS:
            if r["setting"] != setting or r["language"] != lang: continue
            o, k = r["order"], r["K"]; r["ci_lo"], r["ci_hi"] = q(bs[o, k])
            for name, ref in [("vs_K20", bs[o, T]), ("vs_vote_sameK", bv[o, k]), ("pn_minus_given", bs["given", k])]:
                dd = bs[o, k] - ref; r[f"d_{name}"], (r[f"d_{name}_lo"], r[f"d_{name}_hi"]) = float(np.mean(dd)), q(dd)
    pd.DataFrame(MAIN).to_csv(os.path.join(P, "results", f"shift_aware_v2_{lang}.csv"), index=False)
    R = pd.DataFrame(KROWS); R.to_csv(os.path.join(P, "results", f"k_curve_shift_aware_v2_{lang}.csv"), index=False)
    # ---------------- figure ----------------
    BLUE, ORANGE = "#2a78d6", "#eb6834"; INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.6), facecolor=SURF)
    for ax, setting, title in zip(axes, ["original", "dedup"], ["Test, original train / validation (test claims leak into train)", "Test, deduplicated train / validation"]):
        ax.set_facecolor(SURF); ax.spines[["top", "right", "left"]].set_visible(False); ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)
        lo, hi = 1, 0; n_tr = [m["n_train"] for m in MAIN if m["setting"] == setting][0]
        for order, c, lab in [("given", BLUE, "dataset order (first K)"), ("pn", ORANGE, "p/n ranking (top K)")]:
            a = R[(R.setting == setting) & (R.order == order)].sort_values("K")
            ax.fill_between(a.K, a.ci_lo, a.ci_hi, color=c, alpha=0.10, lw=0)
            ax.plot(a.K, a.test_f1, color=c, lw=2.2, marker="o", ms=4.5, label=f"Classifier v2, {lab}"); ax.plot(a.K, a.vote_test_f1, color=c, lw=1.4, ls=(0, (3, 2)), label=f"Majority vote, {lab}")
            lo, hi = min(lo, a.ci_lo.min(), a.vote_test_f1.min()), max(hi, a.ci_hi.max(), a.vote_test_f1.max())
        ax.set_ylim(np.floor((lo - 0.01) * 20) / 20, min(1.0, np.ceil((hi + 0.01) * 20) / 20)); ax.set_xlim(0.4, T + 0.6); ax.set_xticks([1, 3, 5, 10, 15, 20])
        ax.set_xlabel("Number of traces given to the classifier (K)"); ax.set_ylabel("Macro-F1")
        ax.set_title(f"{title}\n{n_tr:,} training claims", loc="left", fontsize=10.5, fontweight="bold", color=INK, pad=10)
    axes[0].legend(frameon=False, fontsize=8.8, labelcolor=INK2, loc="lower right", ncol=2)
    fig.suptitle(f"{lang.capitalize()}: K-curve with classifier version 2, Macro-F1 against the number of traces", x=0.012, ha="left", fontsize=12.5, fontweight="bold", color=INK)
    fig.text(0.012, 0.012, f"{lang.capitalize()} test, {len(RAW['test']['label']):,} claims, classes: {', '.join(classes)}. Classifier refitted for every K. p/n ranking computed from a vote-count model. "
             "Band = 95% bootstrap interval.", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.035, 1, 0.93), w_pad=3)
    out = os.path.join(P, "results", f"k_curve_shift_aware_v2_{lang}.png"); fig.savefig(out, dpi=160, facecolor=SURF); plt.close(fig); log(out)

if __name__ == "__main__":
    for lang in (sys.argv[1:] or ["arabic", "spanish"]): run_language(lang)
