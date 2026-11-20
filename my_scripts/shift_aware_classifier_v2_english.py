"""Redesigned classifier, version 2 (English): four requested changes, each as an ablation step.

  1. PCA size        64 / 128 / 256 / 512 for every embedding block (the best on validation is kept for the next steps)
  2. verdict block   Raw / Duplication_1024 / AE_1024 / Projection_1024
  3. disagreement    std of the trace vectors across the traces of a claim (4096-d) -> PCA
  4. margin          (a) vote margin = top vote share - second vote share, as a feature (also gated by evidence length)
                     (b) decision rule on the classifier output: if p_top - p_second < tau, predict Conflicting (tau from validation)
                     (c) the same output margin as an error detector (AUROC)
Classifier: class-balanced logistic regression, C chosen on validation. AE and Projection are small networks trained on the
train split only (3 seeds). Outputs: results/shift_aware_classifier_v2_english.csv
"""
import json, os, time, warnings, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMB = os.path.join(P, "my data", "qwen3 english embeddings")
POOLED = os.path.join(P, "my data", "pooled_qwen3emb8b_en_v1.npz")
STD = os.path.join(P, "my data", "pooled_std_qwen3emb8b_en_v1.npz")
C = ["False", "True", "Conflicting"]; L = {c: i for i, c in enumerate(C)}
PCA_SIZES = [64, 128, 256, 512]; C_GRID = [0.001, 0.01, 0.1]; SEEDS = [0, 1, 2]; EXP_DIM = 1024; B = 2000; rng = np.random.default_rng(0)
def log(*a): print(*a, flush=True)
def f1s(y, p):
    cm = np.bincount(y * 3 + p, minlength=9).reshape(3, 3); tp = np.diag(cm); d = cm.sum(0) + cm.sum(1)
    return np.where(d > 0, 2 * tp / np.maximum(d, 1), 0.0)

# ---------------- data ----------------
D = {}
for split, fn in [("train", "train"), ("val", "validation"), ("test", "test")]:
    d = json.load(open(os.path.join(P, "task 2", "data", "English", f"{fn}.json"), encoding="utf-8"))
    cn = np.stack([np.bincount([L[v] for v in it["Verdict_list"] if v in L], minlength=3) for it in d]).astype(float)
    sh = cn / np.maximum(cn.sum(1, keepdims=True), 1); ss = np.sort(sh, 1)
    D[split] = {"ids": [it["query_id"] for it in d] if split == "test" else list(range(len(d))), "y": np.array([L[it.get("label", it.get("Label"))] for it in d]),
                "maj": cn.argmax(1), "votes": np.concatenate([sh, -(sh * np.log(sh + 1e-9)).sum(1, keepdims=True), ss[:, -1:]], 1),   # 3 shares, entropy, top share
                "vmargin": ss[:, -1:] - ss[:, -2:-1], "evw": np.array([sum(len(e.split()) for e in it["evidences"]) for it in d])}
if not os.path.exists(STD):                          # disagreement: std of the trace vectors of a claim, once
    out = {}
    for s in D:
        sd = np.zeros((len(D[s]["y"]), 4096), np.float16)
        with np.load(os.path.join(EMB, f"{s}_qwen3emb8b_en_v1.npz"), allow_pickle=False) as z:
            for r, i in enumerate(D[s]["ids"]):
                a = z[f"id_{i}"][1:].astype(np.float32)
                if len(a) > 1: sd[r] = a.std(0)
        out[s] = sd; log("trace std", s)
    np.savez(STD, **out)
with np.load(POOLED) as z, np.load(STD) as zs:
    for s in D: D[s]["claim"], D[s]["tmean"], D[s]["tstd"] = z[f"{s}_claim"].astype(np.float32), z[f"{s}_tmean"].astype(np.float32), zs[s].astype(np.float32)
lw_mu, lw_sd = np.log1p(D["train"]["evw"]).mean(), np.log1p(D["train"]["evw"]).std()
for s in D: D[s]["z"] = ((np.log1p(D[s]["evw"]) - lw_mu) / lw_sd)[:, None]
PC = {}
for k in ["claim", "tmean", "tstd"]:                 # one PCA at the largest size; smaller sizes are its first columns
    p = PCA(max(PCA_SIZES), random_state=0).fit(D["train"][k]); PC[k] = {s: p.transform(D[s][k]).astype(np.float32) for s in D}; log("PCA", k)

# ---------------- verdict expansions ----------------
def expand(kind, V, seed):
    """V: dict split -> (N,d) verdict vector. Returns dict split -> expanded matrix."""
    d = V["train"].shape[1]
    if kind == "raw": return V
    if kind == "dup": return {s: np.tile(V[s], (1, EXP_DIM // d + 1))[:, :EXP_DIM] for s in V}
    torch.manual_seed(seed); sc = StandardScaler().fit(V["train"]); T = {s: torch.tensor(sc.transform(V[s]), dtype=torch.float32) for s in V}
    enc = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, EXP_DIM), nn.ReLU())
    if kind == "ae":                                 # autoencoder: reconstruct the verdict vector through a 1024-d code
        head = nn.Sequential(nn.Linear(EXP_DIM, 256), nn.ReLU(), nn.Linear(256, d)); loss_fn = lambda o: ((o - T["train"]) ** 2).mean()
    else:                                            # projection: trained to predict the label from the verdict vector
        head = nn.Linear(EXP_DIM, 3); y = torch.tensor(D["train"]["y"], dtype=torch.long); cnt = torch.bincount(y, minlength=3).float()
        w = len(y) / (3 * cnt); loss_fn = lambda o: nn.functional.cross_entropy(o, y, weight=w)
    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=1e-3, weight_decay=1e-4)
    for _ in range(300):
        opt.zero_grad(); loss_fn(head(enc(T["train"]))).backward(); opt.step()
    with torch.no_grad(): return {s: enc(T[s]).numpy() for s in V}

# ---------------- one configuration ----------------
yv, yt = D["val"]["y"], D["test"]["y"]; idx = rng.integers(0, len(yt), (B, len(yt)))
majb = np.array([f1s(yt[i], D["test"]["maj"][i]).mean() for i in idx]); MAJ = f1s(yt, D["test"]["maj"]).mean()

def run(name, dim, exp="raw", dis=False, vmargin=False, step=""):
    t0 = time.time(); seeds = SEEDS if exp in ("ae", "proj") else [0]; per_seed = []
    for seed in seeds:
        V = {s: np.concatenate([D[s]["votes"]] + ([D[s]["vmargin"]] if vmargin else []), 1) for s in D}
        E = expand(exp, V, seed); X = {}
        for s in D:
            gate_in = np.concatenate([D[s]["votes"][:, :4]] + ([D[s]["vmargin"]] if vmargin else []), 1)
            X[s] = np.concatenate([E[s], D[s]["z"], gate_in * D[s]["z"], PC["claim"][s][:, :dim], PC["tmean"][s][:, :dim]] + ([PC["tstd"][s][:, :dim]] if dis else []), 1)
        sc = StandardScaler().fit(X["train"]); X = {s: sc.transform(X[s]) for s in X}; best = None
        for Cc in C_GRID:
            lr = LogisticRegression(C=Cc, max_iter=3000, class_weight="balanced").fit(X["train"], D["train"]["y"])
            pv, pt = lr.predict_proba(X["val"]), lr.predict_proba(X["test"]); vf = f1s(yv, pv.argmax(1)).mean()
            if best is None or vf > best[0]: best = (vf, pv, pt, Cc)
        per_seed.append(best + (X["train"].shape[1],))
    tf = [f1s(yt, r[2].argmax(1)).mean() for r in per_seed]; vf, pv, pt, Cc, nfeat = per_seed[0]; p = pt.argmax(1); ft = f1s(yt, p)
    fb = np.array([f1s(yt[i], p[i]).mean() for i in idx]); db = fb - majb
    # margin of the classifier output: decision rule (tau from validation) and error detector
    mv, mt = np.sort(pv, 1), np.sort(pt, 1); mv, mt = mv[:, -1] - mv[:, -2], mt[:, -1] - mt[:, -2]
    taus = np.linspace(0, 0.6, 31); rule = lambda pr, m, t: np.where(m < t, 2, pr.argmax(1))
    tau = taus[int(np.argmax([f1s(yv, rule(pv, mv, t)).mean() for t in taus]))]; pr = rule(pt, mt, tau); fr = f1s(yt, pr)
    r = {"step": step, "model": name, "pca": dim, "verdict": exp, "disagreement": dis, "vote_margin": vmargin, "n_features": nfeat, "C": Cc, "val_f1": float(np.mean([x[0] for x in per_seed])),
         "test_f1": float(np.mean(tf)), "test_f1_std": float(np.std(tf)) if len(tf) > 1 else 0.0, "ci_lo": np.quantile(fb, .025), "ci_hi": np.quantile(fb, .975),
         "diff_vs_vote": float(np.mean(tf)) - MAJ, "diff_lo": np.quantile(db, .025), "diff_hi": np.quantile(db, .975), "test_acc": float(np.mean(p == yt)),
         "f1_False": ft[0], "f1_True": ft[1], "f1_Conflicting": ft[2], "rule_tau": tau, "rule_test_f1": fr.mean(), "rule_f1_Conflicting": fr[2],
         "rule_best_possible": max(f1s(yt, rule(pt, mt, t)).mean() for t in taus), "margin_error_auroc": roc_auc_score(p != yt, -mt)}
    log(f"[{step}] {name:46s} | feats {nfeat:5d} | C {Cc:5.3f} | val {r['val_f1']:.4f} | test {r['test_f1']:.4f} ±{r['test_f1_std']:.4f} | diff {r['diff_vs_vote']:+.4f} "
        f"| C-F1 {ft[2]:.3f} | rule tau {tau:.2f} -> {fr.mean():.4f} | AUROC {r['margin_error_auroc']:.3f} | {time.time() - t0:.0f}s")
    return r

log(f"majority vote: test Macro-F1 {MAJ:.4f}")
ROWS = [run(f"PCA {d}", d, step="1 PCA size") for d in PCA_SIZES]
DIM = max(ROWS, key=lambda r: r["val_f1"])["pca"]; log(f"--> PCA size chosen on validation: {DIM}")
NAMES = {"raw": "Raw", "dup": "Duplication_1024", "ae": "AE_1024", "proj": "Projection_1024"}
ROWS += [run(f"verdict = {NAMES[e]}", DIM, exp=e, step="2 verdict block") for e in ["dup", "ae", "proj"]]
ROWS += [run("+ disagreement", DIM, dis=True, step="3 disagreement"), run("+ vote margin", DIM, vmargin=True, step="4 margin"),
         run("+ disagreement + vote margin", DIM, dis=True, vmargin=True, step="3+4")]
ROWS += [run(f"all changes, verdict = {NAMES[e]}", DIM, exp=e, dis=True, vmargin=True, step="full") for e in ["dup", "ae", "proj"]]
R = pd.DataFrame(ROWS); R.to_csv(os.path.join(P, "results", "shift_aware_classifier_v2_english.csv"), index=False)
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 40)
print(R[["step", "model", "n_features", "val_f1", "test_f1", "test_f1_std", "diff_vs_vote", "diff_lo", "diff_hi", "test_acc", "f1_True", "f1_Conflicting",
         "rule_tau", "rule_test_f1", "rule_best_possible", "margin_error_auroc"]].round(4).to_string(index=False))
