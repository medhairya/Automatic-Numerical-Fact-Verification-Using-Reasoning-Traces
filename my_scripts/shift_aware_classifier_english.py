"""Redesigned verdict classifier for English (CheckThat! 2026 Task 2): vote-anchored, evidence-length gated, linear.

  input  = vote features (3 shares, entropy, top share)
         + gate: z = standardised log(evidence words), and z x vote features
         + content: PCA of the claim embedding and of the mean trace embedding (Qwen3-Embedding-8B, no instruction)
  model  = class-balanced multinomial logistic regression with strong L2 (deterministic: no seeds)
  report = Macro-F1 with bootstrap 95% intervals and the paired difference to the majority vote

Runs on CPU in a few minutes. Outputs: results/shift_aware_classifier_english.csv and _test_predictions.csv
"""
import json, os, warnings, numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMB = os.path.join(P, "my data", "qwen3 english embeddings")
CE = os.path.join(P, "my data", "english-claim+evidence-embeddings")
CACHE = os.path.join(P, "my data", "pooled_qwen3emb8b_en_v1.npz")
C = ["False", "True", "Conflicting"]; L = {c: i for i, c in enumerate(C)}
PCA_DIM = 64; C_GRID = [0.001, 0.01, 0.1, 1.0]; B = 2000; rng = np.random.default_rng(0)

def f1s(y, p):
    cm = np.bincount(y * 3 + p, minlength=9).reshape(3, 3); tp = np.diag(cm); d = cm.sum(0) + cm.sum(1)
    return np.where(d > 0, 2 * tp / np.maximum(d, 1), 0.0)

# ---------------- data ----------------
D = {}
for split, fn in [("train", "train"), ("val", "validation"), ("test", "test")]:
    d = json.load(open(os.path.join(P, "task 2", "data", "English", f"{fn}.json"), encoding="utf-8"))
    D[split] = {"ids": [it["query_id"] for it in d] if split == "test" else list(range(len(d))),
                "y": np.array([L[it.get("label", it.get("Label"))] for it in d]),
                "cn": np.stack([np.bincount([L[v] for v in it["Verdict_list"] if v in L], minlength=3) for it in d]).astype(float),
                "evw": np.array([sum(len(e.split()) for e in it["evidences"]) for it in d])}
if not os.path.exists(CACHE):                       # claim vector + mean of the trace vectors, once
    out = {}
    for split in D:
        N = len(D[split]["y"]); cl = np.zeros((N, 4096), np.float16); tm = np.zeros((N, 4096), np.float16)
        with np.load(os.path.join(EMB, f"{split}_qwen3emb8b_en_v1.npz"), allow_pickle=False) as z:
            for r, i in enumerate(D[split]["ids"]):
                a = z[f"id_{i}"].astype(np.float32); cl[r] = a[0]; tm[r] = a[1:].mean(0) if len(a) > 1 else 0
        out[f"{split}_claim"], out[f"{split}_tmean"] = cl, tm; print("pooled", split, flush=True)
    np.savez(CACHE, **out)
with np.load(CACHE) as z:
    for split in D: D[split]["claim"], D[split]["tmean"] = z[f"{split}_claim"].astype(np.float32), z[f"{split}_tmean"].astype(np.float32)
for split in D:                                     # frozen claim + evidence vectors (optional ablation row)
    with np.load(os.path.join(CE, f"{split}_claimevid_qwen3emb8b4bit_en_v1.npz"), allow_pickle=True) as z:
        pos = {str(i): r for r, i in enumerate(z["__ids__"].tolist())}
        D[split]["ce"] = z["emb"][[pos[str(i)] for i in D[split]["ids"]]].astype(np.float32)

# ---------------- features ----------------
pca = {k: PCA(PCA_DIM, random_state=0).fit(D["train"][k]) for k in ["claim", "tmean", "ce"]}
lw_mu, lw_sd = np.log1p(D["train"]["evw"]).mean(), np.log1p(D["train"]["evw"]).std()
def feats(s, parts):
    cn = D[s]["cn"]; sh = cn / np.maximum(cn.sum(1, keepdims=True), 1); ent = -(sh * np.log(sh + 1e-9)).sum(1, keepdims=True)
    v = np.concatenate([sh, ent, sh.max(1, keepdims=True)], 1); z = ((np.log1p(D[s]["evw"]) - lw_mu) / lw_sd)[:, None]; out = [v]
    if "gate" in parts: out += [z, v[:, :4] * z]
    for k in ["claim", "tmean", "ce"]:
        if k in parts: out.append(pca[k].transform(D[s][k]))
    return np.concatenate(out, 1)

def fit(parts, Cc, balanced=True):
    X = {s: feats(s, parts) for s in D}; sc = StandardScaler().fit(X["train"])
    lr = LogisticRegression(C=Cc, max_iter=5000, class_weight="balanced" if balanced else None).fit(sc.transform(X["train"]), D["train"]["y"])
    return {s: lr.predict_proba(sc.transform(X[s])) for s in D}

# ---------------- evaluation ----------------
yv, yt = D["val"]["y"], D["test"]["y"]; maj = D["test"]["cn"].argmax(1); idx = rng.integers(0, len(yt), (B, len(yt)))
majb = np.array([f1s(yt[i], maj[i]).mean() for i in idx])
def row(name, pt, val_f1, Cc=""):
    ft = f1s(yt, pt); fb = np.array([f1s(yt[i], pt[i]).mean() for i in idx]); db = fb - majb
    return {"model": name, "C": Cc, "val_f1": val_f1, "test_f1": ft.mean(), "ci_lo": np.quantile(fb, .025), "ci_hi": np.quantile(fb, .975),
            "diff_vs_vote": ft.mean() - f1s(yt, maj).mean(), "diff_lo": np.quantile(db, .025), "diff_hi": np.quantile(db, .975),
            "p_better": float(np.mean(db > 0)), "test_acc": float(np.mean(pt == yt)), "f1_False": ft[0], "f1_True": ft[1], "f1_Conflicting": ft[2]}

ROWS = [row("Majority vote of the traces", maj, f1s(yv, D["val"]["cn"].argmax(1)).mean())]
VARIANTS = [("Votes", ["votes"]), ("Votes + gate", ["gate"]), ("Votes + content", ["claim", "tmean"]),
            ("Votes + gate + content  [new pipeline]", ["gate", "claim", "tmean"]),
            ("Votes + gate + claim only", ["gate", "claim"]), ("Votes + gate + traces only", ["gate", "tmean"]),
            ("Votes + gate + content + claim-evidence vector", ["gate", "claim", "tmean", "ce"])]
FINAL = None
for name, parts in VARIANTS:
    runs = [(Cc, fit(parts, Cc)) for Cc in C_GRID]
    vals = [f1s(yv, p["val"].argmax(1)).mean() for _, p in runs]; tests = [f1s(yt, p["test"].argmax(1)).mean() for _, p in runs]
    b = int(np.argmax(vals)); Cc, p = runs[b]                       # C chosen on validation
    r = row(name, p["test"].argmax(1), vals[b], Cc); r["test_f1_min_over_C"], r["test_f1_max_over_C"] = min(tests), max(tests); ROWS.append(r)
    if "new pipeline" in name: FINAL = p
R = pd.DataFrame(ROWS); R.to_csv(os.path.join(P, "results", "shift_aware_classifier_english.csv"), index=False)
pd.set_option("display.width", 260); pd.set_option("display.max_columns", 30)
print(R.round(4).to_string(index=False))
pred = pd.DataFrame(FINAL["test"], columns=[f"prob_{c}" for c in C]); pred.insert(0, "query_id", D["test"]["ids"])
pred["pred"] = [C[i] for i in FINAL["test"].argmax(1)]; pred["gold"] = [C[i] for i in yt]; pred["majority_vote"] = [C[i] for i in maj]
pred.to_csv(os.path.join(P, "results", "shift_aware_classifier_english_test_predictions.csv"), index=False)

print("\nSensitivity of the new pipeline to the PCA size (C chosen on validation):")
for dim in [16, 32, 64, 128, 256]:
    pca = {k: PCA(dim, random_state=0).fit(D["train"][k]) for k in ["claim", "tmean"]}
    runs = [fit(["gate", "claim", "tmean"], Cc) for Cc in C_GRID]; vals = [f1s(yv, p["val"].argmax(1)).mean() for p in runs]
    tests = [f1s(yt, p["test"].argmax(1)).mean() for p in runs]; b = int(np.argmax(vals))
    print(f"  PCA {dim:3d}: val {vals[b]:.4f} | test {tests[b]:.4f} | test range over C {min(tests):.4f}-{max(tests):.4f}")
