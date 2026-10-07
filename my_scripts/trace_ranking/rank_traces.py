"""
Recall@5-optimal ranking of LLM reasoning traces
CLEF CheckThat! 2026 Task 2 -- sub-problem B (trace ranking)

WHY THIS WORKS
--------------
The official scorer marks a trace relevant iff its verdict == gold label. So every trace
with the same verdict is interchangeable, and Recall@5 of a claim depends only on HOW MANY
traces of each verdict are in the top 5 (k_c), never on which ones:

    Recall@5 = k_y / n_y          (y = gold label, n_y = #traces with verdict y)

With p_c = P(gold = c | claim, traces), the expected Recall@5 of a top-5 allocation is

    E[R@5] = sum_c  p_c * k_c / n_c        subject to  sum_c k_c = 5,  0 <= k_c <= n_c

This is linear in k, so the exact optimum is greedy: rank verdict GROUPS by p_c / n_c and
list all traces of the best group first, then the next group, and so on. Rankers that
score traces one at a time (cosine, rerankers, reward models) cannot do this.

The problem therefore reduces to estimating a well-calibrated p_c per claim. Feature blocks
(logistic regression on their concatenation):
  counts : verdict-count vector                                    (always on)
  claim  : PCA of the claim embedding                              (needs npz)
  group  : per-verdict-group trace centroids, cohesion, claim sims (needs per-trace npz)
  knn    : similarity-weighted labels of the 10 nearest training claims
plus
  memory : a test claim whose exact text is in TRAIN+VAL takes that label (always on)
           (AR: 192/395, ES: 278/1135 scored test claims; labels agreed 100%)
  shift  : covariate-shift importance weights, train+val vs unlabeled test (Shimodaira 2000)

Blocks and the regularisation strength are chosen by 5-fold cross-validated Recall@5 on
TRAIN+VAL; the final model is refit on TRAIN+VAL and applied to TEST. Test labels are only
used for the final report. The ranking never uses the verdict classifier.

USAGE (CPU is enough)
    python rank_traces.py --lang all --data-root <data> --emb EN=<dir> --emb AR=<dir> --emb ES=<dir>
Outputs in --out-dir:
    {LANG}_test_predictions.json   scorer.py-compatible (query_id, Claim, Label, Verdict_BoN,
                                   BoN_Verdict_list, score_list, label_probs)
    {LANG}_results.csv             test Recall@k of all methods, all rows and deduped (scorer.py)
"""

import argparse
import csv
import glob
import json
import os
import re
import warnings
from collections import Counter

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

CLASSES = ["true", "false", "conflicting"]
K = 5

# Data files (relative to --data-root; if missing, the basename is searched recursively)
# and embedding npz names (inside the folder given with --emb LANG=<dir>).
LANGS = {
    "EN": dict(
        train="English/train.json", val="English/validation.json", test="English/test.json",
        emb=("{split}_qwen3emb8b_en_v1.npz", "{split}_trace_emb_en.npz"),
    ),
    "AR": dict(
        train="Arabic/clef2026_gpt4_o_mini_train_arabic.json",
        val="Arabic/clef2026_gpt4_o_mini_val_arabic.json",
        test="Arabic/clef_arabic_test_gold_labels-final.json",
        emb="{split}_qwen3emb8b_ar_v1.npz",
    ),
    "ES": dict(
        train="Spanish/spanish_train.json", val="Spanish/spanish_val.json",
        test="Spanish/clef_spanish_test_final_with_gold_labels.json",
        emb="{split}_qwen3emb8b_ar_v1.npz",
    ),
}


def resolve(root, rel):
    p = os.path.join(root, rel)
    if os.path.exists(p):
        return p
    hits = glob.glob(os.path.join(root, "**", os.path.basename(rel)), recursive=True)
    parent = os.path.basename(os.path.dirname(rel)).lower()
    hits.sort(key=lambda h: parent not in h.lower())
    return hits[0] if hits else None


# ============================================================
# DATA
# ============================================================
def load_json(path):
    with open(path, encoding="utf-8") as f:
        s = f.read()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # Local English train.json has one entry whose '"Reasoning_traces": [' key is missing
        return json.loads(re.sub(r'\{\s*\nx?(\s+)"(?![A-Za-z_]+"\s*:)', r'{\n        "Reasoning_traces": [\n\1"', s))


def norm(v):
    return str(v).strip().lower()


def claim_key(text):
    return re.sub(r"\s+", " ", str(text)).strip().lower()


class Data:
    """Arrays for one split. Embedding-derived fields stay None when no npz is available."""

    def __init__(self, items):
        self.items = items
        self.verdicts = [[norm(v) for v in e["Verdict_list"]] for e in items]
        self.gold = [norm(e.get("label", e.get("Label", ""))) for e in items]
        self.y = np.array([CLASSES.index(g) if g in CLASSES else -1 for g in self.gold])
        self.N = np.array([[sum(v == c for v in V) for c in CLASSES] for V in self.verdicts])
        self.qid = [e.get("query_id", e.get("query_index", i)) for i, e in enumerate(items)]
        self.keys = np.array([claim_key(e["claim"]) for e in items], dtype=object)
        self.claim = None       # (n, dim) L2-normalised claim embeddings
        self.claim_pca = None   # (n, d)
        self.group = None       # (n, 9 + 3d) per-verdict-group features
        self.rep = None         # per claim: per-trace representativeness (within-group tie-break)
        self.claim_cos = None   # per claim: per-trace claim cosine (old baseline)

    def take(self, idx):
        d = Data.__new__(Data)
        for k, v in self.__dict__.items():
            if isinstance(v, np.ndarray):
                setattr(d, k, v[idx])
            elif isinstance(v, list):
                setattr(d, k, [v[i] for i in idx])
            else:
                setattr(d, k, v)
        return d


def concat(a, b):
    d = Data.__new__(Data)
    for k, v in a.__dict__.items():
        w = getattr(b, k)
        if isinstance(v, np.ndarray) and isinstance(w, np.ndarray):
            setattr(d, k, np.concatenate([v, w]))
        elif isinstance(v, list) and isinstance(w, list):
            setattr(d, k, v + w)
        else:
            setattr(d, k, None)
    return d


def count_features(N):
    tot = N.sum(1, keepdims=True).clip(min=1)
    return np.hstack([N / tot, np.log1p(N), (N == 0).astype(float)])


# ============================================================
# EMBEDDINGS
# npz layout: key "id_{qid}" (test) or "id_{position}" (train/val) -> (1 + #non-empty traces, dim)
# row 0 = claim, rows 1.. = non-empty reasoning traces in original order.
# A claim-only npz (1 row per claim) is fine: the group block is then disabled.
# ============================================================
def npz_key(z, d, i):
    for k in (f"id_{d.qid[i]}", f"id_{i}"):
        if k in z.files:
            return k
    return None


def read_npz(path, d):
    """Streams one claim at a time and keeps only the claim vector + 3 group centroids per claim,
    so RAM stays small even for 8B float32 embeddings."""
    z = np.load(path)
    claims, cents, coh, rep, ccos, has_traces, missing, mism = [], [], [], [], [], False, 0, 0
    for i, e in enumerate(d.items):
        n_tr = len(d.verdicts[i])
        k = npz_key(z, d, i)
        if k is None:
            missing += 1
            claims.append(None)
            cents.append([None] * 3)
            coh.append([0.0] * 3)
            rep.append(np.zeros(n_tr))
            ccos.append(np.zeros(n_tr))
            continue
        a = z[k].astype(np.float32)
        a /= np.linalg.norm(a, axis=1, keepdims=True) + 1e-9
        claims.append(a[0])
        keep = [j for j, t in enumerate(e["Reasoning_traces"]) if str(t).strip()]
        mism += len(keep) != len(a) - 1
        keep = keep[: len(a) - 1]
        tr = a[1 : 1 + len(keep)]
        has_traces |= len(keep) > 0
        r, c = np.full(n_tr, -1.0), np.full(n_tr, -1.0)
        if keep:
            c[keep] = tr @ a[0]
        V = [d.verdicts[i][j] for j in keep]
        cc, hh = [], []
        for cls in CLASSES:
            g = [j for j, v in enumerate(V) if v == cls]
            if g:
                m = tr[g].mean(0)
                hh.append(float(np.linalg.norm(m)))          # ~ mean pairwise cosine in the group
                m = m / (np.linalg.norm(m) + 1e-9)
                r[[keep[j] for j in g]] = tr[g] @ m          # representativeness (tie-break only)
                cc.append(m)
            else:
                hh.append(0.0)
                cc.append(None)
        cents.append(cc)
        coh.append(hh)
        rep.append(r)
        ccos.append(c)
    z.close()
    dim = next(c for c in claims if c is not None).shape[0]
    d.claim = np.vstack([c if c is not None else np.zeros(dim, np.float32) for c in claims])
    d.rep, d.claim_cos = rep, ccos
    d.cents, d.coh, d.has_traces = cents, np.array(coh), has_traces
    if missing:
        print(f"    WARNING: {missing}/{len(d.items)} claims missing from {os.path.basename(path)}")
    if mism:
        print(f"    WARNING: {mism}/{len(d.items)} claims have a trace-row count different from the JSON")


def finish_embedding_features(splits, dim):
    """PCA fitted on TRAIN+VAL claims and group centroids (unsupervised), applied to every split."""
    fit = []
    for s in ("train", "val"):
        fit.append(splits[s].claim)
        fit += [c[None] for cc in splits[s].cents[:3000] for c in cc if c is not None]
    pca = PCA(dim, random_state=0).fit(np.vstack(fit))
    zero = np.zeros(dim)
    for d in splits.values():
        d.claim_pca = pca.transform(d.claim)
        if d.has_traces:
            rows = []
            for i, cc in enumerate(d.cents):
                sims = [float(c @ d.claim[i]) if c is not None else 0.0 for c in cc]
                pair = [float(cc[a] @ cc[b]) if cc[a] is not None and cc[b] is not None else 0.0
                        for a, b in ((0, 1), (0, 2), (1, 2))]
                blocks = [pca.transform(c[None])[0] if c is not None else zero for c in cc]
                rows.append(np.concatenate([sims, d.coh[i], pair, *blocks]))
            d.group = np.array(rows, np.float32)
        del d.cents, d.coh


# ============================================================
# CLAIM MEMORY
# ============================================================
def exact_memory(bank, q):
    """Label distribution of bank claims with identical text."""
    idx = {}
    for j in np.where(bank.y >= 0)[0]:
        idx.setdefault(bank.keys[j], []).append(j)
    hit, dist = np.zeros(len(q.y), bool), np.zeros((len(q.y), len(CLASSES)))
    for i, k in enumerate(q.keys):
        js = idx.get(k)
        if js:
            hit[i] = True
            dist[i] = np.bincount(bank.y[js], minlength=len(CLASSES)) / len(js)
    return hit, dist


def knn_features(bank, q, same, k=10, chunk=2048):
    """Top-1 / mean cosine to the k nearest training claims and their similarity-weighted labels."""
    ok = np.where(bank.y >= 0)[0]
    Cb, yb = bank.claim[ok], bank.y[ok]
    pos = np.full(len(bank.y), -1)
    pos[ok] = np.arange(len(ok))
    F = np.zeros((len(q.y), 2 + len(CLASSES)))
    for s in range(0, len(q.y), chunk):
        S = q.claim[s : s + chunk] @ Cb.T
        if same:  # leave-one-out: a claim may not vote for itself
            rows = np.arange(s, min(s + chunk, len(q.y)))
            cols = pos[rows]
            S[(rows - s)[cols >= 0], cols[cols >= 0]] = -1
        top = np.argpartition(-S, k, axis=1)[:, :k]
        for r, t in enumerate(top):
            sim = S[r, t]
            w = np.maximum(sim, 0) ** 4
            F[s + r, 0], F[s + r, 1] = sim.max(), sim.mean()
            F[s + r, 2:] = np.bincount(yb[t], weights=w, minlength=len(CLASSES)) / max(w.sum(), 1e-9)
    return F


# ============================================================
# MODEL
# ============================================================
def design(bank, q, cfg, same):
    X = [count_features(q.N)]
    if cfg["claim"]:
        X.append(q.claim_pca)
    if cfg["group"]:
        X.append(q.group)
    if cfg["knn"]:
        X.append(knn_features(bank, q, same))
    return np.hstack(X)


def fit_predict(tr, te, cfg, weights=None):
    """Fits configuration `cfg` on `tr`; returns label probabilities for `te`."""
    ok = tr.y >= 0
    Xtr, Xte = design(tr, tr, cfg, same=True), design(tr, te, cfg, same=False)
    sc = StandardScaler().fit(Xtr[ok])
    m = LogisticRegression(C=cfg["C"], max_iter=5000)
    m.fit(sc.transform(Xtr[ok]), tr.y[ok], sample_weight=None if weights is None else weights[ok])
    P = np.zeros((len(te.y), len(CLASSES)))
    P[:, m.classes_] = m.predict_proba(sc.transform(Xte))
    hit, dist = exact_memory(tr, te)
    P[hit] = 0.9 * dist[hit] + 0.1 * P[hit]
    return P


def shift_weights(tv, te, clip):
    """Importance weights p(test|x)/p(train|x) from a domain classifier on counts (+ claim PCA)."""
    blocks = lambda d: np.hstack([count_features(d.N)] + ([d.claim_pca] if d.claim_pca is not None else []))
    X = np.vstack([blocks(tv), blocks(te)])
    dom = np.r_[np.zeros(len(tv.y)), np.ones(len(te.y))]
    sc = StandardScaler().fit(X)
    m = LogisticRegression(C=0.1, max_iter=5000).fit(sc.transform(X), dom)
    p = m.predict_proba(sc.transform(blocks(tv)))[:, 1]
    w = np.clip(p / (1 - p) * len(tv.y) / len(te.y), 0, clip)
    return w / w.mean()


# ============================================================
# RANKING + EVALUATION
# ============================================================
def rank_claim(p, verdicts, rep=None):
    """Trace order: verdict groups by p_c / n_c, within a group by representativeness."""
    n = Counter(verdicts)
    prio = {v: (p[CLASSES.index(v)] / n[v] if v in CLASSES else -1.0) for v in n}
    rep = np.zeros(len(verdicts)) if rep is None else rep
    return sorted(range(len(verdicts)), key=lambda i: (-prio[verdicts[i]], -rep[i], i))


def orders_from_probs(P, d):
    reps = d.rep or [None] * len(d.y)
    return [rank_claim(p, V, r) for p, V, r in zip(P, d.verdicts, reps)]


def recall_at_k(order, verdicts, gold, k):
    total = sum(v == gold for v in verdicts)
    return sum(verdicts[i] == gold for i in order[:k]) / total if total else 0.0


def evaluate(orders, d, dedupe):
    """dedupe=True mirrors given code/scorer.py (first row per claim text only)."""
    seen, r = set(), {k: [] for k in range(1, K + 1)}
    for i, o in enumerate(orders):
        c = d.items[i]["claim"]
        if dedupe and c in seen:
            continue
        seen.add(c)
        for k in r:
            r[k].append(recall_at_k(o, d.verdicts[i], d.gold[i], k))
    return {k: float(np.mean(v)) for k, v in r.items()}


def baseline_orders(d, kind, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i, V in enumerate(d.verdicts):
        n = Counter(V)
        if kind == "random":
            out.append(list(rng.permutation(len(V))))
        elif kind == "majority_first":
            out.append(sorted(range(len(V)), key=lambda j: (-n[V[j]], j)))
        elif kind == "minority_first":
            out.append(sorted(range(len(V)), key=lambda j: (n[V[j]], j)))
        elif kind == "oracle":
            out.append(sorted(range(len(V)), key=lambda j: (V[j] != d.gold[i], j)))
        elif kind == "claim_cosine":
            out.append(list(np.argsort(-d.claim_cos[i], kind="stable")))
    return out


def cv_score(tv, cfg, folds):
    P = np.zeros((len(tv.y), len(CLASSES)))
    for a, b in folds:
        P[b] = fit_predict(tv.take(a), tv.take(b), cfg)
    return evaluate(orders_from_probs(P, tv), tv, dedupe=False)[K]


# ============================================================
# MAIN
# ============================================================
def run_lang(lang, args):
    spec = LANGS[lang]
    paths = {s: resolve(args.data_root, spec[s]) for s in ("train", "val", "test")}
    if not all(paths.values()):
        print(f"[{lang}] skipped, missing files: {paths}")
        return None
    print(f"\n================ {lang} ================", flush=True)
    S = {s: Data(load_json(p)) for s, p in paths.items()}
    for s in S:
        print(f"  {s:5s}: {len(S[s].y)} claims | labels {dict(Counter(S[s].gold))}")

    emb_dir = dict(e.split("=", 1) for e in args.emb).get(lang)
    use_emb = use_group = False
    if emb_dir and not args.no_emb:
        pats = spec["emb"] if isinstance(spec["emb"], tuple) else (spec["emb"],)
        npz = {s: next((p for p in (resolve(emb_dir, q.format(split=s)) for q in pats) if p), None) for s in S}
        if all(npz.values()):
            for s in S:
                print(f"  reading {npz[s]}", flush=True)
                read_npz(npz[s], S[s])
            use_group = all(S[s].has_traces for s in S)
            finish_embedding_features(S, args.pca_dim)
            use_emb = True
        else:
            print(f"  embeddings not found: {npz}")
    print("  feature blocks: counts, memory" + (", claim, knn" if use_emb else "") + (", group" if use_group else ""))

    tv = concat(S["train"], S["val"])
    te = S["test"]

    # ---- block / C selection by 5-fold CV Recall@5 on TRAIN+VAL ----
    cands = [dict(claim=False, group=False, knn=False, C=1.0)]
    if use_emb:
        for claim in (False, True):
            for group in ((False, True) if use_group else (False,)):
                for knn in (False, True):
                    if claim or group or knn:
                        cands += [dict(claim=claim, group=group, knn=knn, C=C) for C in (0.01, 0.1)]
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(tv.y)), np.maximum(tv.y, 0)))
    scores = []
    for cfg in cands:
        scores.append(cv_score(tv, cfg, folds))
        print(f"    CV R@5 {scores[-1]:.4f}  {cfg}", flush=True)
    top = max(scores)
    # Simplest candidate (fewest blocks) within --min-gain of the best CV score
    size = lambda c: c["claim"] + c["group"] + c["knn"]
    best = min((c for c, s in zip(cands, scores) if s >= top - args.min_gain), key=size)
    print(f"  selected {best} (CV {scores[cands.index(best)]:.4f}, best {top:.4f})")

    # ---- final fit on TRAIN+VAL -> TEST ----
    P = fit_predict(tv, te, best)
    methods = {m: baseline_orders(te, m) for m in ("random", "majority_first", "minority_first", "oracle")}
    if use_emb:
        methods["claim_cosine (old)"] = baseline_orders(te, "claim_cosine")
    methods["p/n counts"] = orders_from_probs(fit_predict(tv, te, cands[0]), te)
    methods["p/n selected"] = orders_from_probs(P, te)
    final = "p/n selected"
    if args.shift_clip > 0:
        P = fit_predict(tv, te, best, weights=shift_weights(tv, te, args.shift_clip))
        final = "p/n selected + shift"
        methods[final] = orders_from_probs(P, te)

    results = {m: (evaluate(o, te, dedupe=False), evaluate(o, te, dedupe=True)) for m, o in methods.items()}
    write_predictions(os.path.join(args.out_dir, f"{lang}_test_predictions.json"), te, P, methods[final])

    out_csv = os.path.join(args.out_dir, f"{lang}_results.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method"] + [f"R@{k} allrows" for k in range(1, K + 1)] + ["R@5 dedup (scorer.py)"])
        for m, (a, d) in results.items():
            w.writerow([m] + [round(a[k], 4) for k in range(1, K + 1)] + [round(d[K], 4)])
    print(f"\n  TEST {'method':24s} R@5 all rows   R@5 dedup (scorer.py)")
    for m, (a, d) in results.items():
        print(f"       {m:24s} {a[K]:.4f}         {d[K]:.4f}")
    print(f"  final ranking = '{final}' -> {out_csv}")
    return {m: (a[K], d[K]) for m, (a, d) in results.items()}


def write_predictions(path, d, P, orders):
    out = []
    for i, e in enumerate(d.items):
        score = [0.0] * len(d.verdicts[i])
        for r, j in enumerate(orders[i]):
            score[j] = float(len(orders[i]) - r)
        out.append({
            "query_id": d.qid[i],
            "Claim": e["claim"],
            "Label": e.get("label", e.get("Label")),
            # Most probable label, NOT the top-ranked trace's verdict (the p/n rule often puts a
            # small minority group first on purpose).
            "Verdict_BoN": CLASSES[int(np.argmax(P[i]))].capitalize(),
            "BoN_Verdict_list": e["Verdict_list"],
            "score_list": score,
            "label_probs": {c.capitalize(): round(float(p), 5) for c, p in zip(CLASSES, P[i])},
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="all", help="EN | AR | ES | all")
    ap.add_argument("--data-root", default="/kaggle/input/datasets/d202511054/clef-2026-checkthat-task-2-data")
    ap.add_argument("--emb", action="append", default=[], help="LANG=<folder with that language's npz files>")
    ap.add_argument("--no-emb", action="store_true", help="counts + memory only")
    ap.add_argument("--pca-dim", type=int, default=32)
    ap.add_argument("--min-gain", type=float, default=0.001, help="CV R@5 margin an extra feature block must win by")
    ap.add_argument("--shift-clip", type=float, default=5.0, help="covariate-shift weight clip; 0 disables")
    ap.add_argument("--out-dir", default="/kaggle/working/trace_ranking")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    summary = {}
    for lang in (LANGS if args.lang == "all" else [args.lang.upper()]):
        r = run_lang(lang, args)
        if r:
            summary[lang] = r
    print("\n==== SUMMARY: test Recall@5 (all rows / dedup) ====")
    for lang, r in summary.items():
        print(f"  {lang}: " + " | ".join(f"{m} {a:.4f}/{d:.4f}" for m, (a, d) in r.items()
                                        if m in ("oracle", "p/n counts", "p/n selected", "p/n selected + shift")))
