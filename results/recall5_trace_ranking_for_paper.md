# Trace ranking for Recall@5 — method, protocol and results (for the paper)

Context for the writer: this describes sub-problem B of CLEF CheckThat! 2026 Task 2 (rank the
15–20 LLM reasoning traces of each numerical claim; metric Recall@5) for English (EN), Spanish (ES)
and Arabic (AR). Every number below comes from our own runs with the code in
`task 2/trace_ranking/rank_traces.py`. The ranking is independent of our verdict classifier
(the MLP used for Macro-F1).

---

## 1. Metric and the key observation

Official definition (from the organisers' `scorer.py`): for claim *q* with gold label *y*,

  Recall@5(q) = (# relevant traces in the top 5) / (# relevant traces among all traces),

where a trace is **relevant iff its verdict equals the gold label**; if no trace is relevant,
Recall@5(q) = 0. The reported score is the mean over claims.

**Observation.** Relevance depends only on a trace's verdict, so all traces with the same verdict are
interchangeable. Let n_c be the number of traces with verdict c and k_c the number of them placed in
the top 5. Then Recall@5(q) = k_y / n_y. For a probability p_c = P(y = c | claim, traces), the
expected Recall@5 of a ranking is

  E[Recall@5] = Σ_c p_c · k_c / n_c,  subject to Σ_c k_c = 5 and 0 ≤ k_c ≤ n_c.

This is linear in k, so it is maximised exactly by a greedy rule:

> **Rank verdict groups by p_c / n_c** (highest first) and list all traces of a group together.

**Proposition (optimal ranking).** Given calibrated label probabilities p, ranking verdict groups by
p_c / n_c maximises expected Recall@5; the order of traces *within* a group does not affect the metric.

Consequences worth stating in the paper:
- Small verdict groups are promoted: one top-5 slot in a group of 2 buys 1/2 recall, in a group of 18
  only 1/18. The optimal ranking therefore often hedges by placing minority-verdict traces first.
- Per-trace scoring methods (claim–trace cosine similarity, cross-encoder rerankers such as
  Qwen3-Reranker, reward models / Best-of-N verifiers) rank by each trace's own relevance. That is the
  probability ranking principle, which is known to be suboptimal when relevance is correlated
  (Gordon & Lenk 1992; Chen & Karger 2006; Wang & Zhu 2009). Here relevance is perfectly correlated
  within a verdict group, so these methods cannot reach the optimum.
- The whole problem reduces to estimating well-calibrated p(y | claim) per claim. This is the
  plug-in approach to Bayes-optimal ranking (cf. Cossock & Zhang 2008; Xia et al. top-k consistency).
- The top-ranked trace is often NOT the predicted verdict; ranking and verdict prediction are
  decoupled.

**Ceiling ("best possible").** Even knowing the gold label, the best achievable per-claim value is
min(5, n_y)/n_y (0 if n_y = 0). Because most claims have all 20 traces agreeing with the gold
label (capped at 5/20 = 0.25) and some have none (0), the ceiling is low: 0.2822 (EN), 0.3224 (ES),
0.3426 (AR) on test when every row is scored.

---

## 2. Estimating p(y | claim)

Multinomial logistic regression (scikit-learn, standardised features) on the concatenation of
feature blocks:

| Block | Features | Dim |
|---|---|---|
| counts (always) | for each verdict c ∈ {True, False, Conflicting}: share n_c/N, log(1+n_c), indicator n_c = 0 | 9 |
| claim | PCA (32 dims) of the Qwen3-Embedding-8B claim embedding | 32 |
| group | for each verdict group: cosine(claim, group centroid), group cohesion (norm of the mean of L2-normalised trace embeddings), pairwise centroid cosines, PCA (32) of each centroid | 105 |
| knn | 10 nearest training claims by claim-embedding cosine: top-1 and mean similarity, label distribution weighted by sim⁴ | 5 |

Two further components:

- **Exact claim memory (always on).** If a test claim's text (lower-cased, whitespace-normalised)
  occurs in train+val, set p = 0.9 · (label distribution of those training claims) + 0.1 · p_model.
  Motivation: 192 of 395 scored AR test claims and 278 of 1,135 ES test claims occur verbatim in
  train/val, with identical labels in 100% of cases. EN test has no such overlap (and only 3 of 1,708
  EN test claims match CLEF 2025 claims). This is legitimate (training labels only) but the overlap
  must be reported.
- **Covariate-shift weighting.** A domain classifier (logistic regression, C = 0.1) separates
  train+val claims from the unlabelled test claims using the counts and claim blocks; training
  examples are weighted by the clipped odds p(test|x)/p(train|x) (clip at 5, normalised to mean 1),
  i.e. standard importance weighting (Shimodaira 2000). No test labels are used. Motivation: English
  test traces are far more reliable than train traces (per-trace accuracy 0.764 vs 0.478; when ≥95% of
  traces agree, the majority is correct 87% of the time on test vs 53% on train), so an unweighted model
  hedges too much.

Embeddings: Qwen3-Embedding-8B, last-token pooling, L2-normalised (claim = row 0, each non-empty
trace = one row). AR and ES use per-trace embeddings. For the reported EN numbers only the claim
embedding was available (no per-trace EN embeddings), so the group block was not available for EN.

---

## 3. Experimental protocol

- Splits (claims): EN 6,400 / 1,600 / 2,558 (train/val/test); ES 2,246 / 562 / 1,164;
  AR 2,608 / 652 / 511. Arabic gold labels contain no Conflicting instances.
- Model selection: 5-fold stratified cross-validation on train+val, scoring every combination of the
  optional blocks {claim, group, knn} × C ∈ {0.01, 0.1} (counts-only uses C = 1) by Recall@5. The
  simplest configuration within 0.001 of the best CV score is chosen. Test labels are never used
  for selection.
- Final model: refit on train+val with the selected configuration, applied once to test.
- Selected configurations: EN = counts + claim + knn, C = 0.01 (CV 0.2967);
  AR = counts + knn, C = 0.01 (CV 0.2713); ES = counts + knn, C = 0.01 (CV 0.2888).
  The group block was never selected.
- Scoring: Recall@5 averaged over **all test rows**. The provided `scorer.py` instead keeps only the
  first row per claim text (EN 2,558 → 1,708, AR 511 → 395, ES 1,164 → 1,135 rows); those values are
  also given below. The #1 leaderboard Arabic value (0.3406) exceeds the deduplicated ceiling (0.3198),
  so the leaderboard evidently scores all rows; comparisons with the leaderboard therefore use all rows.

---

## 4. Results (test Recall@5)

### 4.1 Headline

| Language | Before | Now (final) | #1 on leaderboard | Best possible |
|---|---|---|---|---|
| English | 0.2522 | **0.2552** | 0.2421 | 0.2822 |
| Spanish | 0.3153 | **0.3155** | 0.2946 | 0.3224 |
| Arabic | 0.3320 | **0.3419** | 0.3406 | 0.3426 |

- "Before" = our previous version of the same ranker: p/n rule with counts + exact memory (+ embedding
  features chosen on the validation split only, fit on train). EN had no embeddings, so it was counts-only.
- "Now" = final system: blocks selected by 5-fold CV on train+val, refit on train+val, plus
  covariate-shift weighting.
- "Best possible" = oracle with the gold label known.
- Share of the ceiling reached: EN 90.4%, ES 97.9%, AR 99.8%.

### 4.2 Ablation and baselines (test; all rows / deduplicated as in scorer.py)

| Method | EN | ES | AR |
|---|---|---|---|
| Random order | 0.2138 / 0.2154 | 0.2170 / 0.2170 | 0.2156 / 0.2103 |
| Majority verdict first | 0.2043 / 0.2045 | 0.1972 / 0.1967 | 0.1967 / 0.1950 |
| Minority verdict first | 0.2463 / 0.2514 | 0.2762 / 0.2779 | 0.2748 / 0.2615 |
| Claim–trace cosine (Qwen3-Emb-8B) | n/a* | 0.2161 / 0.2161 | 0.2473 / 0.2399 |
| p/n, counts only | 0.2522 / 0.2612 | 0.2834 / – | 0.3220 / – |
| p/n, counts + exact memory | 0.2522 / 0.2612 | 0.3134 / 0.3160 | 0.3250 / 0.3073 |
| p/n, + CV-selected blocks | 0.2527 / 0.2608 | 0.3158 / 0.3185 | 0.3399 / 0.3165 |
| **p/n, + covariate-shift weighting (final)** | **0.2552 / 0.2621** | **0.3155 / 0.3182** | **0.3419 / 0.3190** |
| Oracle (gold label known) | 0.2822 / 0.2886 | 0.3224 / 0.3250 | 0.3426 / 0.3198 |

\* No per-trace EN embeddings were available, so EN cosine cannot be computed.
"–" = not recorded for that variant.

Additional reference points:
- EN validation, organisers' reward-model scores (`clef_dev_predictions.json`): 0.3123 vs 0.3224 for
  the counts-only p/n ranker (validation split, all rows).
- DS@GT ARC (arXiv 2607.25069), the only other published Task 2 system we found: EN 24.77 (LoRA-tuned
  LLM verifier + Best-of-N), AR 33.41 (AraBERT).

### 4.3 Where the remaining English gap comes from

The EN gap to the oracle (0.030 before shift weighting) decomposes as: 70% from over-hedging (the gold
label was the majority verdict but minority traces were placed in the top 5), 30% from gold-minority
claims where too few minority traces were promoted. Claims with unanimous verdicts or with no correct
trace cost nothing.

---

## 5. Negative results (useful for the discussion section)

- A stronger verdict classifier does not translate into Recall@5. Our MLP classifier is far more accurate
  on EN test (79% vs 70% for the counts model), yet plugging its probabilities into the p/n rule gives
  0.2498 vs 0.2522. Both have the same log-loss (0.652 vs 0.653), and the allocation changes on only
  142 of 2,558 claims.
- Evidence-quality features (number/length of evidence snippets, number and word overlap with the claim,
  fact-check vocabulary) hurt EN test (0.239–0.244): test evidence is out of distribution (≈6.1 snippets
  per claim vs 2.8 in train).
- Unsupervised label-shift adaptation on the unlabelled test set (Saerens EM prior correction,
  Dirichlet-multinomial mixture EM) degenerated and lowered Recall@5 in every language.
- Pooling the verdicts of duplicated EN test rows, gradient-boosted trees and interaction features gave
  no gain.
- Fine-tuning Qwen3-Embedding-8B on claim + evidence → label was judged not worthwhile: the achievable
  head-room is ≤ 0.03–0.04 Recall@5 and, as shown above, higher label accuracy does not transfer to
  the metric.

---

## 6. Caveats the paper must state

1. **Covariate-shift weighting was enabled after observing test results** (it cannot be selected by
   cross-validation because the shift does not exist inside train+val). Report both "+ CV-selected
   blocks" and "+ shift weighting". It helps EN (+0.0025) and AR (+0.0020) and is neutral on ES
   (−0.0003).
2. **The Arabic margin over #1 (0.3419 vs 0.3406) is 0.0013**, within noise for 511 test rows.
3. **Train/test overlap in AR and ES.** Exact claim memory exploits verbatim duplicates between
   train/val and test (AR 49%, ES 24% of scored test claims). Report it, and the ablation without it.
4. **Two scoring conventions.** All-rows (used for the leaderboard comparison) vs the organisers'
   `scorer.py`, which drops repeated claims. Always state which one a number uses.
5. English was run with claim embeddings only; per-trace EN embeddings are not part of these results.

---

## 7. References to cite

- W. S. Cooper / S. Robertson — probability ranking principle; M. D. Gordon & P. Lenk (1992), "When is the
  probability ranking principle suboptimal?", JASIS.
- H. Chen & D. R. Karger (2006), "Less is more: probabilistic models for retrieving fewer relevant
  documents", SIGIR.
- J. Wang & J. Zhu (2009), "Portfolio theory of information retrieval", SIGIR.
- D. Cossock & T. Zhang (2008), "Statistical analysis of Bayes optimal subset ranking", IEEE Trans. IT.
- F. Xia, T.-Y. Liu & H. Li (2009), "Top-k consistency of learning to rank methods", NeurIPS.
- H. Shimodaira (2000), "Improving predictive inference under covariate shift by weighting the
  log-likelihood function", J. Stat. Planning and Inference.
- M. Saerens, P. Latinne & C. Decaestecker (2002), "Adjusting the outputs of a classifier to new a
  priori probabilities", Neural Computation.
- DS@GT ARC at CheckThat! 2026, arXiv:2607.25069.
- CLEF-2026 CheckThat! Lab overview, arXiv:2602.09516.
