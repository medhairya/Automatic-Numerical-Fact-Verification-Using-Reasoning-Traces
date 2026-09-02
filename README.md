# Automatic Numerical Fact Verification using Reasoning Traces

<p align="center">

# Learning Robust Numerical Fact Verification through Large Language Model Reasoning Traces


## Highlights

- **134,554** dense semantic embeddings generated
- **6,469** semantically matched claims across CLEF CheckThat! 2025 & 2026
- **15–20** reasoning traces generated per claim
- **1024-dimensional** embedding representation
- Supervised learning with **MLP** and **XGBoost**
- Extensive **ablation studies**
- **Embedding geometry** and **clustering analysis**
- Future direction: **Learning-to-Rank Reasoning Traces**

---

# Overview

Automatic verification of **numerical claims** remains considerably more difficult than ordinary textual fact-checking because correctness depends on exact quantities, temporal grounding, comparisons, and arithmetic reasoning rather than semantic similarity alone.

This project studies whether **LLM-generated reasoning traces** can be used as a rich semantic representation for predicting the veracity of numerical claims.

Instead of relying only on retrieved evidence, we investigate the semantic structure of reasoning traces through embedding analysis, supervised learning, ablation experiments, and clustering analysis.

---

# Research Highlights

## Novel Contributions

- Uses **multiple reasoning traces** instead of a single explanation.
- Large-scale reasoning trace embedding corpus.
- Semantic overlap analysis between CLEF CheckThat! 2025 and 2026.
- Embedding-space geometric analysis using SVD and centroid distances.
- Comparison of MLP and XGBoost classifiers.
- Extensive feature ablation experiments.
- Investigation of verdict representation strategies.
- Foundation for future **reasoning trace ranking** research.

---

# Pipeline Architecture

```text
                            Numerical Claim
                                   │
                                   ▼
                  Multiple LLM Reasoning Traces
                    (15–20 traces per claim)
                                   │
                                   ▼
                     Dense Semantic Embeddings
                       (bge-large-en-v1.5)
                                   │
                                   ▼
          ┌──────────────────────────────────────────────┐
          │           Feature Engineering                │
          │----------------------------------------------│
          │ • Claim Embeddings                           │
          │ • Reasoning Trace Embeddings                 │
          │ • Attention Features                         │
          │ • Disagreement Features                      │
          │ • Verdict Representations                    │
          └──────────────────────────────────────────────┘
                                   │
                                   ▼
                   Supervised Learning Models
                 ┌────────────────────────────┐
                 │      MLP                   │
                 │      XGBoost               │
                 └────────────────────────────┘
                                   │
                                   ▼
               True / False / Conflicting Verdict
```

---

# Dataset

| Metric | Value |
|---------|------:|
| Total Claims | 10,558 |
| Common Claims | 6,469 |
| Reasoning Traces | 15–20 per claim |
| Trace Embeddings | 128,085 |
| Total Embeddings | 134,554 |
| Classes | True / False / Conflicting |

---

# Methodology

1. Semantic overlap analysis
2. Embedding generation
3. Embedding-space analysis
4. Supervised classification
5. Ablation study
6. Unsupervised clustering

---

# Models

| Category | Models |
|----------|--------|
| Embeddings | bge-large-en-v1.5, Qwen3-Embedding-8B, F2LLM-v2-4B |
| Classifiers | MLP, XGBoost |

---

# Repository Structure

```text
Automatic-Numerical-Fact-Verification/
│
├── data/
├── notebooks/
├── models/
├── results/
├── figures/
├── utils/
├── requirements.txt
└── README.md
```

---


# Key Findings

- Raw embeddings alone do not separate veracity classes well.
- XGBoost consistently performs competitively on reasoning-trace representations.
- Attention mechanisms improve reasoning aggregation.
- Verdict representations significantly influence downstream performance.
- Reasoning traces capture useful semantic information beyond the original claim.

---

# Future Work

- Learning-to-Rank reasoning traces
- Cross-encoder reranking
- Transformer aggregation
- Graph Neural Networks
- Retrieval-Augmented Reasoning
- End-to-End differentiable ranking

---

# Acknowledgements

- CLEF CheckThat! Lab
- QuanTemp Benchmark
- Dhirubhai Ambani University
