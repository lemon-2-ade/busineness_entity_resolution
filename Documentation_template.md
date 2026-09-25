# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** [Date]

---

## 1. Executive Summary

The pipeline has four stages: country-agnostic normalisation (with a
native-script→English dictionary *learned from the training pairs*),
geo-partitioned three-view sparse blocking (name / address / joint IDF cosine),
a two-stage LightGBM matcher, and F0.5-aware post-processing. The key idea is
to model **competition**: every S2/S3 record belongs to at most one S1 entity,
so each candidate is scored against the other S1 entities retrieving the same
record, first in blocking-score space (stage 1) and then in probability space
(stage 2). Macro F0.5 on held-out *states* of the training data is **0.979**,
with a blocking recall ceiling of 98.3% at 24 candidates per S1.

---

## 2. Methodology

### 2.1 Problem Analysis

* 2.2M / 1.73M S1 entities (train / test) against about 10M S2+S3 records per
  split, so an all-pairs comparison is impossible.
* 5.6% of S1 are singletons; mean cluster size is 3.46.
* **Each S2/S3 record matches at most one S1** (7.64M pairs over 7.64M unique
  records). 26% of S2/S3 records match nothing: these are distractors and hard
  negatives (same name next door, a different business at the same address).
* True pairs always share the country, and share the canonical state 95% of
  the time. 4.7% of S2/S3 records have no state.
* **S1 names are highly non-unique**: only about 50–60% of core names are
  unique, versus 96% of addresses. The address is the primary key.
* Noise in names: legal-suffix swaps, honorifics, injected fillers
  (Center/Services/Partners), brackets and symbols, IDs and phone numbers, web
  domains, dba/formerly/aka aliases, unrelated alias names, word shuffles,
  typos, and native-script transliterations (about 23% of India S2 names).
* Noise in addresses: abbreviations, reordering, missing components, `NULL`,
  door/plot prefixes, injected numbers, leading zeros, and **truncated house
  numbers** (5004→004, 3833→383).
* The test set adds France (15% of test S1): no canonical states, and French
  legal forms and street types.

### 2.2 Solution Strategy

**Approach type:** Blocking + two-stage GBDT classifier + constrained assignment (hybrid).
**Core innovations:**

1. Geo-partitioned multi-view blocking with block-local IDF. Recall went from
   96.1% to 99.0% before truncation, and the work dropped about 20x.
2. Competition features on the full candidate graph (margin of this S1 over the
   best other S1 for the same record), plus a stage-2 re-scorer on
   stage-1 probability context.
3. A native-script transliteration dictionary learned by token alignment of
   training pairs (96.4% coverage of test native tokens).
4. Fuzzy house-number features that tell digit truncation (same house) apart
   from nearby numbers (different house).
5. Validation by held-out states, which is a closed world and mimics a new geography.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** within each (country, canonical state) block, plus a
  per-country pool of state-less S2/S3 records (countries without canonical
  states, such as France, form one block), three IDF-weighted cosine top-k
  searches using `sparse_dot_topn`:
  - name view: core/alias word tokens + char 3-grams of the compact name (top-15)
  - address view: core tokens + adjacent bigrams such as `14_watkins` (top-15)
  - joint view: the two concatenated (top-30); the residual pool uses 5/5/10.
  Tokens with in-block document frequency above 5,000 are ignored. The union is
  truncated to joint-rank ≤ 20 ∪ name-rank ≤ 5 ∪ address-rank ≤ 10.
- **Candidate pairs generated:** 41,644,908 for 1,732,544 test S1 (24.0 per
  S1; reduction ratio ≈ 1 − 41.6M / (1.73M × 10.0M) ≈ 99.9998%). Train:
  53.8M pairs (24.4 per S1).
- **How you ensured true matches were not lost:** recall was measured on
  training data at every design iteration: 94.9% →
  96.1% (joint view) → 99.0% (geo blocks) → 98.3% final (after the truncation
  and cap needed for runtime). Three complementary views mean an alias name is
  still caught by the address, and a missing address is still caught by the name.

---

## 4. Matching Model

**Features used (70 in stage 1, +9 in stage 2):**
- Name features: rapidfuzz ratio / token-sort / token-set / partial on the
  normalised and core names; compact-name ratio, Jaro-Winkler, partial and
  Levenshtein (domains, word splits); best-of main/alias name; consonant-skeleton
  similarity (transliteration); token Jaccard and coverage; first-token and
  prefix equality; legal-form agreement or conflict; domain / alias /
  native-script / ALL-CAPS flags; name frequency in the split.
- Address features: ratio / token-sort / token-set / partial-token-set on core
  address; word-only similarity; Jaccard and coverage; token counts; empty
  address; exact house-number agreement (any / first / Jaccard / subset /
  conflict); fuzzy house numbers (best / first / share explained / truncation).
- Other: the three blocking cosines and ranks; competition context (S1-side
  count, gap and relative score; S2/S3-side count, rank, gap and **margin over
  the best other S1**); source flag (S3). Stage 2 adds p, rank of p in the S1,
  best and other-best p in the S1, Σp, count of p > 0.5, best p of another S1
  for the same record, margin, and count of competing S1 with p > 0.1.
- Deliberately not used: country one-hot (open set), and state-agreement
  features (France has no states, so they would cause a train/test shift).

**Model type:** LightGBM binary classifiers (MIT). Stage 1: 127 leaves,
lr 0.05, 1,614 trees, trained on 6.1M pairs from 250k S1. Stage 2: 63 leaves,
1,101 trees, trained on a disjoint held-out block of states (177k S1, 4.3M pairs).
**Threshold selection method:** each S2/S3 record is kept only for its
best-scoring S1 (exclusivity), then a probability threshold of 0.70 is applied.
This was chosen on the validation states by macro F0.5, and compared against an
expected-F0.5-optimal top-k rule per entity. The two tied (0.9791 vs 0.9793),
so I kept the simpler and more singleton-conservative threshold.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro), validation (132k S1 in 24 held-out states):**

| Model | Macro F0.5 | Singleton F0.5 | Non-singleton F0.5 | Precision | Recall |
|---|---|---|---|---|---|
| v1 stage 1 (65 features), thr 0.70 | 0.9750 | 0.973 | 0.975 | 0.993 | 0.943 |
| v2 stage 1 (70 features), thr 0.70 | 0.9773 | 0.975 | 0.977 | 0.994 | 0.949 |
| **v2 stage 1 + stage 2, thr 0.70 (submitted)** | **0.9791** | 0.986 | 0.979 | 0.994 | 0.952 |

- **Common false positives (wrong merges):** the same or a similar name with
  the house number off by a few (207 vs 218 Povo Rd; F409 vs F416), which are
  deliberate hard negatives; and generic names paired with an S2 record whose
  address is empty.
- **Common false negatives (missed matches):** S2/S3 copies with an empty or
  heavily truncated address plus a generic name ("Regional Institute Co"); names
  replaced by an unrelated alias with only a partial address; native-script
  names with a partial address; and pairs lost in blocking (1.7% of pairs).

---

## 6. Conclusion

A carefully engineered classic ER stack beats brute force here. Blocking
inside geographic blocks with local IDF gave the largest single improvement.
Modelling *competition* between S1 entities for the same record gave the most
predictive features. A learned transliteration dictionary and fuzzy
house-number features dealt with the dataset's specific noise. Everything is
country-agnostic, so the unseen French slice goes through the same pipeline.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`:
```
src/ber/normalize.py   preprocess.py   blocking.py   candidates.py
        features.py    train.py        postprocess.py predict.py
        metrics.py     embed.py (optional GPU feature)
scripts/eda.py, scripts/eval_blocking.py
README.md, requirements.txt, requirements-gpu.txt
```
Entry points (run from the package root with `PYTHONPATH=src`):
```
python -m ber.preprocess --data data --out artifacts
python -m ber.candidates --split train ; python -m ber.candidates --split test
python -m ber.train
python -m ber.predict --out output     # -> output/matching_results.tsv, output/candidate_pairs.tsv
```

### B. Additional Results

Blocking design iterations (pair recall on a 50k-S1 train sample):

| Design | Recall | Candidates / S1 |
|---|---|---|
| name + address views, country-wide | 94.9% | 30 |
| + joint view | 96.1% | 42 |
| + geo-partitioning, local IDF | 99.0% | 54 |
| + rank truncation, df cap 5k (final) | 98.3% | 24 |

Post-processing on the validation fold: exclusivity + threshold 0.70 gives
0.9791; threshold 0.50 gives 0.9769; the expected-F top-k rule gives 0.9793;
threshold 0.5 without exclusivity gives 0.9768.

Test-set summary: 41,644,908 candidate pairs → 5,741,614 matches (3.31 per S1); 6.2% of S1 predicted singletons; France 3.05 matches/S1 with 8.1% empty vs India 3.35 / 5.9% and US 3.38 / 5.8%.
