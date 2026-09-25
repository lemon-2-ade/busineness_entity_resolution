# Business Entity Resolution — Amazon ML Challenge 2026

Blocking + gradient-boosted pairwise matcher + F0.5-aware post-processing for
linking Source-2 / Source-3 business records to the deduplicated Source-1
reference list. CPU-only by default; an optional GPU embedding feature is
included (see below).

The methodology write-up is
[`Documentation_template.md`](Documentation_template.md).

## Layout

```
src/ber/
  normalize.py    country-agnostic name/address normalisation, learned Indic-script dictionaries
  preprocess.py   stage 1: normalise every record once -> artifacts/<split>_s{1,2,3}.parquet
  blocking.py     3-view (name / address / joint) IDF-cosine top-k retrieval inside geo blocks
  candidates.py   stage 2: run blocking for a split -> artifacts/cands_<split>.parquet
  features.py     pairwise similarity features + competition ("context") features
  train.py        stage 3: LightGBM matcher, geo-held-out validation, post-processing tuning
  postprocess.py  exclusivity (one S1 per S2/S3 record) + threshold / expected-F0.5 selection
  predict.py      stage 4: score test candidates, write output/*.tsv
  metrics.py      macro F0.5 exactly as the leaderboard defines it
  embed.py        OPTIONAL: multilingual-e5-small name embeddings (GPU) -> extra feature
scripts/
  eda.py             statistics that drove the design
  eval_blocking.py   blocking recall / volume on a train sample
```

## Reproduce end-to-end

Data is expected at `data/{train,test}/*.tsv` (a symlink to the challenge
`dataset/` folder works). All commands run from this directory.

```bash
pip install -r requirements.txt
export PYTHONPATH=src MALLOC_ARENA_MAX=2

python -m ber.preprocess --data data --out artifacts          # ~10 min (train+test)
python -m ber.candidates --art artifacts --split train        # ~30 min on 2 cores
python -m ber.candidates --art artifacts --split test         # ~21 min on 2 cores
python -m ber.train      --art artifacts --data data          # ~55 min; prints validation F0.5
python -m ber.predict    --art artifacts --out output         # ~2 h on 2 cores
python scripts/check_outputs.py --out output --test-dir data/test   # quick self-check
python3 path/to/student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv \
    --test-dir data/test
```

Timings are for a 2-vCPU / 8 GB machine; every stage is multi-threaded
(`--threads` / `--workers`, default = all cores) and memory-bounded
(peak ≈ 6 GB), so a bigger box is proportionally faster.

### Optional: GPU name embeddings

```bash
pip install -r requirements-gpu.txt
python -m ber.embed --art artifacts --split train --device cuda
python -m ber.embed --art artifacts --split test  --device cuda
python -m ber.train --art artifacts --data data --emb     # adds feature `emb_name_cos`
python -m ber.predict --art artifacts --out output         # picks it up from the model config
```

Model: `intfloat/multilingual-e5-small` (MIT, 118M parameters). This path was
not validated in the development environment (no GPU); compare its printed
validation F0.5 with the CPU run and keep whichever is higher.

## Rules compliance

* No external data, APIs or lookups: every dictionary is either generic
  language knowledge (abbreviations, legal forms, state names) or learned from
  the training pairs (Indic-script token map). Test-set statistics used
  (IDF, name frequency) are unsupervised.
* `country` is treated as an open set of strings — nothing is keyed on
  US/India, France flows through the same code.
* Final model: LightGBM (MIT). The optional embedding model is MIT, 118M params.
