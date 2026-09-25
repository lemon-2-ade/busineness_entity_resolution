"""OPTIONAL (GPU) stage: multilingual name embeddings as an extra matcher feature.

    python -m ber.embed --art artifacts --split train --device cuda
    python -m ber.embed --art artifacts --split test  --device cuda
then run train / predict with `--emb` to add the `emb_name_cos` feature.

Model: intfloat/multilingual-e5-small (MIT licence, 118M params, well under
the 8B limit).  It reads the *raw* business name, so it sees Devanagari /
Tamil / Telugu / Bengali / ... script directly and can relate
"लक्ष्मी प्रडिउसार प्राइवेट लिमिटेड" to "Lakshmi Producer Private Limited"
without our learned transliteration dictionary, and it has seen French.

Embeddings are L2-normalised float16 memmaps:
    artifacts/emb_<split>_s1.npy, artifacts/emb_<split>_s23.npy
(row order = <split>_s1 and concat(<split>_s2, <split>_s3), as everywhere).
This stage is not required: the default pipeline is CPU-only and its
reported validation numbers do not include this feature.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import polars as pl

MODEL = 'intfloat/multilingual-e5-small'


def encode_to_memmap(texts, path, model_name=MODEL, batch=1024, device='cuda', max_len=32):
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    if device.startswith('cuda'):
        model = model.half()
    dim = model.config.hidden_size
    out = np.lib.format.open_memmap(path, mode='w+', dtype=np.float16, shape=(len(texts), dim))
    # sort by length for efficient padding
    order = np.argsort([len(t) for t in texts], kind='stable')
    t0 = time.time()
    with torch.inference_mode():
        for i in range(0, len(texts), batch):
            idx = order[i:i + batch]
            enc = tok(['query: ' + texts[j] for j in idx], padding=True, truncation=True, max_length=max_len,
                      return_tensors='pt').to(device)
            h = model(**enc).last_hidden_state
            m = enc['attention_mask'].unsqueeze(-1).to(h.dtype)
            e = (h * m).sum(1) / m.sum(1)
            e = torch.nn.functional.normalize(e.float(), dim=-1)
            out[idx] = e.cpu().numpy().astype(np.float16)
            if (i // batch) % 500 == 0:
                print(f'  {i + len(idx)}/{len(texts)} {time.time() - t0:.0f}s', flush=True)
    out.flush()


def load(art, split):
    p1, p2 = f'{art}/emb_{split}_s1.npy', f'{art}/emb_{split}_s23.npy'
    if not (os.path.exists(p1) and os.path.exists(p2)):
        return None
    return np.load(p1, mmap_mode='r'), np.load(p2, mmap_mode='r')


def pair_cos(E, i1, i23, chunk=1_000_000):
    E1, E2 = E
    out = np.empty(len(i1), dtype=np.float32)
    for s in range(0, len(i1), chunk):
        a = np.asarray(E1[i1[s:s + chunk]], dtype=np.float32)
        b = np.asarray(E2[i23[s:s + chunk]], dtype=np.float32)
        out[s:s + chunk] = (a * b).sum(1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--art', default='artifacts')
    ap.add_argument('--split', default='test')
    ap.add_argument('--model', default=MODEL)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--batch', type=int, default=1024)
    a = ap.parse_args()
    s1 = pl.read_parquet(f'{a.art}/{a.split}_s1.parquet', columns=['business_name'])
    encode_to_memmap(s1['business_name'].to_list(), f'{a.art}/emb_{a.split}_s1.npy', a.model, a.batch, a.device)
    s23 = pl.concat([pl.read_parquet(f'{a.art}/{a.split}_s{i}.parquet', columns=['business_name']) for i in (2, 3)])
    encode_to_memmap(s23['business_name'].to_list(), f'{a.art}/emb_{a.split}_s23.npy', a.model, a.batch, a.device)


if __name__ == '__main__':
    main()
