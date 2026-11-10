#!/usr/bin/env python3
"""Train a skip-gram Word2Vec model on text8 and export it for the static web demo.

End to end: download corpus -> train (gensim) -> sanity checks -> export
  word2vec/model/meta.json   (vocab in frequency order + metadata)
  word2vec/model/vectors.bin (raw int8, row-major, count * dim bytes)

Usage:
  pip install gensim numpy
  python3 training/train_word2vec.py [--data-dir /tmp/w2v] [--top 30000] [--epochs 5]
"""
import argparse
import datetime
import gzip
import json
import logging
import os
import shutil
import time
import urllib.request
import zipfile

import numpy as np
from gensim.models import Word2Vec
from gensim.models.word2vec import Text8Corpus

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, "word2vec", "model")

# text8 = first 10^8 bytes of a cleaned English Wikipedia dump (Matt Mahoney).
SOURCES = [
    ("http://mattmahoney.net/dc/text8.zip", "zip"),
    # Mirror used by gensim.downloader (same file, gzipped).
    ("https://github.com/RaRe-Technologies/gensim-data/releases/download/text8/text8.gz", "gz"),
]

PARAMS = dict(sg=1, vector_size=100, window=5, min_count=5, negative=10, epochs=5)
SEED = 42


def fetch_text8(data_dir):
    path = os.path.join(data_dir, "text8")
    if os.path.exists(path) and os.path.getsize(path) == 100_000_000:
        return path
    os.makedirs(data_dir, exist_ok=True)
    for url, kind in SOURCES:
        archive = os.path.join(data_dir, os.path.basename(url))
        try:
            print(f"Downloading {url} ...")
            urllib.request.urlretrieve(url, archive)
            if kind == "zip":
                with zipfile.ZipFile(archive) as z, z.open("text8") as src, open(path, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            else:
                with gzip.open(archive, "rb") as src, open(path, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            if os.path.getsize(path) == 100_000_000:
                return path
            print("Unexpected size, trying next source")
        except Exception as e:  # noqa: BLE001
            print(f"  failed: {e}")
    raise SystemExit("Could not download text8 from any source")


def sanity(wv):
    for w in ["king", "france", "computer"]:
        print(f"\nmost_similar({w!r}):")
        for n, s in wv.most_similar(w, topn=10):
            print(f"  {n:15s} {s:.3f}")
    print("\nking - man + woman:")
    for n, s in wv.most_similar(positive=["king", "woman"], negative=["man"], topn=10):
        print(f"  {n:15s} {s:.3f}")


def export(model, top, corpus_desc):
    wv = model.wv
    words = wv.index_to_key[:top]  # gensim sorts vocab by descending frequency
    vecs = np.stack([wv[w] for w in words]).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    q = np.clip(np.round(vecs * 127), -127, 127).astype(np.int8)
    os.makedirs(OUT_DIR, exist_ok=True)
    q.tofile(os.path.join(OUT_DIR, "vectors.bin"))
    meta = {
        "dim": int(q.shape[1]),
        "count": len(words),
        "words": words,
        "corpus": corpus_desc,
        "params": PARAMS,
        "trained": datetime.date.today().isoformat(),
        "quantization": "int8, rows L2-normalized then multiplied by 127 and rounded",
    }
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, separators=(",", ":"))
    return words


def verify(wv, top):
    """Reload exported files and compare int8 neighbors with gensim's."""
    meta = json.load(open(os.path.join(OUT_DIR, "meta.json")))
    q = np.fromfile(os.path.join(OUT_DIR, "vectors.bin"), dtype=np.int8)
    q = q.reshape(meta["count"], meta["dim"]).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    idx = {w: i for i, w in enumerate(meta["words"])}
    vocab = set(meta["words"])
    print("\nVerification (exported int8 top-10 vs gensim top-10 restricted to export vocab):")
    for w in ["king", "france", "computer", "paris", "music"]:
        sims = q @ q[idx[w]]
        sims[idx[w]] = -1
        mine = [meta["words"][i] for i in np.argsort(-sims)[:10]]
        ref = [n for n, _ in wv.most_similar(w, topn=200) if n in vocab][:10]
        print(f"  {w:10s} overlap {len(set(mine) & set(ref))}/10  {mine}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/tmp/w2v-data")
    ap.add_argument("--top", type=int, default=30000)
    ap.add_argument("--epochs", type=int, default=PARAMS["epochs"])
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    PARAMS["epochs"] = args.epochs

    logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)
    corpus_path = fetch_text8(args.data_dir)

    t0 = time.time()
    model = Word2Vec(
        Text8Corpus(corpus_path), workers=args.workers, seed=SEED, **PARAMS
    )
    elapsed = time.time() - t0
    print(f"\nTraining took {elapsed / 60:.1f} min; full vocab {len(model.wv)} words")

    sanity(model.wv)
    corpus_desc = (
        "text8: first 100M characters of a cleaned English Wikipedia dump "
        "(Matt Mahoney, ~17M tokens, lowercase a-z only)"
    )
    words = export(model, args.top, corpus_desc)
    print(f"\nExported {len(words)} words to {OUT_DIR}")
    verify(model.wv, args.top)


if __name__ == "__main__":
    main()
