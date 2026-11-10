# Word2Vec training

`train_word2vec.py` produces the embedding files used by the web demo in
`word2vec/model/`. It downloads the corpus, trains, prints sanity checks,
exports, and verifies the export, all in one run.

## Rerun

```bash
pip install gensim numpy
python3 training/train_word2vec.py --data-dir /tmp/w2v-data
```

Options: `--top 30000` (words exported), `--epochs 5`, `--workers 4`.
On 4 CPU cores training takes roughly 10 minutes. The corpus (~100 MB) is
stored in `--data-dir`; do not commit it.

## Corpus

**text8**: the first 100,000,000 characters of a cleaned English Wikipedia
dump (Matt Mahoney), about 17M tokens, lowercase letters only. Downloaded
from `mattmahoney.net/dc/text8.zip`, falling back to the identical file on
the gensim-data GitHub release (`text8.gz`).

## Parameters (gensim 4 `Word2Vec`)

| param | value |
|---|---|
| sg | 1 (skip-gram) |
| vector_size | 100 |
| window | 5 |
| min_count | 5 |
| negative | 10 |
| epochs | 5 |
| seed | 42 |

## Export format

- `word2vec/model/meta.json`: `{"dim", "count", "words", "corpus", "params",
  "trained", "quantization"}`. `words` holds the top-`count` words in
  descending corpus frequency; `words[i]` is row `i` of the matrix.
- `word2vec/model/vectors.bin`: raw signed int8, row-major, `count * dim`
  bytes. Each row is `round(127 * v / ||v||)`, clipped to [-127, 127], so
  `cosine(a, b) ≈ dot(a, b) / 127²` (re-normalizing the rows is slightly more
  accurate).

The script reloads both files with numpy at the end and checks that the
int8 nearest neighbours match gensim's.
