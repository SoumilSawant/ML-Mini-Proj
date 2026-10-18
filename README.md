# Logion: Machine Learning for Greek Philology

> **Status:** reproducible research infrastructure and prototype, not a full
> reproduction. The paper's Psellus data and compatible fine-tuned checkpoints
> are not distributed here.

A production-oriented implementation of the **Logion** framework described in
Cowen-Breen, Brooks, Haubold & Graziosi, *Logion: Machine Learning for Greek
Philology* (arXiv:2305.01099). Logion assists philologists working on premodern
Greek texts by

1. **flagging likely scribal errors** (out-of-distribution detection on a
   fine-tuned BERT model), and
2. **reconstructing gaps** caused by material deterioration, given only the
   gap's character length.

## Installation

```bash
pip install -r requirements.txt
```

Only `torch` + `numpy` are needed to use the pure algorithms; `transformers` is
required when loading real Hugging Face BERT models.

## Package layout

```
logion/
├── constants.py      # itacized vowels, feature layout (78 = 7x10 + 8), special tokens
├── utils.py          # scribal_distance (modified Levenshtein), Greek folding
├── features.py       # 78-dim FCNN features: 7x10 beam probs + char one-hot
├── models.py         # GapLengthEstimator (1-hidden-layer FCNN) + training loop
├── decoding.py       # MaskedSpanBeamSearch (iterative multi-mask beam search)
├── scribal_error.py  # ScribalErrorDetector
├── gap_filler.py     # GapFiller
└── config.py         # dataclass configuration
```

## 1. Scribal error detection

```python
from logion import ScribalErrorDetector, ScribalErrorConfig

detector = ScribalErrorDetector.from_pretrained(
    "path/to/psellus-finetuned-bert",
    config=ScribalErrorConfig(confidence_threshold=0.5, max_scribal_distance=3.0),
)

tokens = ["οὗτος", "δὶς", "βασιλεύσας", "ηὔχετο", "…"]
metrics = detector.compute_metrics(tokens, target_idx=1)
print(metrics.chance, metrics.confidence)          # p(w_i | w_-i), max_w p(w | w_-i)

ratio = detector.chance_confidence_ratio(tokens, target_idx=1, k=1)
flags  = detector.detect("οὗτος δὶς βασιλεύσας ηὔχετο …", mode="threshold")
```

`threshold_filter` flags words with `confidence >= 0.5` **and** scribal
distance `< 3`, then ranks them by increasing chance (paper, 2.3.3).

`scribal_distance` is a Wagner–Fischer edit distance where substitutions
*between the itacized vowels* `ι`, `υ`, `η` cost 0.5 instead of 1.0, with
accents folded and final sigma normalised first.

## 2. Gap filling

```python
from logion import GapFiller, GapFillerConfig

filler = GapFiller.from_pretrained(
    tokenizer,
    model_paths={2: "…/bert-mask-2", 3: "…/bert-mask-3",
                 4: "…/bert-mask-4", 5: "…/bert-mask-5"},
    base_model_path="…/psellus-finetuned-bert",   # used for single-mask searches
    estimator_path="…/length_estimator.pt",
    config=GapFillerConfig(beam_size=10, top_k=10),
)

text = "Οὐκ οἶδα ??? πότερον"          # '???' marks a gap of 4 characters
suggestions = filler.fill_gap(text, n_chars=4)
for s in suggestions[:5]:
    print(f"{s.text!r:>15}  toks={s.num_tokens}  P={s.score:.2e}")
```

`fill_gap` performs `(n // 2 + 2)` beam searches (one per candidate token
count, using the 2/3/4/5-mask fine-tuned models), computes
`P(gap_toks = k | context)` with the FCNN, and returns spans whose character
length is exactly `n`, ranked by `P(s | c) * P(gap_toks = toks(s) | c)`.

### Training the gap-length estimator

```python
from logion.features import build_gap_features
from logion.models import GapLengthEstimator, train_gap_length_estimator

# For each artificial gap (paper, 3.1.1): mask words until (3..10) characters
# are covered, run the (n//2+2) beam searches, store the top-10 probabilities
# per token length, and label with the true number of tokens.
features = [build_gap_features(beam_probs, n_chars) for beam_probs, n_chars in dataset]
labels   = [gap_toks - 1 for _, gap_toks in dataset]

estimator = GapLengthEstimator(hidden_dim=64)
train_gap_length_estimator(estimator, features, labels, epochs=50)
estimator.save("length_estimator.pt")
```

## Running the demo (no fine-tuned checkpoints required)

The paper's internal checkpoints are not public, but the full pipeline can be
exercised end-to-end with a publicly available Ancient-Greek masked-LM:

```bash
pip install -r requirements.txt
python run_demo.py --examples 8 --epochs 60 --beam-size 3 --top-k 3
```

The script

1. loads `wantuta/bert_ancient_greek_mlm` and uses it for every role
   (base model **and** the 2/3/4/5 consecutive-mask models),
2. generates artificial gaps (3–10 characters) from a tiny built-in corpus,
3. runs the beam searches, trains the 78→7 FCNN, and saves it to
   `artifacts/length_estimator.pt`,
4. evaluates `fill_gap` on held-out gaps and runs `ScribalErrorDetector`
   (threshold + ratio modes).

Expect only modest accuracy here: the demo model is not the Psellus-fine-tuned
BERT used in the paper, and the estimator is trained on a handful of examples.
It exists to prove the machinery works.

## Testing

The unit tests use tiny fake models/tokenizers, so they run without downloading
any BERT checkpoint:

```bash
python -m pytest
```

The suite is offline and uses fake tokenizers/models. It covers persistence,
model-size selection, exact character filtering, invalid inputs, multi-token
scoring, artificial gaps, and metric calculations.

## Data preparation and training

Datasets are deliberately external. Provide a UTF-8 text file with one passage
per line; do not commit TLG, Psellus, or other restricted material.

```bash
python scripts/prepare_gaps.py corpus.txt gaps.jsonl --count 100 --seed 2026
```

The reusable `logion.data`, `logion.pipeline`, and `logion.models` APIs create
artificial gaps, generate 78-dimensional FCNN features, train the estimator,
and save it. The `scripts/train_length_estimator.py` CLI now makes a seeded
train/validation split, reports held-out top-1 gap-token-count accuracy, and
writes a `.metrics.json` sidecar. If JSONL feature records include `source_id`,
the split is grouped by source so examples from the same source cannot leak
across train and validation. Without `source_id`, splitting is per record and
may overstate generalization when records share a source.

```bash
python scripts/train_length_estimator.py features.jsonl artifacts/length_estimator.pt \
  --epochs 50 --validation-fraction 0.1 --seed 2026
```

Feature records must contain `features` (78 floats) and `label` (the paper's
`gap_toks - 1`, an integer from 0 to 6). Generate train and validation gaps
from separate corpus splits before extracting features for a paper-style
evaluation. The separate 2/3/4/5-mask model training remains a configurable
checkpoint/fine-tuning step because the paper's data and compute are not
available here.

## Evaluation

Use `logion.evaluation.ranking_metrics` and `grouped_accuracy` in experiment
scripts to calculate top-1/5/10, correction recovery, precision/recall, and
grouped gap accuracy. A missing exact-length suggestion is represented by
`None`, never counted as a hit. No dictionary or restricted word list is
bundled; provide one to an experiment when frequency filtering is required.

## Running the local web app

The project includes a local browser interface for gap reconstruction and
scribal-error detection. It loads the public Ancient Greek masked-language
model once, then serves the UI at `http://127.0.0.1:8000`.

```bash
python web_app.py --artifact artifacts/length_estimator.pt
```

Open `http://127.0.0.1:8000` in a browser. The first startup may take a few
seconds while the model is loaded from Hugging Face. Use `Ctrl+C` to stop the
server.

## Limitations and reproducibility

The public `wantuta/bert_ancient_greek_mlm` demo is not the paper's Psellus-
fine-tuned model. Tokenizer vocabulary, checkpoint versions, corpus licenses,
random seeds, hardware, and beam-search settings affect results. Demo held-out
numbers are toy measurements only; no paper accuracy is claimed. A paper-style
run requires compatible checkpoints, the licensed corpus, a documented split,
and recorded software/configuration versions.

## Notable design points

* **`MaskedSpanBeamSearch`** extends the token-level MLM distribution to a
  distribution over *sequences* via iterative left-to-right beam search,
  re-running the model each step with the previously chosen tokens written back
  (paper 2.2 / 3.1.1).
* **Feature layout** follows the paper exactly: input dimension 78 = flatten of
  `7 x 10` (top-10 probabilities of the beam search for `tokens = 1..7`) plus an
  8-dim one-hot encoding of `n_chars` (`3..10`); the FCNN has one hidden layer,
  7 outputs, and is trained with cross-entropy loss.
* Character lengths exclude punctuation (`span_character_length`), matching the
  paper's accounting.
