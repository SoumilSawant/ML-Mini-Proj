"""End-to-end run of the Logion framework with a real Ancient Greek BERT.

The paper's own fine-tuned checkpoints are not public, so this demo:

* loads ``wantuta/bert_ancient_greek_mlm`` (a proper Ancient-Greek masked-LM),
* reuses it for every role: base model and the 2/3/4/5 consecutive-mask models,
* generates artificial gaps from a tiny built-in corpus (paper, 3.1.1),
* trains the 78->7 gap-length FCNN on those examples,
* evaluates ``fill_gap`` on held-out gaps,
* runs ``ScribalErrorDetector`` (threshold + ratio modes) on a sentence and on
  an artificially corrupted word.

Usage
-----
    python run_demo.py --examples 8 --epochs 40 --beam-size 3 --top-k 3
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from pathlib import Path

from transformers import AutoModelForMaskedLM, AutoTokenizer

from logion import (
    GapFiller,
    GapFillerConfig,
    GapLengthEstimator,
    ScribalErrorConfig,
    ScribalErrorDetector,
    fold_greek,
    scribal_distance,
    span_character_length,
)
from logion.features import build_gap_features
from logion.data import read_corpus

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("logion-demo")

MODEL_NAME = "wantuta/bert_ancient_greek_mlm"
GAP_MARKER = "[MASK]"
ARTIFACTS = Path(__file__).parent / "artifacts"

# Public-domain premodern Greek passages used to create artificial gaps.
CORPUS = [
    "Οὗτος δὶς βασιλεύσας ἠύχετο καὶ τρὶς καὶ τετράκις· ἦ δὲ γάρ, φησι, μετὰ νέφος ὁ ἥλιος.",
    "Μῆνιν ἄειδε θεά Πηληϊάδεω Ἀχιλλῆος οὐλομένην, ἣ μυρί' Ἀχαιοῖς ἄλγε' ἔθηκε, πολλὰς δ' ἰφθίμους ψυχὰς Ἄϊδι προΐαψεν ἡρώων.",
    "Ἄνδρα μοι ἔννεπε, Μοῦσα, πολύτροπον, ὃς μάλα πολλὰ πλάγχθη, ἐπεὶ Τροίης ἱερὸν πτολίεθρον ἔπερσε.",
    "Πάντες ἄνθρωποι τοῦ εἰδέναι ὀρέγονται φύσει· σημεῖον δ' ἡ τῶν αἰσθήσεων ἀγάπησις.",
    "Οὐκ οἶδα πότερον ἐγὼ μὲν ὀρθῶς ἢ μὴ προΐσταμαι τῶν πραγμάτων.",
    "Ἐν ἀρχῇ ἦν ὁ λόγος, καὶ ὁ λόγος ἦν πρὸς τὸν θεόν, καὶ θεὸς ἦν ὁ λόγος.",
    "Σωκράτης δέ ἐστιν ἀνὴρ σοφός τε καὶ δίκαιος καὶ τῶν Ἀθηναίων ἅπασι γνώριμος.",
    "Ἀλλ' ὅτε δή ῥ' ἴκοντο καὶ ἀλλήλους ἴδον ὀφθαλμοῖσιν, ὣς ἄρα τῷ πρόσθεν δῖος Ἀγαμέμνων ἀνέστη.",
]


def split_words(sentence: str) -> list[str]:
    return sentence.split()


def char_count(word: str) -> int:
    return span_character_length(word)


def generate_gap(sentence: str, rng: random.Random) -> tuple[str, int, str] | None:
    """Mask consecutive words until 3-10 characters are covered (paper 3.1.1)."""

    words = split_words(sentence)
    if len(words) < 4:
        return None

    start = rng.randint(0, len(words) - 2)
    target = rng.randint(3, 10)
    chosen: list[str] = []
    total = 0
    for word in words[start:]:
        counted = char_count(word)
        if counted == 0:
            continue
        chosen.append(word)
        total += counted
        if total >= target:
            break

    if not (3 <= total <= 10) or len(chosen) >= 7:
        return None

    left, right = words[:start], words[start + len(chosen):]
    context = " ".join([*left, GAP_MARKER, *right]).strip()
    return context, total, " ".join(chosen)


def build_training_set(
    tokenizer, filler: GapFiller, num_examples: int, beam_size: int, top_k: int, seed: int,
    corpus: list[str] | None = None,
) -> tuple[list[list[float]], list[int]]:
    rng = random.Random(seed)
    features: list[list[float]] = []
    labels: list[int] = []
    made = 0
    attempts = 0

    while made < num_examples and attempts < num_examples * 50:
        attempts += 1
        sentence = rng.choice(corpus or CORPUS)
        gap = generate_gap(sentence, rng)
        if gap is None:
            continue
        context, n_chars, span = gap

        span_ids = tokenizer.encode(span, add_special_tokens=False)
        if not (1 <= len(span_ids) <= 7):
            continue

        probabilities, _ = filler.beam_search_probabilities(
            context, n_chars, beam_size=beam_size, top_k=top_k
        )
        if not probabilities:
            continue

        features.append(build_gap_features(probabilities, n_chars).tolist())
        labels.append(len(span_ids) - 1)
        made += 1
        logger.info(
            "example %02d: n_chars=%d toks=%d span=%r",
            made, n_chars, len(span_ids), span,
        )

    logger.info("built %d training examples (%d attempts)", made, attempts)
    return features, labels


def demo_gap_filling(filler: GapFiller, tokenizer, seed: int, beam_size: int, top_k: int,
                     corpus: list[str]) -> None:
    print("\n" + "=" * 70)
    print("GAP FILLING — held-out gaps")
    print("=" * 70)

    rng = random.Random(seed)
    tested = 0
    hits_top1 = hits_top5 = 0

    for sentence in corpus:
        if tested >= 3:
            break
        gap = generate_gap(sentence, rng)
        if gap is None:
            continue
        context, n_chars, span = gap
        if len(tokenizer.encode(span, add_special_tokens=False)) > 6:
            continue

        suggestions = filler.fill_gap(
            context, n_chars, beam_size=beam_size, top_k=top_k, num_results=10
        )
        ground = fold_greek(span).replace(" ", "")
        matched = [
            i for i, s in enumerate(suggestions)
            if fold_greek(s.text).replace(" ", "") == ground
        ]
        rank = matched[0] + 1 if matched else None
        if rank == 1:
            hits_top1 += 1
        if rank is not None and rank <= 5:
            hits_top5 += 1

        tested += 1
        print(f"\ngap context : {context}")
        print(f"ground truth: {span!r}  (n_chars={n_chars}, toks={len(tokenizer.encode(span, add_special_tokens=False))})")
        if not suggestions:
            print("  (no exact character-length match found)")
            continue
        for i, suggestion in enumerate(suggestions[:5], start=1):
            marker = " <- MATCH" if rank == i else ""
            print(f"  top{i}: {suggestion.text!r:>14} toks={suggestion.num_tokens} "
                  f"score={suggestion.score:.2e}{marker}")

    print(f"\nheld-out gap accuracy: top-1 {hits_top1}/{tested}, top-5 {hits_top5}/{tested}")


def demo_error_detection(detector: ScribalErrorDetector, tokenizer, beam_size: int) -> None:
    print("\n" + "=" * 70)
    print("SCRIBAL ERROR DETECTION")
    print("=" * 70)

    sentence = "Οὗτος δὶς βασιλεύσας ἠύχετο καὶ τρὶς καὶ τετράκις."

    print(f"\n-- threshold flags on: {sentence}")
    flags = detector.detect(sentence, mode="threshold", show_progress=False)
    for flag in flags[:6]:
        print(f"  pos {flag.position:2d} {flag.token!r:>12} "
              f"chance={flag.chance:.2e} conf={flag.confidence:.3f} "
              f"dist={flag.scribal_distance:.2f} -> {flag.suggestion!r}")

    print("\n-- ranking by chance-confidence ratio (k=1), worst first")
    ranked = detector.detect(sentence, mode="ratio", k=1)
    for metric in ranked[:5]:
        ratio = metric.chance_confidence_ratio
        print(f"  pos {metric.position:2d} {metric.token!r:>14} "
              f"ratio={ratio:.3f} chance={metric.chance:.2e} -> {metric.suggestion!r}")

    # Deliberately corrupt one word and re-rank.
    corrupted = sentence.replace("ἠύχετο", "ἠνχετο")
    print(f"\n-- same sentence with a corrupted word: {corrupted}")
    ranked = detector.detect(corrupted, mode="ratio", k=1)
    for metric in ranked[:5]:
        ratio = metric.chance_confidence_ratio
        print(f"  pos {metric.position:2d} {metric.token!r:>14} "
              f"ratio={ratio:.3f} chance={metric.chance:.2e} -> {metric.suggestion!r}")

    print("\n-- scribal distance sanity checks")
    for a, b in [("ι", "υ"), ("η", "ι"), ("α", "ε"), ("λόγος", "λόγος")]:
        print(f"  d({a!r}, {b!r}) = {scribal_distance(a, b):.2f}")


def main() -> None:
    # Windows consoles often default to cp1252, while Greek output is part of
    # the demo's purpose.  Do not let reporting fail after successful inference.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=int, default=8, help="training gaps to generate")
    parser.add_argument("--epochs", type=int, default=50, help="FCNN training epochs")
    parser.add_argument("--beam-size", type=int, default=3, help="training beam size")
    parser.add_argument("--top-k", type=int, default=3, help="training top-k")
    parser.add_argument("--eval-beam-size", type=int, default=10, help="evaluation beam size")
    parser.add_argument("--eval-top-k", type=int, default=10, help="evaluation top-k")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--model-name", default=MODEL_NAME, help="Hugging Face model or local checkpoint")
    parser.add_argument("--corpus", help="optional UTF-8 corpus (one passage per line)")
    args = parser.parse_args()

    t_start = time.time()

    corpus = read_corpus(args.corpus) if args.corpus else CORPUS
    print(f"\nLoading {args.model_name} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForMaskedLM.from_pretrained(args.model_name)
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"Could not load model {args.model_name!r}: {exc}") from exc
    model.eval()
    print(f"loaded model & tokenizer in {time.time() - t_start:.1f}s")

    config = GapFillerConfig(beam_size=args.beam_size, top_k=args.top_k)
    filler = GapFiller(
        {size: model for size in (2, 3, 4, 5)},
        tokenizer,
        base_model=model,
        device="cpu",
        config=config,
    )

    print(f"\nGenerating {args.examples} artificial gaps and beam probabilities ...")
    t0 = time.time()
    features, labels = build_training_set(
        tokenizer, filler, args.examples, args.beam_size, args.top_k, args.seed, corpus
    )
    print(f"feature generation took {time.time() - t0:.1f}s")

    print(f"\nTraining the gap-length FCNN ({args.epochs} epochs) ...")
    history = filler.fit_length_estimator(features, labels, epochs=args.epochs, verbose=True, seed=args.seed)
    print(f"final loss: {history[-1]:.4f} (first: {history[0]:.4f})")
    ARTIFACTS.mkdir(exist_ok=True)
    # Save the exact instance trained above.  A separate estimator here would
    # contain random, untrained weights (the original prototype bug).
    filler.length_estimator.save(str(ARTIFACTS / "length_estimator.pt"))
    reloaded = GapLengthEstimator(hidden_dim=config.hidden_dim, dropout=config.dropout)
    reloaded.load(str(ARTIFACTS / "length_estimator.pt"))
    print("verified saved estimator can be loaded")
    print(f"saved estimator -> {ARTIFACTS / 'length_estimator.pt'}")

    detector = ScribalErrorDetector(model, tokenizer, device="cpu", config=ScribalErrorConfig())

    demo_gap_filling(
        filler,
        tokenizer,
        seed=args.seed + 1,
        beam_size=args.eval_beam_size,
        top_k=args.eval_top_k, corpus=corpus,
    )
    demo_error_detection(detector, tokenizer, beam_size=args.eval_beam_size)

    print(f"\nTotal runtime: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
