"""Tokenizer construction: SentencePiece BPE, trained from scratch per language.

Stage 7 (spec §1.3 / plan §5). Two completely independent tokenizers, zero
shared state -- the vocabulary-difference assertion in
`assert_independence.py` is the structural check that this was actually
respected, not just claimed.

**Sentence granularity, and why it isn't "one document per line".** Feeding
SentencePiece one full document per line risks silent truncation --
`max_sentence_length` defaults to 4192 bytes, well under a typical Wikipedia
article or archive.org book chapter. Splitting on sentence-ending punctuation
(Devanagari/Bengali danda `।`, plus `.`/`!`/`?` for the residual Latin
punctuation that survives cleaning) keeps each training line well inside that
limit without needing a real sentence tokenizer -- the same "cheap,
interpretable, tuned by reading examples" posture as the language-ID function
words in `common/clean/filters.py`.

**Subsampling, not the full corpus.** `input_sentence_size` +
`shuffle_input_sentence=True` is SentencePiece's own answer to training on a
500M-token corpus without holding it all in memory during training --
documented in plan §5 as "the single setting that decides whether training
fits in memory." This module writes the full train-split text to a flat
file (disk is not the constraint, 151GB free) and lets SentencePiece do the
reservoir sampling internally, rather than reimplementing it.

**Trained on the TRAIN split only.** Fitting on val/test would leak split
boundaries into the vocabulary itself -- a subtler version of the same
leakage §4.6 already guards against for the corpus split.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import regex
import sentencepiece as spm

from common.io import Doc, iter_docs, enable_utf8_console

__all__ = [
    "TokenizerConfig", "extract_train_corpus", "train_one", "sweep",
    "measure", "select_best", "main",
]

# Danda (U+0964) is the Devanagari/Bengali-Assamese sentence terminator;
# double danda (U+0965) closes verse. `.`/`!`/`?` catch residual Latin
# punctuation (numbers, code-mixed fragments) that survives cleaning.
_SENTENCE_SPLIT_RE = regex.compile(r"(?<=[।॥.!?])\s+")

DEFAULT_VOCAB_SWEEP = (8000, 16000, 24000, 32000, 48000, 64000)


@dataclass
class TokenizerConfig:
    language: str
    character_coverage: float = 0.9995
    model_type: str = "bpe"
    input_sentence_size: int = 4_000_000
    shuffle_input_sentence: bool = True
    byte_fallback: bool = True


def _iter_split_docs(corpus_dir: Path | str, ids_path: Path | str) -> Iterator[Doc]:
    """Yield only the documents whose id is listed in `ids_path`.

    Used to measure tokenizer statistics on the val split specifically --
    measuring against the whole deduped corpus would silently include the
    train-split text the model was fit on, defeating the point of a
    held-out number.
    """
    ids_path = Path(ids_path)
    with ids_path.open("r", encoding="utf-8") as f:
        wanted = {line.strip() for line in f if line.strip()}
    for doc in iter_docs(corpus_dir):
        if doc.id in wanted:
            yield doc


def _split_sentences(text: str) -> Iterator[str]:
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        for sent in _SENTENCE_SPLIT_RE.split(line):
            sent = sent.strip()
            if sent:
                yield sent


def extract_train_corpus(
    language: str,
    train_ids_path: Path | str,
    corpus_dir: Path | str,
    out_path: Path | str,
) -> dict:
    """Write every train-split document's sentences to a flat UTF-8 text file.

    One sentence per line -- SentencePiece's expected input shape, and short
    enough per line to never hit `max_sentence_length` truncation regardless
    of how long the source document was.

    Returns counts (docs matched, sentences written) for a sanity check
    against the split's known doc count before the expensive sweep runs.
    """
    train_ids_path = Path(train_ids_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with train_ids_path.open("r", encoding="utf-8") as f:
        wanted = {line.strip() for line in f if line.strip()}

    docs_matched = 0
    sentences_written = 0
    t0 = time.time()

    with out_path.open("w", encoding="utf-8", newline="\n") as out:
        for doc in iter_docs(corpus_dir):
            if doc.id not in wanted:
                continue
            docs_matched += 1
            for sent in _split_sentences(doc.text):
                out.write(sent)
                out.write("\n")
                sentences_written += 1

    return {
        "language": language,
        "docs_wanted": len(wanted),
        "docs_matched": docs_matched,
        "sentences_written": sentences_written,
        "elapsed_seconds": round(time.time() - t0, 1),
        "out_path": str(out_path),
    }


def train_one(
    corpus_path: Path | str,
    model_prefix: Path | str,
    vocab_size: int,
    config: TokenizerConfig,
) -> None:
    """Train one SentencePiece model. Thin wrapper -- all the judgment calls
    (coverage, byte-fallback, subsampling) live in `TokenizerConfig`.
    """
    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type=config.model_type,
        character_coverage=config.character_coverage,
        input_sentence_size=config.input_sentence_size,
        shuffle_input_sentence=config.shuffle_input_sentence,
        byte_fallback=config.byte_fallback,
        # Explicit ids matching the standard LM convention; unk must exist,
        # bos/eos are used by the Phase 2 training loop, pad supports batching.
        unk_id=0, bos_id=1, eos_id=2, pad_id=3,
        train_extremely_large_corpus=False,
    )


def sweep(
    language: str,
    corpus_path: Path | str,
    out_dir: Path | str,
    vocab_sizes: tuple[int, ...] = DEFAULT_VOCAB_SWEEP,
    config: TokenizerConfig | None = None,
) -> dict:
    """Train one model per vocab size. Returns elapsed time per run --
    the sweep is a coarse decision (fertility/UNK on held-out text, done by
    `select_best`), not something to interleave per-run judgment into.
    """
    config = config or TokenizerConfig(language=language)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for v in vocab_sizes:
        prefix = out_dir / f"{language}_{config.model_type}_{v}"
        t0 = time.time()
        train_one(corpus_path, prefix, v, config)
        results[v] = {"elapsed_seconds": round(time.time() - t0, 1),
                       "model": str(prefix) + ".model"}
    return results


def measure(model_path: Path | str, held_out_docs: Iterator[Doc]) -> dict:
    """Fertility, bytes/char per token, byte-fallback rate, singleton share --
    the spec's required tokenizer statistics (§1.3), computed on held-out
    (val-split) text so the numbers describe generalisation, not fit.
    """
    sp = spm.SentencePieceProcessor(model_file=str(model_path))

    total_words = 0
    total_tokens = 0
    total_bytes = 0
    byte_fallback_tokens = 0
    token_counts: dict[int, int] = {}

    for doc in held_out_docs:
        words = doc.text.split()
        total_words += len(words)
        total_bytes += len(doc.text.encode("utf-8"))
        ids = sp.encode(doc.text, out_type=int)
        total_tokens += len(ids)
        for tid in ids:
            token_counts[tid] = token_counts.get(tid, 0) + 1
            piece = sp.id_to_piece(tid)
            if piece.startswith("<0x") and piece.endswith(">"):
                byte_fallback_tokens += 1

    singleton_tokens = sum(1 for c in token_counts.values() if c == 1)

    return {
        "model": str(model_path),
        "vocab_size": sp.vocab_size(),
        "fertility_tokens_per_word": total_tokens / total_words if total_words else None,
        "bytes_per_token": total_bytes / total_tokens if total_tokens else None,
        "byte_fallback_rate": byte_fallback_tokens / total_tokens if total_tokens else None,
        "distinct_tokens_used": len(token_counts),
        "singleton_token_share": singleton_tokens / len(token_counts) if token_counts else None,
        "held_out_words": total_words,
        "held_out_tokens": total_tokens,
    }


def select_best(measurements: list[dict]) -> dict:
    """Pick the vocab size minimizing fertility among candidates with a
    byte-fallback rate near zero (i.e. the vocab is actually covering the
    script, not leaning on byte fallback to paper over undertrained merges).

    This is a starting point, not a substitute for reading the numbers --
    plan §5 flags that Hindi and Assamese are expected to land on different
    optimal sizes, which is a finding to report, not an error to fix.
    """
    viable = [m for m in measurements if (m["byte_fallback_rate"] or 1) < 0.01]
    pool = viable or measurements
    return min(pool, key=lambda m: m["fertility_tokens_per_word"])


def main(argv: list[str] | None = None) -> int:
    """CLI. Example:
        python -m common.tokenization.tokenizer --lang hindi --extract --sweep --select
    """
    import argparse

    enable_utf8_console()
    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("--lang", required=True, choices=("hindi", "assamese"))
    p.add_argument("--extract", action="store_true", help="rebuild the flat train-corpus text file")
    p.add_argument("--sweep", action="store_true", help="run the vocab-size sweep")
    p.add_argument("--select", action="store_true", help="measure sweep results on val split and pick the best")
    p.add_argument("--vocab-sizes", default="", help="comma-separated override of the default sweep")
    args = p.parse_args(argv)

    lang = args.lang
    corpus_path = Path(f"{lang}/tokenizer/train_corpus.txt")
    out_dir = Path(f"{lang}/tokenizer")

    if args.extract:
        print(f"== extracting train corpus for {lang}")
        r = extract_train_corpus(lang, f"{lang}/splits/train.ids", f"data/{lang}/clean_deduped", corpus_path)
        print(f"   {r['docs_matched']:,}/{r['docs_wanted']:,} docs matched, "
              f"{r['sentences_written']:,} sentences written ({r['elapsed_seconds']}s)")

    if args.sweep:
        vocab_sizes = tuple(int(v) for v in args.vocab_sizes.split(",")) if args.vocab_sizes else DEFAULT_VOCAB_SWEEP
        print(f"== sweeping {lang}: {vocab_sizes}")
        results = sweep(lang, corpus_path, out_dir, vocab_sizes)
        for v, r in results.items():
            print(f"   vocab={v}: {r['elapsed_seconds']}s -> {r['model']}")
        (out_dir / "sweep_timing.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    if args.select:
        vocab_sizes = tuple(int(v) for v in args.vocab_sizes.split(",")) if args.vocab_sizes else DEFAULT_VOCAB_SWEEP
        print(f"== measuring {lang} sweep on val split")
        measurements = []
        for v in vocab_sizes:
            model = out_dir / f"{lang}_bpe_{v}.model"
            if not model.exists():
                print(f"   skip vocab={v}: model not found")
                continue
            m = measure(model, _iter_split_docs(f"data/{lang}/clean_deduped", f"{lang}/splits/val.ids"))
            measurements.append(m)
            print(f"   vocab={v}: fertility={m['fertility_tokens_per_word']:.3f} "
                  f"bytes/tok={m['bytes_per_token']:.2f} "
                  f"byte_fallback={m['byte_fallback_rate']:.4%}")
        best = select_best(measurements)
        print(f"== selected vocab={best['vocab_size']} (fertility={best['fertility_tokens_per_word']:.3f})")
        stats_path = Path(f"{lang}/stats/tokenizer_sweep.json")
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps({"measurements": measurements, "selected": best},
                                          indent=2, ensure_ascii=False), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
