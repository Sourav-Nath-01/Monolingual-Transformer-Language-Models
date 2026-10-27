"""Pre-tokenise the cleaned corpus into flat ``uint16`` token arrays.

Phase 2 trains on Kaggle, not on the laptop that built the corpus, and that
constraint drives this module's existence. Shipping 6.4 GB of compressed JSONL
text to a Kaggle Dataset and re-tokenising it every session would waste most of
a 12-hour GPU slot before a single optimiser step ran. Tokenising once, here,
turns the corpus into ~2.4 GB of token ids that upload once and memory-map
instantly on the other side.

Three decisions worth stating, because each is a trade-off rather than an
obvious default:

**``uint16``, not ``int32`` or ``int64``.**
    Both tokenizers have a 16,000-entry vocabulary (Phase 1 report section 8.1),
    and 16,000 < 65,536, so every id fits in two bytes. That halves the file
    against ``int32`` and quarters it against the ``int64`` NumPy would pick by
    default -- 1.18 GB rather than 4.7 GB for Hindi's train split. The saving is
    not cosmetic: it decides whether the array memory-maps comfortably inside a
    Kaggle notebook's RAM budget. An assertion below enforces the vocabulary
    bound rather than trusting it, since a future vocabulary change that broke
    this would otherwise corrupt ids silently by wrapping.

**Documents joined by ``eos``, not concatenated bare.**
    Training samples are fixed-length windows cut from one long stream, so
    windows routinely straddle a document boundary. Without a separator the
    model sees one document's ending flow into an unrelated document's opening
    as though it were continuous text, and learns that transition as a real
    linguistic pattern. Emitting ``eos`` at each boundary makes the split
    explicit and gives generation a natural stopping symbol. The cost is one
    token per document -- for Hindi, 989,016 tokens against 590M, under 0.2%.

**Streaming writes, never a full in-memory list.**
    590M ids in a Python list would cost roughly 24 GB (28 bytes per boxed int
    plus pointers). Ids are accumulated in a fixed ``uint16`` buffer and flushed
    to an open file handle as it fills, so peak memory stays at the buffer size
    regardless of corpus size -- the same streaming discipline every Phase 1
    stage uses.

Output layout, one file per language per split::

    data/tokenized/<lang>_<split>.bin     raw little-endian uint16, no header
    data/tokenized/<lang>_<split>.json    token count, vocab, tokenizer, dtype

The ``.bin`` files carry no header so they can be opened directly with
``np.memmap(path, dtype=np.uint16, mode="r")``; the sidecar JSON holds the
metadata a header would otherwise provide, and doubles as the provenance record
tying each array to the exact tokenizer that produced it.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sentencepiece as spm

from common.io import iter_docs, enable_utf8_console

__all__ = ["TokenizeConfig", "tokenize_split", "tokenize_language", "main"]

# 8M ids = 16 MB per flush. Large enough that write syscalls are amortised,
# small enough to stay negligible against a Kaggle notebook's RAM.
_BUFFER_TOKENS = 8_000_000


@dataclass
class TokenizeConfig:
    """Where a language's corpus, splits and tokenizer live.

    Attributes:
        language: ``"hindi"`` or ``"assamese"``.
        vocab_size: Selected vocabulary (16,000 for both -- Phase 1 section 8.1).
        corpus_dir: Deduplicated shard directory to read documents from.
        splits_dir: Directory holding ``train.ids`` / ``val.ids`` / ``test.ids``.
        out_dir: Destination for the ``.bin`` / ``.json`` pairs.
    """

    language: str
    vocab_size: int = 16000
    corpus_dir: str = ""
    splits_dir: str = ""
    out_dir: str = "data/tokenized"

    def __post_init__(self) -> None:
        self.corpus_dir = self.corpus_dir or f"data/{self.language}/clean_deduped"
        self.splits_dir = self.splits_dir or f"{self.language}/splits"

    @property
    def model_path(self) -> str:
        """Path to the selected SentencePiece model for this language."""
        return f"{self.language}/tokenizer/{self.language}_bpe_{self.vocab_size}.model"


def _load_ids(path: Path | str) -> set[str]:
    """Read one split's document-id manifest into a set."""
    with Path(path).open("r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def tokenize_split(
    cfg: TokenizeConfig,
    split: str,
    sp: spm.SentencePieceProcessor,
    log=print,
) -> dict:
    """Tokenise one split into a flat ``uint16`` array on disk.

    Streams the corpus once, keeps only documents whose id is in this split's
    manifest, and appends ``encode(text) + [eos]`` for each. Written to a
    ``.tmp`` file and renamed into place only on success, so an interrupted run
    can never leave a truncated array that a later run mistakes for complete --
    the same atomic-completion rule ``common/io.py`` applies to corpus shards.

    Args:
        cfg: Language paths and vocabulary size.
        split: ``"train"``, ``"val"`` or ``"test"``.
        sp: Loaded SentencePiece processor for this language.
        log: Progress sink; swap for a no-op to silence.

    Returns:
        Metadata dict (also written alongside the array as JSON): token count,
        document count, vocabulary size, dtype, and the tokenizer that made it.
    """
    ids = _load_ids(Path(cfg.splits_dir) / f"{split}.ids")
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{cfg.language}_{split}.bin"
    tmp_path = out_path.with_suffix(".bin.tmp")

    eos = sp.eos_id()
    buf = np.empty(_BUFFER_TOKENS, dtype=np.uint16)
    n_buf = 0
    n_tokens = 0
    n_docs = 0
    t0 = time.time()

    with tmp_path.open("wb") as fh:
        for doc in iter_docs(cfg.corpus_dir):
            if doc.id not in ids:
                continue
            piece = sp.encode(doc.text, out_type=int)
            piece.append(eos)

            # Copy in chunks rather than assuming the document fits the free
            # buffer space. Archive.org books run to hundreds of thousands of
            # tokens and can exceed the whole buffer on their own, so this loop
            # handles a piece larger than one buffer as well as a partial fill.
            pos = 0
            while pos < len(piece):
                space = _BUFFER_TOKENS - n_buf
                take = min(space, len(piece) - pos)
                buf[n_buf : n_buf + take] = piece[pos : pos + take]
                n_buf += take
                pos += take
                if n_buf == _BUFFER_TOKENS:
                    fh.write(buf.tobytes())
                    n_buf = 0

            n_tokens += len(piece)
            n_docs += 1
            if n_docs % 100_000 == 0:
                log(f"    {cfg.language}/{split}: {n_docs:,} docs, {n_tokens:,} tokens")

        if n_buf:
            fh.write(buf[:n_buf].tobytes())

    tmp_path.replace(out_path)

    meta = {
        "language": cfg.language,
        "split": split,
        "tokens": n_tokens,
        "documents": n_docs,
        "vocab_size": sp.vocab_size(),
        "dtype": "uint16",
        "eos_id": eos,
        "tokenizer": cfg.model_path,
        "bytes_on_disk": out_path.stat().st_size,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    (out_dir / f"{cfg.language}_{split}.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    log(f"  {cfg.language}/{split}: {n_tokens:,} tokens from {n_docs:,} docs "
        f"-> {meta['bytes_on_disk']/1e9:.2f} GB ({meta['elapsed_seconds']}s)")
    return meta


def tokenize_language(cfg: TokenizeConfig, log=print) -> dict:
    """Tokenise all three splits for one language.

    Loads the tokenizer once and reuses it across splits. Asserts the
    vocabulary fits ``uint16`` before writing anything, so a vocabulary change
    fails loudly here rather than silently wrapping ids inside the arrays.
    """
    sp = spm.SentencePieceProcessor(model_file=cfg.model_path)
    assert sp.vocab_size() <= np.iinfo(np.uint16).max + 1, (
        f"vocab {sp.vocab_size()} exceeds uint16; widen the dtype in this module"
    )
    log(f"{cfg.language}: tokenizer={cfg.model_path} vocab={sp.vocab_size()}")
    return {split: tokenize_split(cfg, split, sp, log=log)
            for split in ("train", "val", "test")}


def main(argv: list[str] | None = None) -> int:
    """CLI. Example::

        python -m common.tokenization.tokenize_corpus --lang hindi
        python -m common.tokenization.tokenize_corpus --lang assamese --out-dir data/tokenized
    """
    import argparse

    enable_utf8_console()
    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("--lang", required=True, choices=("hindi", "assamese"))
    p.add_argument("--vocab-size", type=int, default=16000)
    p.add_argument("--out-dir", default="data/tokenized")
    args = p.parse_args(argv)

    cfg = TokenizeConfig(language=args.lang, vocab_size=args.vocab_size,
                         out_dir=args.out_dir)
    res = tokenize_language(cfg)
    total = sum(m["tokens"] for m in res.values())
    print(f"\n{args.lang}: {total:,} tokens across train/val/test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
