"""Batch sampling from the memory-mapped ``uint16`` token arrays.

Reads the ``.bin`` files produced by :mod:`common.tokenization.tokenize_corpus`. The design
constraint is Kaggle: a notebook has finite RAM shared with the CUDA context,
and Hindi's train array alone is 1.18 GB, so the array is **memory-mapped**
rather than loaded. The OS pages in only the windows actually touched, which
keeps resident memory roughly at the batch size no matter how large the corpus
grows.

**Random-offset sampling, not sequential epochs.** Each batch draws
``batch_size`` independent random start offsets and cuts ``block_size + 1``
tokens from each. This is the standard approach for LM pretraining at this
scale and is chosen deliberately over shuffling fixed chunks:

* At 590M tokens and a realistic step budget, training sees well under one full
  pass, so "epoch" is not a meaningful unit here -- what matters is that every
  step draws an unbiased sample of the corpus, which random offsets give.
* Chunk boundaries are not fixed across the run, so the model is not repeatedly
  shown the same arbitrary cut points and cannot memorise them as structure.

The cost, stated honestly: sampling with replacement means some windows repeat
and some are never seen within a given budget. At this corpus size the
duplication rate is negligible, and the alternative (tracking a shuffled index
over ~1.15M possible windows) buys nothing for the extra bookkeeping.

**Why ``x`` and ``y`` overlap by one token.** Causal LM predicts token ``t+1``
from tokens ``<= t``, so the target sequence is the input shifted left by one:
a window of ``block_size + 1`` tokens yields ``x = w[:-1]`` and ``y = w[1:]``.
Every position in the block therefore contributes a training signal, rather
than only the final one.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

__all__ = ["TokenArray", "get_batch", "iter_eval_batches"]


class TokenArray:
    """A memory-mapped ``uint16`` token array plus its sidecar metadata.

    Attributes:
        path: The ``.bin`` file.
        tokens: Read-only ``np.memmap`` view of the whole array.
        meta: Parsed sidecar JSON (token count, vocab, tokenizer provenance).
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"{self.path} not found -- run `python -m common.tokenization.tokenize_corpus "
                f"--lang <lang>` first, or point --data-dir at the Kaggle dataset"
            )
        self.tokens = np.memmap(self.path, dtype=np.uint16, mode="r")
        meta_path = self.path.with_suffix(".json")
        self.meta = (json.loads(meta_path.read_text(encoding="utf-8"))
                     if meta_path.exists() else {})

        # The sidecar is the authority on length; a mismatch means the array was
        # truncated (interrupted write, partial upload) and every downstream
        # number would be quietly wrong, so fail loudly here instead.
        expected = self.meta.get("tokens")
        if expected is not None and expected != len(self.tokens):
            raise ValueError(
                f"{self.path}: sidecar says {expected:,} tokens but the file holds "
                f"{len(self.tokens):,} -- array is truncated or mismatched"
            )

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def vocab_size(self) -> int | None:
        """Vocabulary recorded at tokenisation time, if the sidecar is present."""
        return self.meta.get("vocab_size")

    def n_windows(self, block_size: int) -> int:
        """How many distinct ``block_size + 1`` windows the array contains."""
        return max(0, len(self.tokens) - block_size)

    def __repr__(self) -> str:
        return (f"TokenArray({self.path.name}, {len(self):,} tokens, "
                f"vocab={self.vocab_size})")


def get_batch(
    arr: TokenArray,
    batch_size: int,
    block_size: int,
    device: str = "cpu",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample one training batch of random windows.

    Args:
        arr: Source token array.
        batch_size: Windows per batch.
        block_size: Tokens per window (model context length).
        device: Destination device for the tensors.
        generator: Optional seeded RNG, so a run can be made reproducible.

    Returns:
        ``(x, y)``, both ``(batch_size, block_size)`` int64, with ``y`` equal to
        ``x`` shifted one token left.
    """
    hi = arr.n_windows(block_size)
    if hi <= 0:
        raise ValueError(
            f"{arr.path.name} holds {len(arr):,} tokens, too few for "
            f"block_size={block_size}"
        )
    ix = torch.randint(hi, (batch_size,), generator=generator)

    # np.memmap slices are copied into torch tensors here; astype(np.int64) is
    # required because embedding lookup needs int64 indices and uint16 has no
    # torch equivalent.
    x = torch.stack([
        torch.from_numpy(arr.tokens[i : i + block_size].astype(np.int64))
        for i in ix.tolist()
    ])
    y = torch.stack([
        torch.from_numpy(arr.tokens[i + 1 : i + 1 + block_size].astype(np.int64))
        for i in ix.tolist()
    ])

    if device.startswith("cuda"):
        # pin_memory + non_blocking overlaps the host->device copy with compute.
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def iter_eval_batches(
    arr: TokenArray,
    batch_size: int,
    block_size: int,
    device: str = "cpu",
    limit_batches: int | None = None,
):
    """Iterate *non-overlapping consecutive* windows, for evaluation.

    Evaluation must not use :func:`get_batch`'s random sampling: perplexity and
    bits-per-byte are corpus statistics, and sampling with replacement would
    make them depend on the draw rather than on the held-out set. This walks the
    array in order with a stride of exactly ``block_size``, so every token is
    scored once and once only, and the number is reproducible across runs.

    Args:
        arr: Held-out token array (val or test).
        batch_size: Windows per batch.
        block_size: Tokens per window.
        device: Destination device.
        limit_batches: Stop after this many batches. Used to keep the
            during-training validation cheap; leave ``None`` for final metrics
            so the full split is scored.

    Yields:
        ``(x, y)`` batches, both ``(<=batch_size, block_size)`` int64.
    """
    n = arr.n_windows(block_size)
    starts = range(0, n, block_size)
    batch: list[int] = []
    emitted = 0

    for s in starts:
        batch.append(s)
        if len(batch) == batch_size:
            yield _stack(arr, batch, block_size, device)
            batch = []
            emitted += 1
            if limit_batches is not None and emitted >= limit_batches:
                return
    if batch and (limit_batches is None or emitted < limit_batches):
        yield _stack(arr, batch, block_size, device)


def _stack(arr: TokenArray, starts: list[int], block_size: int, device: str):
    """Build one ``(x, y)`` pair from explicit start offsets."""
    x = torch.stack([
        torch.from_numpy(arr.tokens[i : i + block_size].astype(np.int64))
        for i in starts
    ])
    y = torch.stack([
        torch.from_numpy(arr.tokens[i + 1 : i + 1 + block_size].astype(np.int64))
        for i in starts
    ])
    return x.to(device), y.to(device)
