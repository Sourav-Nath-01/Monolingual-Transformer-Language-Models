"""Turn the reasoning JSONL into padded, answer-masked training batches.

Two things make this different from Phase 2's batching, and both matter:

**Loss on the answer only.** Pretraining predicts every token, because every
token is the thing being learned. Here the prompt is *given* -- it is the
question, not the answer -- so training the model to predict the question text
wastes capacity on something it is never asked to produce and dilutes the
gradient from the tokens that carry the task. Prompt positions are set to the
ignore index, so they contribute context but no loss.

**Padding, not a continuous stream.** Phase 2 cut fixed-length windows out of
one long token array, where every position was real. Reasoning examples are
discrete and vary in length, so short ones are padded to the batch maximum and
the padding is masked out of the loss the same way the prompt is.

The ignore index is ``-100``, which is what ``F.cross_entropy`` skips by
default, so no custom loss function is needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

__all__ = ["ReasoningExample", "load_jsonl", "encode_examples", "ReasoningBatches"]

IGNORE = -100


class ReasoningExample:
    """One prompt/answer pair with its token ids and the answer's start index."""

    __slots__ = ("id", "task", "template", "prompt", "answer", "target",
                 "ids", "answer_start")

    def __init__(self, row: dict, sp) -> None:
        self.id = row["id"]
        self.task = row["task"]
        self.template = row["template"]
        self.prompt = row["prompt"]
        self.answer = row["answer"]
        # Falls back to the bare answer so datasets generated before the
        # rationale was introduced still load.
        self.target = row.get("target", row["answer"])

        # The answer prefix ("उत्तर:" / "উত্তৰ:") is part of the prompt, so the
        # model learns to produce its working *after* seeing that cue.
        p_ids = sp.encode(row["full_prompt"], out_type=int)

        # Two things are supervised here that were not, each after a measured
        # failure. The rationale, because training only on the final entity gave
        # the model no intermediate step and left it at chance (0.4355 against a
        # 0.4512 baseline). And EOS, because without it the loss rewards
        # producing the answer but never stopping -- 93% prefix-match against
        # 0.2% exact-match on transitive.
        a_ids = sp.encode(" " + self.target, out_type=int) + [sp.eos_id()]
        self.ids = p_ids + a_ids
        self.answer_start = len(p_ids)

    def __len__(self) -> int:
        return len(self.ids)


def load_jsonl(path: Path | str, answer_prefix: str) -> list[dict]:
    """Read one split, attaching the prompt the model actually sees.

    Args:
        path: The ``reasoning_<split>.jsonl`` file.
        answer_prefix: ``"उत्तर:"`` or ``"উত্তৰ:"`` -- appended to every prompt so
            the model has a consistent cue for where the answer begins.
    """
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        r["full_prompt"] = f"{r['prompt']} {answer_prefix}"
        rows.append(r)
    return rows


def encode_examples(rows: list[dict], sp, block_size: int,
                    log=print) -> list[ReasoningExample]:
    """Tokenise every row, dropping any that will not fit the context window."""
    out, dropped = [], 0
    for r in rows:
        ex = ReasoningExample(r, sp)
        if len(ex) > block_size:
            dropped += 1
            continue
        out.append(ex)
    if dropped:
        log(f"    dropped {dropped} examples longer than block_size={block_size}")
    return out


class ReasoningBatches:
    """Yields padded, answer-masked batches from a list of examples.

    Args:
        examples: Encoded examples.
        batch_size: Sequences per batch.
        pad_id: Token used for padding; masked out of the loss regardless.
        device: Where to place the tensors.
        shuffle: True for training, False so evaluation order is reproducible.
        seed: Shuffle seed.
    """

    def __init__(self, examples: list[ReasoningExample], batch_size: int,
                 pad_id: int, device: str, shuffle: bool = True,
                 seed: int = 0) -> None:
        self.examples = examples
        self.batch_size = batch_size
        self.pad_id = pad_id
        self.device = device
        self.shuffle = shuffle
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return (len(self.examples) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        order = (torch.randperm(len(self.examples), generator=self.generator).tolist()
                 if self.shuffle else range(len(self.examples)))
        batch: list[ReasoningExample] = []
        for i in order:
            batch.append(self.examples[i])
            if len(batch) == self.batch_size:
                yield self._collate(batch)
                batch = []
        if batch:
            yield self._collate(batch)

    def _collate(self, batch: list[ReasoningExample]):
        """Pad to the batch maximum and mask prompt + padding out of the loss."""
        width = max(len(e) for e in batch)
        x = torch.full((len(batch), width - 1), self.pad_id, dtype=torch.long)
        y = torch.full((len(batch), width - 1), IGNORE, dtype=torch.long)
        for r, e in enumerate(batch):
            ids = e.ids
            n = len(ids) - 1
            x[r, :n] = torch.tensor(ids[:-1], dtype=torch.long)
            tgt = torch.tensor(ids[1:], dtype=torch.long)
            # Position j of y predicts ids[j+1]; supervise only where that
            # target lies inside the answer.
            keep = torch.arange(n) >= (e.answer_start - 1)
            y[r, :n] = torch.where(keep, tgt, torch.full_like(tgt, IGNORE))
        return x.to(self.device), y.to(self.device)
