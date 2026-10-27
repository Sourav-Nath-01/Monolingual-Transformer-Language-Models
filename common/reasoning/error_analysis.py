"""Classify *why* the finetuned models get reasoning items wrong.

Spec section 3.3 asks for error analysis, and an aggregate exact-match number
cannot supply one: 0.8460 and 0.7555 say how often each model is right but
nothing about the shape of the remaining error. Six hand-picked failures hinted
at something specific -- the model states the correct *value* and then names the
wrong *entity*:

    सतीश, आशीष और नरेश की उम्र क्रमशः 76 साल, 62 साल और 60 साल है।
    सबसे ज़्यादा उम्र किसकी है?
    gold सतीश  ->  generated "76 साल, इसलिए आशीष"

76 is the right number. The comparison succeeded; the binding from that number
back to its owner failed. This module measures how general that is.

Every ``compare2``/``order3`` row carries ``meta.values``, the entity -> value
map the generator used, so both halves of the prediction can be scored
independently:

``correct``
    The named entity is the gold entity.
``binding_in_prompt``
    The stated value is the gold value, but a *different entity from the
    prompt* is named. The comparison worked; the value-to-owner link broke.
``binding_off_prompt``
    The stated value is the gold value, but the named entity does not occur in
    the prompt at all -- recalled from the training pool rather than read off
    the question. A strictly worse failure than the above.
``comparison``
    The stated value is not the gold value: the model selected the wrong
    operand, so the comparison itself failed.
``other``
    No value could be parsed from the continuation (malformed output).

``transitive`` rows are scored correct/incorrect only, because their rationale
restates an ordering chain rather than one decisive value. Both models score
1.0000 there, so there is nothing to partition.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch

from common.modeling.model import ModelConfig
from common.reasoning.data import load_jsonl
from common.reasoning.evaluate import _clean, load_checkpoint
from common.reasoning.finetune import ANSWER_PREFIX

__all__ = ["generate_predictions", "classify", "analyse", "main"]


@torch.no_grad()
def generate_predictions(ckpt_path: str | Path, lang: str, split: str = "test",
                         data_dir: str = "", device: str | None = None,
                         max_new_tokens: int = 24, batch_size: int = 32,
                         limit: int | None = None, log=print) -> list[dict]:
    """Free-generate a continuation for every row and return them with the row.

    Mirrors :func:`common.reasoning.evaluate.score_checkpoint` -- same greedy
    decoding, same left-padding, same ``_clean`` -- but keeps every prediction
    instead of aggregating, which is what per-example classification needs.
    """
    import sentencepiece as spm

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_checkpoint(ckpt_path, device)
    mcfg = ModelConfig(**ck["model_config"])
    sp = spm.SentencePieceProcessor(
        model_file=f"{lang}/tokenizer/{lang}_bpe_{mcfg.vocab_size}.model")

    data_dir = data_dir or f"{lang}/train"
    rows = load_jsonl(Path(data_dir) / f"reasoning_{split}.jsonl", ANSWER_PREFIX[lang])
    if limit:
        rows = rows[:limit]

    out: list[dict] = [None] * len(rows)                      # type: ignore[list-item]
    order = sorted(range(len(rows)), key=lambda i: len(sp.encode(rows[i]["full_prompt"])))
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        enc = [sp.encode(rows[i]["full_prompt"], out_type=int) for i in idx]
        width = max(len(e) for e in enc)
        x = torch.full((len(enc), width), sp.pad_id(), dtype=torch.long)
        for r, e in enumerate(enc):
            x[r, width - len(e):] = torch.tensor(e, dtype=torch.long)
        x = x.to(device)
        if width + max_new_tokens > mcfg.block_size:
            x = x[:, -(mcfg.block_size - max_new_tokens):]

        gen = model.generate(x, max_new_tokens=max_new_tokens, greedy=True,
                             eos_id=sp.eos_id())
        for r, i in enumerate(idx):
            text = sp.decode(gen[r, x.size(1):].tolist())
            out[i] = {"row": rows[i], "generated": text.strip(),
                      "pred": _clean(text, rows[i].get("marker"))}
        if start % (batch_size * 20) == 0:
            log(f"      {start + len(idx):,}/{len(rows):,}")
    return out


def _first_number(text: str) -> int | None:
    """The first integer in a string, or None. Values are written in ASCII digits."""
    m = re.search(r"\d+", text)
    return int(m.group()) if m else None


def _stated_value(generated: str, marker: str | None) -> int | None:
    """The value the model asserted in its rationale, i.e. before the marker."""
    head = generated.split(marker, 1)[0] if marker and marker in generated else generated
    return _first_number(head)


def classify(rec: dict) -> str:
    """Bucket one prediction. See the module docstring for the categories."""
    row = rec["row"]
    gold = row["answer"].strip()
    pred = rec["pred"]
    if pred == gold:
        return "correct"
    if row["task"] == "transitive":
        return "transitive_wrong"

    values = (row.get("meta") or {}).get("values") or {}
    stated = _stated_value(rec["generated"], row.get("marker"))
    if stated is None:
        return "other"
    # The value the gold answer actually carries in this prompt.
    if values.get(gold) != stated:
        return "comparison"
    return "binding_in_prompt" if pred in values else "binding_off_prompt"


def analyse(ckpt_path: str | Path, lang: str, split: str = "test",
            data_dir: str = "", limit: int | None = None, log=print) -> dict:
    """Generate, classify, and summarise the error distribution."""
    log(f"  {lang}/{split} generating...")
    recs = generate_predictions(ckpt_path, lang, split, data_dir,
                                limit=limit, log=log)
    buckets = Counter()
    by_task = defaultdict(Counter)
    examples = defaultdict(list)
    for rec in recs:
        cat = classify(rec)
        buckets[cat] += 1
        by_task[rec["row"]["task"]][cat] += 1
        if cat != "correct" and len(examples[cat]) < 4:
            examples[cat].append({
                "id": rec["row"]["id"], "task": rec["row"]["task"],
                "prompt": rec["row"]["prompt"], "gold": rec["row"]["answer"],
                "pred": rec["pred"], "generated": rec["generated"][:160],
                "values": (rec["row"].get("meta") or {}).get("values"),
            })

    n = len(recs)
    wrong = n - buckets["correct"]
    result = {
        "language": lang, "split": split, "checkpoint": str(ckpt_path),
        "examples_scored": n,
        "counts": dict(buckets),
        "share_of_all": {k: v / n for k, v in sorted(buckets.items())},
        "share_of_errors": {k: v / wrong for k, v in sorted(buckets.items())
                            if k != "correct"} if wrong else {},
        "by_task": {t: dict(c) for t, c in sorted(by_task.items())},
        "error_examples": {k: v for k, v in sorted(examples.items())},
    }
    log(f"    n={n:,}  correct={buckets['correct']:,} ({buckets['correct']/n:.4f})"
        f"  errors={wrong:,}")
    for k, v in sorted(buckets.items()):
        if k == "correct":
            continue
        log(f"      {k:20s} {v:5,}  {v/n:.4f} of all  {v/max(1,wrong):.4f} of errors")
    return result


def main(argv: list[str] | None = None) -> int:
    """CLI. Example::

        python -m common.reasoning.error_analysis --lang hindi \
            --checkpoint hindi/train/finetuned/best.pt
    """
    import argparse
    from common.io import enable_utf8_console

    enable_utf8_console()
    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("--lang", required=True, choices=("hindi", "assamese"))
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test",
                   choices=("val", "test", "test_unseen"))
    p.add_argument("--data-dir", default="")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default="")
    args = p.parse_args(argv)

    result = analyse(args.checkpoint, args.lang, args.split, args.data_dir,
                     limit=args.limit)
    path = Path(args.out or f"{args.lang}/eval/error_analysis_{args.split}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
