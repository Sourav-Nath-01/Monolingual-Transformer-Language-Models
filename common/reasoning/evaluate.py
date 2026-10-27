"""Score a checkpoint on the held-out reasoning test set.

Spec section 3.1 asks for "pretrained vs. finetuned accuracy (or exact match) on
the synthetic test set, qualitative successes/failures, and a comparison between
Model H and Model L".

**Free generation, not teacher forcing.** During training, accuracy is measured
by argmax at the supervised positions, which is cheap and fine for model
selection. It is not what the report should quote: on the shipped test split
17.2% of Hindi answers and 34.2% of Assamese ones tokenise to more than one
token, and a teacher-forced score credits a model that would have derailed
after the first token. Here the model
generates from the prompt with nothing given, and the decoded string must equal
the gold answer exactly.

**Chance is reported alongside.** A two-option comparison is 50% at chance and a
three-option ordering is 33%, so a raw accuracy means little on its own. The
per-task breakdown carries its own chance baseline, computed from the number of
candidate entities named in each prompt.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch

from common.modeling.model import ModelConfig, TransformerLM
from common.reasoning.data import load_jsonl
from common.reasoning.finetune import ANSWER_PREFIX

__all__ = ["load_checkpoint", "score_checkpoint", "main"]

# Entities named per task -- the denominator for a random-guess baseline.
CHANCE = {"compare2": 1 / 2, "order3": 1 / 3, "transitive": 1 / 2}


def load_checkpoint(path: str | Path, device: str):
    """Rebuild the model from a checkpoint written by either phase."""
    ck = torch.load(path, map_location=device, weights_only=False)
    model = TransformerLM(ModelConfig(**ck["model_config"])).to(device)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    return model, ck


def _clean(text: str, marker: str | None = None) -> str:
    """Extract the predicted entity from the model's continuation.

    The target is ``<rationale>, <marker> <entity>``, so the prediction is
    whatever follows the marker -- ``इसलिए`` in Hindi, ``গতিকে`` in Assamese.
    Scoring only the entity keeps exact-match comparable with the earlier
    answer-only runs, and means a model that shows correct working but draws the
    wrong conclusion still scores zero. That is deliberate: the conclusion is
    the task.

    Falls back to "text up to the first sentence break" when no marker is
    present, so output from a pre-rationale checkpoint still scores.
    """
    text = text.strip()
    if marker and marker in text:
        text = text.split(marker, 1)[1]
    text = re.split(r"[।\n|.?!,;]", text, maxsplit=1)[0]
    return " ".join(text.split()).strip()


@torch.no_grad()
def score_checkpoint(ckpt_path: str | Path, lang: str, split: str = "test",
                     data_dir: str = "", device: str | None = None,
                     max_new_tokens: int = 24, batch_size: int = 32,
                     limit: int | None = None, log=print) -> dict:
    """Generate an answer for every test prompt and exact-match it.

    Returns a dict with overall accuracy, per-task and per-template breakdowns,
    the chance baseline, and a handful of successes and failures for the report.
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

    by_task = defaultdict(lambda: [0, 0, 0])      # task -> [exact, prefix, total]
    by_template = defaultdict(lambda: [0, 0])
    correct = prefixed = 0
    wins, losses = [], []

    # Group by prompt length so padding inside a batch stays minimal; generation
    # is left-padded-free because every sequence in a batch has equal length.
    order = sorted(range(len(rows)), key=lambda i: len(sp.encode(rows[i]["full_prompt"])))
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        enc = [sp.encode(rows[i]["full_prompt"], out_type=int) for i in idx]
        width = max(len(e) for e in enc)
        # Pad on the LEFT so every prompt ends at the same position and the
        # continuation begins from the true final token of each prompt.
        x = torch.full((len(enc), width), sp.pad_id(), dtype=torch.long)
        for r, e in enumerate(enc):
            x[r, width - len(e):] = torch.tensor(e, dtype=torch.long)
        x = x.to(device)
        if width + max_new_tokens > mcfg.block_size:
            x = x[:, -(mcfg.block_size - max_new_tokens):]

        out = model.generate(x, max_new_tokens=max_new_tokens, greedy=True,
                             eos_id=sp.eos_id())
        for r, i in enumerate(idx):
            gen = sp.decode(out[r, x.size(1):].tolist())
            pred = _clean(gen, rows[i].get("marker"))
            gold = rows[i]["answer"].strip()
            hit = pred == gold
            # A base model has no reason to stop after the answer, so it often
            # names the right entity and keeps going ("माधव की लंबाई 185..."),
            # which strict matching scores as zero. Prefix match credits that,
            # and reporting both keeps the finetuning gain from looking larger
            # than it is.
            pre = pred.startswith(gold)
            correct += hit
            prefixed += pre
            by_task[rows[i]["task"]][0] += hit
            by_task[rows[i]["task"]][1] += pre
            by_task[rows[i]["task"]][2] += 1
            by_template[rows[i]["template"]][0] += hit
            by_template[rows[i]["template"]][1] += 1
            rec = {"id": rows[i]["id"], "task": rows[i]["task"],
                   "prompt": rows[i]["prompt"], "gold": gold, "pred": pred,
                   "generated": gen.strip()[:160]}
            if hit and len(wins) < 6:
                wins.append(rec)
            elif not hit and len(losses) < 6:
                losses.append(rec)

    n = len(rows)
    chance = sum(CHANCE[t] * c[2] for t, c in by_task.items()) / max(1, n)
    result = {
        "language": lang, "split": split, "checkpoint": str(ckpt_path),
        "checkpoint_step": ck.get("step"),
        "examples": n,
        "exact_match": correct / max(1, n),
        "prefix_match": prefixed / max(1, n),
        "chance_baseline": chance,
        "by_task": {t: {"exact_match": c[0] / c[2], "prefix_match": c[1] / c[2],
                        "n": c[2], "chance": CHANCE[t]}
                    for t, c in sorted(by_task.items())},
        "by_template": {t: {"exact_match": c[0] / c[1], "n": c[1]}
                        for t, c in sorted(by_template.items())},
        "successes": wins, "failures": losses,
    }
    log(f"  {lang}/{split} [{Path(ckpt_path).name}] exact-match "
        f"{result['exact_match']:.4f} | prefix-match {result['prefix_match']:.4f} "
        f"| chance {chance:.4f} | n={n:,}")
    for t, c in sorted(by_task.items()):
        log(f"      {t:11s} exact {c[0]/c[2]:.4f}  prefix {c[1]/c[2]:.4f}  "
            f"(chance {CHANCE[t]:.2f}, n={c[2]:,})")
    return result


def main(argv: list[str] | None = None) -> int:
    """CLI. Example::

        python -m common.reasoning.evaluate --lang hindi \
            --pretrained hindi/model/best.pt \
            --finetuned hindi/train/finetuned/best.pt
    """
    import argparse
    from common.io import enable_utf8_console

    enable_utf8_console()
    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("--lang", required=True, choices=("hindi", "assamese"))
    p.add_argument("--pretrained", required=True)
    p.add_argument("--finetuned", default=None)
    p.add_argument("--split", default="test",
                   choices=("val", "test", "test_unseen"))
    p.add_argument("--data-dir", default="")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default="")
    args = p.parse_args(argv)

    out = {}
    print("pretrained:")
    out["pretrained"] = score_checkpoint(args.pretrained, args.lang, args.split,
                                         args.data_dir, limit=args.limit)
    if args.finetuned:
        print("finetuned:")
        out["finetuned"] = score_checkpoint(args.finetuned, args.lang, args.split,
                                            args.data_dir, limit=args.limit)
        d = out["finetuned"]["exact_match"] - out["pretrained"]["exact_match"]
        out["improvement"] = d
        print(f"\n  {args.lang}: {out['pretrained']['exact_match']:.4f} -> "
              f"{out['finetuned']['exact_match']:.4f}  ({d:+.4f})")

    path = Path(args.out or f"{args.lang}/eval/reasoning_{args.split}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
