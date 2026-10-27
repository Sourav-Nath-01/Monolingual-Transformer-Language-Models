"""Attention on a comparative-reasoning prompt, for one checkpoint.

Spec section 3.2: "Reuse the Phase 2 attention toolkit on the finetuned
checkpoints. Compare pretrained vs. finetuned heatmaps for at least one early
and one late layer per model. Comment on whether finetuning changed local vs.
long-range attention or head specialization, especially on comparative-reasoning
prompts."

Two choices follow from that wording:

**The probe is a reasoning prompt, not corpus text.** Phase 2 measured attention
on a passage from the held-out corpus, which is the right probe for a language
model. The question here is different -- what finetuning did to the mechanism on
*the task* -- so the input is a transitive-comparison prompt drawn from the
reasoning test split. Attention on news prose would answer a question nobody
asked.

**Both checkpoints see the identical prompt.** Run this once per checkpoint with
a different ``--tag``; the prompt is selected by a fixed index, not sampled, so
the pretrained and finetuned figures are directly comparable cell by cell. A
randomly drawn prompt per run would make every difference uninterpretable.

The statistics themselves -- normalised entropy and mean attention distance --
are computed by ``common.modeling.attention``, unchanged from Phase 2, so the
numbers are on the same scale as the pretraining analysis.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from common.modeling.attention import (collect_attention, head_statistics,
                                       plot_head_summary, plot_heatmaps)
from common.modeling.model import ModelConfig, TransformerLM
from common.reasoning.data import load_jsonl
from common.reasoning.finetune import ANSWER_PREFIX

__all__ = ["probe_checkpoint", "main"]

# A fixed index into the transitive subset of the test split. Transitive is the
# task where long-range attention should matter most: the answer depends on
# chaining two premises, so the query at the question must reach back to both.
PROMPT_INDEX = 0


def probe_checkpoint(ckpt_path: str | Path, lang: str, tag: str,
                     out_dir: str | Path, data_dir: str = "",
                     device: str | None = None, log=print) -> dict:
    """Collect and plot attention for one checkpoint on a fixed reasoning prompt.

    Args:
        ckpt_path: Pretrained or finetuned checkpoint.
        lang: ``"hindi"`` or ``"assamese"``.
        tag: ``"pretrained"`` or ``"finetuned"`` -- goes into the filenames so
            the two runs do not overwrite each other.
        out_dir: Where the figures and statistics land.
        data_dir: Defaults to ``<lang>/train``.
        device: Defaults to CUDA when available.

    Returns:
        The statistics dict, also written as JSON.
    """
    import sentencepiece as spm

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    mcfg = ModelConfig(**ck["model_config"])
    model = TransformerLM(mcfg).to(device)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()

    sp = spm.SentencePieceProcessor(
        model_file=f"{lang}/tokenizer/{lang}_bpe_{mcfg.vocab_size}.model")

    rows = load_jsonl(Path(data_dir or f"{lang}/train") / "reasoning_test.jsonl",
                      ANSWER_PREFIX[lang])
    transitive = [r for r in rows if r["task"] == "transitive"]
    row = transitive[PROMPT_INDEX]

    ids = sp.encode(row["full_prompt"], out_type=int)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    tokens = [sp.id_to_piece(i) for i in ids]

    attns = collect_attention(model, idx)
    stats = head_statistics(attns)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    early, late = 0, mcfg.n_layer - 1
    heads = list(range(min(3, mcfg.n_head)))
    hm = plot_heatmaps(attns, [early, late], heads, tokens,
                       out / f"attention_{tag}_{lang}.png", f"{lang} ({tag})")
    sm = plot_head_summary(stats, out / f"attention_summary_{tag}_{lang}.png",
                           f"{lang} ({tag})")

    payload = {
        "language": lang, "tag": tag, "checkpoint": str(ckpt_path),
        "checkpoint_step": ck.get("step"),
        "prompt_id": row["id"], "task": row["task"],
        "prompt": row["full_prompt"], "answer": row["answer"],
        "n_tokens": len(ids),
        **stats,
    }
    path = out / f"attention_stats_{tag}_{lang}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                    encoding="utf-8")

    log(f"  {lang} [{tag}] step {ck.get('step')} on a {row['task']} prompt "
        f"({len(ids)} tokens)")
    log(f"    layer mean entropy : "
        f"{[round(x, 3) for x in stats['layer_mean_entropy']]}")
    log(f"    layer mean distance: "
        f"{[round(x, 2) for x in stats['layer_mean_distance']]}")
    log(f"    wrote {hm}\n    wrote {sm}\n    wrote {path}")
    return payload


def main(argv: list[str] | None = None) -> int:
    """CLI. Example::

        python -m common.reasoning.attention_compare --lang hindi \
            --checkpoint hindi/model/best.pt --tag pretrained
    """
    import argparse
    from common.io import enable_utf8_console

    enable_utf8_console()
    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("--lang", required=True, choices=("hindi", "assamese"))
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tag", required=True, choices=("pretrained", "finetuned"))
    p.add_argument("--out-dir", default="")
    p.add_argument("--data-dir", default="")
    args = p.parse_args(argv)

    probe_checkpoint(args.checkpoint, args.lang, args.tag,
                     args.out_dir or f"{args.lang}/eval", args.data_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
