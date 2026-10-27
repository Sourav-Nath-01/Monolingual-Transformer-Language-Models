"""Language-model evaluation: perplexity, bits-per-byte, generation quality.

Implements spec section 2.3 for one model at a time; the report places the two
languages side by side.

**Why bits-per-byte matters here more than usual.** Perplexity is measured *per
token*, and a token means different things in the two models: Hindi's tokenizer
has fertility 1.3067 and Assamese's 1.6019 (Phase 1 report section 8.2), so
Assamese spends ~23% more tokens on the same amount of text. Comparing raw
perplexities across the two would therefore reward the tokenizer that happens to
chop text more finely, not the model that predicts language better. BPB
renormalises the same total log-likelihood onto UTF-8 bytes, a unit both
languages share, which is precisely why the spec asks for it alongside PPL.

    bpb = (sum of token NLL in nats) / ln(2) / (total UTF-8 bytes)

Note the denominator is bytes of the *decoded* text, not tokens, and Indic
scripts cost ~3 bytes per character in UTF-8 -- so BPB values here are not
comparable to English-corpus BPB numbers in the literature.

**On BLEU/chrF/ROUGE-L for open-ended generation.** These are reference-based
metrics designed for tasks with a correct answer (translation, summarisation).
A language model continuing a prefix has no single correct continuation, so a
fluent, sensible continuation that differs from the held-out one scores near
zero. They are reported because the spec requires them, and interpreted in the
report as *relative* signals -- comparing decoding temperatures and comparing
Hindi to Assamese under identical conditions -- rather than as absolute quality
scores. chrF is the most informative of the three for these languages because
it works on character n-grams, so it gives partial credit for correct
morphology where token-level BLEU gives none.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch

from common.modeling.data import TokenArray, iter_eval_batches
from common.modeling.model import ModelConfig, TransformerLM

__all__ = ["intrinsic_metrics", "lcs_length", "rouge_l", "distinct_n",
           "repetition_rate", "generation_metrics", "main"]


# ------------------------------------------------------------------ intrinsic

@torch.no_grad()
def intrinsic_metrics(
    model: TransformerLM,
    arr: TokenArray,
    sp,
    device: str,
    batch_size: int = 16,
    limit_batches: int | None = None,
) -> dict:
    """Cross-entropy, perplexity and bits-per-byte over a held-out split.

    Scores consecutive non-overlapping windows so every token counts exactly
    once (see :func:`common.modeling.data.iter_eval_batches` for why random sampling
    would be wrong here).

    The byte count is obtained by decoding each window's target ids back to text
    and measuring its UTF-8 length -- decoding rather than assuming a
    bytes-per-token constant, since that constant is exactly what differs
    between the two languages.

    Returns:
        Dict with ``cross_entropy`` (nats/token), ``perplexity``,
        ``bits_per_byte``, and the token/byte totals behind them.
    """
    model.eval()
    total_nll = 0.0      # summed over tokens, in nats
    total_tokens = 0
    total_bytes = 0

    for x, y in iter_eval_batches(arr, batch_size, model.cfg.block_size,
                                  device, limit_batches=limit_batches):
        logits, _ = model(x)
        # reduction="sum" so batches of differing size weight correctly; a mean
        # of per-batch means would over-weight a short final batch.
        nll = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
        total_nll += nll.item()
        total_tokens += y.numel()
        for row in y.tolist():
            total_bytes += len(sp.decode(row).encode("utf-8"))

    ce = total_nll / max(1, total_tokens)
    return {
        "cross_entropy_nats_per_token": ce,
        "perplexity": math.exp(min(ce, 20)),
        "bits_per_byte": (total_nll / math.log(2)) / max(1, total_bytes),
        "tokens_scored": total_tokens,
        "bytes_scored": total_bytes,
        "bytes_per_token": total_bytes / max(1, total_tokens),
    }


# ------------------------------------------------------- generation-quality

def lcs_length(a: list, b: list) -> int:
    """Longest common subsequence length, via the standard O(len(a)*len(b)) DP.

    Only two rows of the DP table are kept, since each row depends solely on the
    previous one -- relevant because continuations here run to hundreds of
    tokens and the full table would be needlessly large.
    """
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, yv in enumerate(b, 1):
            cur[j] = prev[j - 1] + 1 if x == yv else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def rouge_l(pred: list, ref: list) -> dict:
    """ROUGE-L: LCS-based precision, recall and F1.

    Implemented directly rather than pulled from a package: it is a short,
    well-defined computation, and the spec requires being able to explain every
    metric reported. LCS credits in-order overlap without demanding contiguity,
    so it tolerates insertions that exact n-gram matching would penalise fully.
    """
    if not pred or not ref:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    l = lcs_length(pred, ref)
    p = l / len(pred)
    r = l / len(ref)
    f = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
    return {"precision": p, "recall": r, "f1": f}


def distinct_n(sequences: list[list], n: int) -> float:
    """Distinct-n: unique n-grams / total n-grams across all generations.

    A degeneracy detector. A model stuck in a loop emits the same n-grams
    repeatedly, driving this toward 0; healthy varied text sits well above.
    Computed over the pooled corpus of generations, not averaged per-sample, so
    a model that repeats *across* samples is penalised too.
    """
    total, uniq = 0, set()
    for seq in sequences:
        for i in range(len(seq) - n + 1):
            uniq.add(tuple(seq[i : i + n]))
            total += 1
    return len(uniq) / max(1, total)


def repetition_rate(sequences: list[list], n: int = 4) -> float:
    """Fraction of n-grams that are repeats *within* their own sequence.

    Complements Distinct-n: this one catches a single sample looping, which
    corpus-level Distinct-n can mask if different samples loop on different
    phrases.
    """
    rep, total = 0, 0
    for seq in sequences:
        seen = set()
        for i in range(len(seq) - n + 1):
            g = tuple(seq[i : i + n])
            if g in seen:
                rep += 1
            seen.add(g)
            total += 1
    return rep / max(1, total)


@torch.no_grad()
def generation_metrics(
    model: TransformerLM,
    arr: TokenArray,
    sp,
    device: str,
    n_prompts: int = 64,
    prompt_tokens: int = 64,
    gen_tokens: int = 96,
    temperatures: tuple[float, ...] = (0.5, 1.0, 1.5),
    seed: int = 0,
) -> dict:
    """Generate continuations from held-out prefixes and score them.

    For each decoding setting (greedy, then each temperature) the same prompts
    are used, so differences across settings are attributable to decoding rather
    than to prompt choice. Each prompt is a ``prompt_tokens``-long prefix taken
    from the held-out array; the reference is the ``gen_tokens`` that actually
    followed it in the corpus.

    Returns:
        Dict keyed by decoding setting, each holding BLEU/chrF (if ``sacrebleu``
        is importable), ROUGE-L F1, Distinct-1/2, repetition rate, and a few
        decoded samples for the report's qualitative section.
    """
    model.eval()
    g = torch.Generator().manual_seed(seed)
    span = prompt_tokens + gen_tokens
    hi = max(1, len(arr) - span - 1)
    starts = torch.randint(hi, (n_prompts,), generator=g).tolist()

    prompts, references = [], []
    for s in starts:
        prompts.append([int(t) for t in arr.tokens[s : s + prompt_tokens]])
        references.append([int(t) for t in
                           arr.tokens[s + prompt_tokens : s + span]])

    try:
        import sacrebleu
    except ImportError:          # scored without them rather than failing outright
        sacrebleu = None

    settings = [("greedy", None)] + [(f"temp_{t}", t) for t in temperatures]
    out: dict = {}

    for name, temp in settings:
        idx = torch.tensor(prompts, dtype=torch.long, device=device)
        gen = model.generate(idx, max_new_tokens=gen_tokens,
                             temperature=temp or 1.0, greedy=(temp is None),
                             eos_id=sp.eos_id())
        conts = [row[prompt_tokens:].tolist() for row in gen]

        pred_txt = [sp.decode(c) for c in conts]
        ref_txt = [sp.decode(r) for r in references]

        entry: dict = {
            "rouge_l_f1": sum(rouge_l(c, r)["f1"]
                              for c, r in zip(conts, references)) / len(conts),
            "distinct_1": distinct_n(conts, 1),
            "distinct_2": distinct_n(conts, 2),
            "repetition_rate_4gram": repetition_rate(conts, 4),
            "mean_generated_tokens": sum(len(c) for c in conts) / len(conts),
        }
        if sacrebleu is not None:
            # BLEU-4 (sacrebleu's default max n-gram order) on corpus level.
            entry["bleu4"] = sacrebleu.corpus_bleu(
                pred_txt, [ref_txt]).score
            entry["chrf"] = sacrebleu.corpus_chrf(pred_txt, [ref_txt]).score
        entry["samples"] = [
            {"prompt": sp.decode(prompts[i]),
             "generated": pred_txt[i],
             "reference": ref_txt[i]}
            for i in range(min(3, len(pred_txt)))
        ]
        out[name] = entry

    return out


# ------------------------------------------------------------------------ CLI

def main(argv: list[str] | None = None) -> int:
    """CLI. Example::

        python -m common.modeling.evaluate --lang hindi --checkpoint hindi/model/best.pt
    """
    import argparse
    import sentencepiece as spm

    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("--lang", required=True, choices=("hindi", "assamese"))
    p.add_argument("--checkpoint", default="")
    p.add_argument("--data-dir", default="data/tokenized")
    p.add_argument("--split", default="val", choices=("val", "test"))
    p.add_argument("--out", default="")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--limit-batches", type=int, default=None,
                   help="cap intrinsic scoring; omit to score the whole split")
    p.add_argument("--n-prompts", type=int, default=64)
    args = p.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = args.checkpoint or f"{args.lang}/model/best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    mcfg = ModelConfig(**ckpt["model_config"])
    model = TransformerLM(mcfg).to(device)
    model.load_state_dict(ckpt["model"])

    sp = spm.SentencePieceProcessor(
        model_file=f"{args.lang}/tokenizer/{args.lang}_bpe_{mcfg.vocab_size}.model")
    arr = TokenArray(Path(args.data_dir) / f"{args.lang}_{args.split}.bin")

    print(f"{args.lang}: checkpoint step {ckpt.get('step')}, split={args.split}, "
          f"{len(arr):,} tokens, device={device}")

    results = {
        "language": args.lang,
        "checkpoint": str(ckpt_path),
        "checkpoint_step": ckpt.get("step"),
        "split": args.split,
        "parameters": model.parameter_report(),
        "intrinsic": intrinsic_metrics(model, arr, sp, device,
                                       args.batch_size, args.limit_batches),
        "generation": generation_metrics(model, arr, sp, device,
                                         n_prompts=args.n_prompts),
    }

    i = results["intrinsic"]
    print(f"  CE {i['cross_entropy_nats_per_token']:.4f} nats/token | "
          f"PPL {i['perplexity']:.2f} | BPB {i['bits_per_byte']:.4f}")
    for k, v in results["generation"].items():
        print(f"  {k:10s} rougeL {v['rouge_l_f1']:.4f} "
              f"d1 {v['distinct_1']:.3f} d2 {v['distinct_2']:.3f} "
              f"rep4 {v['repetition_rate_4gram']:.3f}"
              + (f" bleu {v['bleu4']:.2f} chrf {v['chrf']:.2f}" if "bleu4" in v else ""))

    out = Path(args.out or f"{args.lang}/eval/metrics_{args.split}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
