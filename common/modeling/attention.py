"""Attention analysis: heatmaps, per-head entropy, mean attention distance.

Spec section 2.3's final block. Because multi-head attention here is
student-implemented (:class:`common.modeling.model.CausalSelfAttention`), the weights are
directly inspectable -- ``forward(..., return_attn=True)`` hands back the
post-softmax probabilities for every layer, shape ``(B, n_head, T, T)``.

Two summary statistics turn those matrices into something comparable across
heads, layers and languages:

**Entropy** measures how *diffuse* a head is. For query position ``t`` the
attention row is a probability distribution over keys ``0..t``, and its Shannon
entropy is low when the head concentrates on a few positions, high when it
spreads mass broadly. Raw entropy is not comparable across positions, though,
because position ``t`` can attend to ``t+1`` keys and the maximum possible
entropy therefore grows with ``t``. Reporting **normalised** entropy
(divided by ``ln(t+1)``, the uniform-distribution maximum) puts every position
on a 0-1 scale where 1 means "uniform over everything visible" -- otherwise
later positions would look artificially more diffuse purely because they see
more.

**Mean attention distance** measures how *far back* a head looks: the
attention-weighted mean of ``t - j`` over key positions ``j <= t``. A head with
distance near 0 attends to the immediately preceding tokens (local, often
positional); a large distance means it reaches far back into the context
(long-range, more often content-driven). This is the statistic that separates
the two head types the spec asks to discuss.

Both are averaged over queries and batch, with position 0 excluded: that row can
only attend to itself, giving entropy exactly 0 and distance exactly 0 by
construction, so including it drags every head's average toward zero for a
reason that says nothing about what the head learned.

Plotting and computation are kept in separate functions per the course's code
guidelines -- :func:`head_statistics` returns numbers, :func:`plot_heatmaps`
draws. Every figure carries a title and axis labels, since the spec states
plots missing labels may receive zero.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

__all__ = ["collect_attention", "head_statistics", "plot_heatmaps",
           "plot_head_summary", "main"]


# Fonts that actually carry Devanagari and Bengali glyphs, most-preferred first.
# Windows ships Nirmala UI; Kaggle's Linux image usually has a Noto Indic face.
# matplotlib's default (DejaVu Sans) has neither, and silently renders missing
# glyphs as empty boxes -- so axis labels would be unreadable rather than wrong
# in any way that raises. Hence explicit resolution with a visible fallback.
_INDIC_FONT_CANDIDATES = (
    "Nirmala UI", "Noto Sans Devanagari", "Noto Serif Devanagari",
    "Noto Sans Bengali", "Noto Serif Bengali", "Lohit Devanagari",
    "Lohit Bengali", "Mangal", "Vrinda", "FreeSerif",
)


def resolve_indic_font() -> str | None:
    """Return an installed font name that can render Indic scripts, or ``None``.

    Checked against matplotlib's own font list rather than assumed, because the
    answer differs between the Windows machine that built the corpus and the
    Kaggle Linux image that trains on it.
    """
    import matplotlib.font_manager as fm
    installed = {f.name for f in fm.fontManager.ttflist}
    for name in _INDIC_FONT_CANDIDATES:
        if name in installed:
            return name
    return None


def _axis_labels(tokens: list[str], font: str | None) -> list[str]:
    """Token labels if a usable font exists, else positional indices.

    Falling back to ``t0, t1, ...`` is deliberate: boxes of tofu look like a
    broken figure, whereas indices are honest and still let a reader line the
    axes up against the example text printed in the caption/JSON.

    SentencePiece marks a preceding space with U+2581 (LOWER ONE EIGHTH BLOCK).
    Indic text fonts carry the letters but not that block character, so it is
    rewritten to a plain underscore -- otherwise every word-initial token would
    render with one tofu box regardless of how good the font's script coverage
    is. Same substitution the Phase 1 report uses for its worked examples.
    """
    if font is None:
        return [f"t{i}" for i in range(len(tokens))]
    return [t.replace("▁", "_") for t in tokens]


@torch.no_grad()
def collect_attention(model, idx: torch.Tensor) -> list[torch.Tensor]:
    """Run the model and return per-layer attention probabilities.

    Args:
        model: A :class:`common.modeling.model.TransformerLM`.
        idx: ``(B, T)`` token ids.

    Returns:
        List of ``n_layer`` tensors, each ``(B, n_head, T, T)``.
    """
    model.eval()
    _, _, attns = model(idx, return_attn=True)
    return attns


def head_statistics(attns: list[torch.Tensor]) -> dict:
    """Per-head normalised entropy and mean attention distance.

    Args:
        attns: Output of :func:`collect_attention`.

    Returns:
        Dict holding ``entropy`` and ``distance``, each a nested list indexed
        ``[layer][head]``, plus per-layer means for the report's summary table.
    """
    ent_by_layer, dist_by_layer = [], []

    for a in attns:
        # a: (B, H, T, T); a[b, h, t, j] = weight query t puts on key j.
        B, H, T, _ = a.shape
        device = a.device
        pos = torch.arange(T, device=device)

        # Shannon entropy per (b, h, t), in nats.
        ent = -(a.clamp_min(1e-12) * a.clamp_min(1e-12).log()).sum(dim=-1)  # (B,H,T)

        # Normalise by ln(t+1), the maximum entropy available at position t.
        max_ent = torch.log(pos.float() + 1).clamp_min(1e-12)               # (T,)
        ent_norm = ent / max_ent

        # Distance: sum_j a[...,t,j] * (t - j).
        offsets = pos.view(1, 1, T, 1) - pos.view(1, 1, 1, T)               # (1,1,T,T)
        dist = (a * offsets.clamp_min(0)).sum(dim=-1)                        # (B,H,T)

        # Drop position 0 (degenerate: attends only to itself).
        if T > 1:
            ent_norm = ent_norm[:, :, 1:]
            dist = dist[:, :, 1:]

        ent_by_layer.append(ent_norm.mean(dim=(0, 2)).tolist())   # per head
        dist_by_layer.append(dist.mean(dim=(0, 2)).tolist())

    return {
        "entropy": ent_by_layer,
        "distance": dist_by_layer,
        "layer_mean_entropy": [sum(l) / len(l) for l in ent_by_layer],
        "layer_mean_distance": [sum(l) / len(l) for l in dist_by_layer],
        "n_layer": len(attns),
        "n_head": len(ent_by_layer[0]) if ent_by_layer else 0,
    }


def plot_heatmaps(attns: list[torch.Tensor], layers: list[int], heads: list[int],
                  tokens: list[str], out_path: Path | str, language: str) -> Path:
    """Draw query-vs-key attention heatmaps for selected layers and heads.

    Spec section 2.3 asks for at least one early layer, one late layer, and
    multiple heads, with a colour scale and example sentences in the model's
    language -- hence the token labels on both axes.

    Args:
        attns: Per-layer attention from :func:`collect_attention`.
        layers: Layer indices to draw (rows of the figure grid).
        heads: Head indices to draw (columns).
        tokens: Decoded token strings for axis ticks.
        out_path: PNG destination.
        language: Used in the figure title.
    """
    import matplotlib
    matplotlib.use("Agg")                     # headless: Kaggle has no display
    import matplotlib.pyplot as plt

    font = resolve_indic_font()
    if font:
        plt.rcParams["font.family"] = font
    labels = _axis_labels(tokens, font)

    T = min(len(tokens), attns[0].shape[-1])
    fig, axes = plt.subplots(len(layers), len(heads),
                             figsize=(4.2 * len(heads), 4.0 * len(layers)),
                             squeeze=False)

    for r, li in enumerate(layers):
        for c, hi in enumerate(heads):
            ax = axes[r][c]
            m = attns[li][0, hi, :T, :T].detach().cpu().numpy()
            im = ax.imshow(m, cmap="viridis", aspect="auto",
                           vmin=0.0, vmax=float(m.max()))
            ax.set_title(f"Layer {li} · Head {hi}", fontsize=10)
            ax.set_xlabel("Key position (attended to)", fontsize=8)
            ax.set_ylabel("Query position (attending from)", fontsize=8)
            step = max(1, T // 12)
            ax.set_xticks(range(0, T, step))
            ax.set_yticks(range(0, T, step))
            ax.set_xticklabels([labels[i] for i in range(0, T, step)],
                               rotation=90, fontsize=6)
            ax.set_yticklabels([labels[i] for i in range(0, T, step)], fontsize=6)
            fig.colorbar(im, ax=ax, fraction=0.046, label="Attention weight")

    note = "" if font else "  (tick labels shown as positions: no Indic font available)"
    fig.suptitle(f"{language.title()} — causal self-attention "
                 f"(lower triangle only, by construction){note}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_head_summary(stats: dict, out_path: Path | str, language: str) -> Path:
    """Plot per-head entropy and mean distance as layer-by-head heatmaps.

    Gives the whole model's head behaviour in one figure, which is what the
    report's "which heads are local vs content-based" discussion draws on.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ent = np.array(stats["entropy"])
    dist = np.array(stats["distance"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, data, title, cbar in (
        (axes[0], ent, "Normalised attention entropy\n(1 = uniform, 0 = fully focused)",
         "Entropy / ln(t+1)"),
        (axes[1], dist, "Mean attention distance\n(tokens back from the query)",
         "Distance (tokens)"),
    ):
        im = ax.imshow(data, cmap="magma", aspect="auto")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Head index", fontsize=9)
        ax.set_ylabel("Layer index", fontsize=9)
        ax.set_xticks(range(data.shape[1]))
        ax.set_yticks(range(data.shape[0]))
        fig.colorbar(im, ax=ax, fraction=0.046, label=cbar)

    fig.suptitle(f"{language.title()} — per-head attention summary", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI. Example::

        python -m common.modeling.attention --lang hindi --checkpoint hindi/model/best.pt
    """
    import argparse
    import sentencepiece as spm

    from common.modeling.data import TokenArray
    from common.modeling.model import ModelConfig, TransformerLM

    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("--lang", required=True, choices=("hindi", "assamese"))
    p.add_argument("--checkpoint", default="")
    p.add_argument("--data-dir", default="data/tokenized")
    p.add_argument("--split", default="val")
    p.add_argument("--seq-len", type=int, default=48,
                   help="tokens shown in the heatmaps; longer is unreadable")
    p.add_argument("--out-dir", default="")
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

    # One real held-out passage, so the heatmap shows genuine language.
    start = len(arr) // 2
    ids = [int(t) for t in arr.tokens[start : start + args.seq_len]]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    tokens = [sp.id_to_piece(i) for i in ids]

    attns = collect_attention(model, idx)
    stats = head_statistics(attns)

    out_dir = Path(args.out_dir or f"{args.lang}/eval")
    early, late = 0, mcfg.n_layer - 1
    heads = list(range(min(3, mcfg.n_head)))
    hm = plot_heatmaps(attns, [early, late], heads, tokens,
                       out_dir / f"attention_heatmaps_{args.lang}.png", args.lang)
    sm = plot_head_summary(stats, out_dir / f"attention_summary_{args.lang}.png",
                           args.lang)

    stats_out = {"language": args.lang, "checkpoint": str(ckpt_path),
                 "checkpoint_step": ckpt.get("step"),
                 "example_text": sp.decode(ids), **stats}
    (out_dir / f"attention_stats_{args.lang}.json").write_text(
        json.dumps(stats_out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"{args.lang}: layers={stats['n_layer']} heads={stats['n_head']}")
    print(f"  layer mean entropy : "
          f"{[round(x,3) for x in stats['layer_mean_entropy']]}")
    print(f"  layer mean distance: "
          f"{[round(x,2) for x in stats['layer_mean_distance']]}")
    print(f"  wrote {hm}\n  wrote {sm}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
