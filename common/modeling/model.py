"""Decoder-only (GPT-style) Transformer, built from primitive PyTorch layers.

Spec section 2.1 forbids ``nn.Transformer*``, HuggingFace model classes, and any
pre-built attention block: only ``nn.Linear``, ``nn.Embedding``, ``nn.LayerNorm``
and ``nn.Dropout`` are permitted, and multi-head attention, positional
embeddings and causal masking must be student-implemented. Every tensor
operation in the forward pass below is therefore written out explicitly, with
shapes annotated at each step, because the phase is assessed on being able to
explain what each one computes.

Architecture choices, each stated with its reason:

**Learned absolute positional embeddings.**
    Chosen over sinusoidal for one practical reason and one honest limitation.
    Practical: with a fixed 512-token context and hundreds of millions of
    training tokens, there is ample data to learn position representations, and
    learned embeddings let the model allocate capacity to the positions it
    actually sees rather than to a fixed frequency basis. The limitation, which
    matters and is not hidden: a learned table has exactly ``block_size`` rows,
    so the model **cannot** process a sequence longer than the context it was
    trained on -- position 512 has no embedding to look up and the forward pass
    raises rather than silently degrading. Sinusoidal encodings and RoPE both
    extrapolate past their training length; learned ones do not. The trade is
    accepted because Phase 2 and Phase 3 both evaluate at the training context
    length, so extrapolation is never exercised.

**Pre-norm, not post-norm.**
    LayerNorm is applied to each sublayer's *input* (``x + attn(ln(x))``) rather
    than to its output (``ln(x + attn(x))``). Pre-norm leaves an unnormalised
    residual path running clean from embedding to final norm, so gradients reach
    early layers without passing through a LayerNorm at every step. Post-norm
    stacks tend to need learning-rate warmup and careful initialisation to stay
    stable at depth; pre-norm is markedly more forgiving, which matters when a
    10-layer stack gets one training run on a limited GPU budget rather than the
    luxury of restarts. A final ``ln_f`` before the output head compensates for
    the fact that pre-norm leaves the last block's output unnormalised.

**Tied input/output embeddings.**
    ``lm_head.weight`` *is* ``tok_emb.weight`` -- the same tensor, not a copy.
    This was committed to in Phase 1 (report section 8.1) as the specific
    justification for choosing a 16k vocabulary over 8k: tied, 16k costs
    ``16000 x 384`` = 6.14M parameters (24.6% of the ~25M budget); untied it
    would cost that twice over, 49.2%, which is the crowding-out problem the
    vocabulary correction existed to avoid. Beyond the parameter saving there is
    a modelling argument: the input embedding maps a token id to a vector and
    the output head scores a hidden state against every token, so both learn a
    representation of the same vocabulary -- tying makes that shared structure
    explicit rather than asking the model to learn it twice from scratch.

**Causal masking, and why the scaling factor matters** are documented at their
implementation sites in :class:`CausalSelfAttention`, since both are pure
mechanics rather than architecture choices.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ModelConfig", "CausalSelfAttention", "FeedForward", "Block",
           "TransformerLM"]


@dataclass
class ModelConfig:
    """Hyperparameters defining one model. Serialised alongside every checkpoint.

    Defaults target ~25M trainable parameters at a 16,000 vocabulary, which is
    what both languages use. The depth/width split is justified in
    :meth:`TransformerLM.parameter_report`.

    Attributes:
        vocab_size: Tokenizer vocabulary. Sets embedding and output-head width.
        block_size: Context length in tokens. Hard cap -- learned positional
            embeddings cannot extrapolate past this (see module docstring).
        n_layer: Number of stacked Transformer blocks.
        n_head: Attention heads per block. Must divide ``n_embd``.
        n_embd: Model dimension ``d_model``.
        d_ff: Feed-forward inner width. Conventionally ``4 * n_embd``.
        dropout: Applied to embeddings, attention probabilities and sublayer
            outputs.
        tie_embeddings: Share the token embedding with the output head.
    """

    vocab_size: int = 16000
    block_size: int = 512
    n_layer: int = 10
    n_head: int = 6
    n_embd: int = 384
    d_ff: int = 1536
    dropout: float = 0.1
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head "
                f"({self.n_head}); each head takes n_embd // n_head dimensions"
            )

    @property
    def head_dim(self) -> int:
        """Per-head subspace width, ``d_k = n_embd / n_head``."""
        return self.n_embd // self.n_head

    def save(self, path: Path | str) -> None:
        """Write config as JSON so a checkpoint is reproducible from the README."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "ModelConfig":
        """Rebuild a config from its JSON file."""
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention, implemented from first principles.

    For input ``X`` of shape ``(B, T, d)`` this computes, per head:

        Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k) + M) @ V

    where ``M`` is an additive mask holding ``0`` on allowed positions and
    ``-inf`` on future ones.

    **Why divide by sqrt(d_k).** ``Q @ K^T`` sums ``d_k`` products of roughly
    unit-variance terms, so the dot products have variance proportional to
    ``d_k`` and grow as the head dimension grows. Feeding large-magnitude logits
    to softmax drives it toward a one-hot distribution, where the gradient is
    almost everywhere near zero -- attention saturates and stops learning.
    Dividing by ``sqrt(d_k)`` rescales the logits back to unit variance, keeping
    softmax in its responsive range. Without it, this model (``d_k = 64``) would
    see logits about 8x larger and train far worse.

    **Why the mask is additive and applied before softmax.** Adding ``-inf``
    before the softmax makes those positions contribute exactly zero weight
    *after* normalisation, so the remaining weights still sum to one. Zeroing
    the probabilities after softmax instead would leave rows that no longer sum
    to one, silently rescaling every attended value. The mask is registered as a
    buffer, not a parameter: it is fixed geometry, must not receive gradients,
    but should still follow the model across ``.to(device)`` and be saved with
    the checkpoint.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.head_dim

        # One fused projection producing Q, K and V together: mathematically
        # identical to three separate Linears, one matmul instead of three.
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)

        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        # Lower-triangular ones: row t has 1s in columns 0..t. Stored as a
        # buffer so it moves with the model and is not a learnable parameter.
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        self.register_buffer("causal_mask", mask.view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        """Run masked multi-head attention.

        Args:
            x: ``(B, T, d)`` input activations.
            return_attn: Also return the post-softmax attention probabilities,
                shape ``(B, n_head, T, T)``. Used by the Phase 2 attention
                analysis (heatmaps, entropy, mean distance); off during
                training since materialising them costs memory for nothing.

        Returns:
            ``(B, T, d)`` output, or ``(output, attn)`` when ``return_attn``.
        """
        B, T, C = x.shape

        # (B, T, 3d) -> three (B, T, d) tensors.
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)

        # Split the channel dim into heads and move heads ahead of time, so the
        # matmuls below act on (T, head_dim) matrices independently per head:
        #   (B, T, d) -> (B, T, n_head, head_dim) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # (B, h, T, hd) @ (B, h, hd, T) -> (B, h, T, T); scale per the docstring.
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Additive causal mask over the top-left (T, T) corner of the buffer,
        # so shorter-than-block_size sequences are handled without rebuilding it.
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))

        att = F.softmax(att, dim=-1)
        attn_weights = att if return_attn else None
        att = self.attn_dropout(att)

        # (B, h, T, T) @ (B, h, T, hd) -> (B, h, T, hd)
        y = att @ v

        # Concatenate heads back: (B, h, T, hd) -> (B, T, h, hd) -> (B, T, d).
        # .contiguous() is required because transpose only changes strides, and
        # .view() needs a contiguous buffer to reinterpret.
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_dropout(self.proj(y))
        return (y, attn_weights) if return_attn else y


class FeedForward(nn.Module):
    """Position-wise feed-forward network: ``Linear -> GELU -> Linear -> Dropout``.

    Applied identically and independently at every position -- it mixes
    *channels*, never positions, which is precisely why attention is the only
    component that moves information across the sequence. The inner width is
    wider than ``n_embd`` (4x by default) so the non-linearity operates in a
    higher-dimensional space before projecting back, which is where most of the
    block's parameters and representational capacity live.

    GELU rather than ReLU: it is smooth near zero rather than hard-clipping, so
    small-magnitude activations pass a usable gradient instead of being killed
    outright, and it is the activation the GPT family established for this
    architecture.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, cfg.d_ff)
        self.proj = nn.Linear(cfg.d_ff, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, T, d) -> (B, T, d)``, via ``(B, T, d_ff)``."""
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """One pre-norm Transformer block: masked attention, then feed-forward.

    Both sublayers are wrapped in residual connections::

        x = x + attn(ln_1(x))
        x = x + ffn(ln_2(x))

    The residual additions are what make depth trainable: they give gradients an
    identity path back to earlier layers, so a 10-block stack does not attenuate
    signal the way 10 chained transformations otherwise would. Remove them and
    the model still runs but becomes materially harder to optimise -- the
    classic deep-network degradation the residual connection was invented for.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.ffn = FeedForward(cfg)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        """``(B, T, d) -> (B, T, d)``, optionally also returning attention."""
        if return_attn:
            attn_out, attn_w = self.attn(self.ln_1(x), return_attn=True)
            x = x + attn_out
            x = x + self.ffn(self.ln_2(x))
            return x, attn_w
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x


class TransformerLM(nn.Module):
    """Decoder-only Transformer language model.

    Forward pass, in the order spec section 2.1 lays out:

    1. token embedding + learned positional embedding, summed, then dropout
    2. ``n_layer`` pre-norm blocks of masked multi-head attention + FFN
    3. final LayerNorm
    4. linear projection to vocabulary logits (weight-tied to the token
       embedding), trained with causal cross-entropy against the next token
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            # Same tensor object, so one gradient accumulates from both uses
            # and the optimiser updates a single set of weights.
            self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

        # GPT-2's scaled init for residual-path output projections: with
        # n_layer residual additions accumulating into the stream, variance
        # grows with depth unless these are shrunk by 1/sqrt(2 * n_layer).
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Normal(0, 0.02) for Linear/Embedding weights; zero Linear biases."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_attn: bool = False,
    ):
        """Map token ids to next-token logits, and optionally the LM loss.

        Args:
            idx: ``(B, T)`` int64 token ids, ``T <= block_size``.
            targets: ``(B, T)`` ids shifted one position left. When given, the
                causal LM cross-entropy is returned alongside the logits.
            return_attn: Collect per-layer attention probabilities.

        Returns:
            ``(logits, loss)``, or ``(logits, loss, attns)`` when
            ``return_attn``. ``loss`` is ``None`` if ``targets`` is ``None``.

        Raises:
            ValueError: If ``T`` exceeds ``block_size`` -- the learned
                positional table has no row for those positions. Raising is
                deliberate: silently truncating would corrupt evaluation.
        """
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(
                f"sequence length {T} exceeds block_size {self.cfg.block_size}; "
                "learned positional embeddings cannot extrapolate (see module docstring)"
            )

        pos = torch.arange(T, device=idx.device)          # (T,)
        x = self.tok_emb(idx) + self.pos_emb(pos)          # (B,T,d) + (T,d) broadcast
        x = self.drop(x)

        attns = []
        for block in self.blocks:
            if return_attn:
                x, a = block(x, return_attn=True)
                attns.append(a)
            else:
                x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)                           # (B, T, vocab)

        loss = None
        if targets is not None:
            # Flatten batch and time so cross_entropy sees one prediction per
            # position: (B*T, vocab) against (B*T,).
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1)
            )

        return (logits, loss, attns) if return_attn else (logits, loss)

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count parameters, counting a tied weight once (``set`` by identity)."""
        seen, total = set(), 0
        for p in self.parameters():
            if trainable_only and not p.requires_grad:
                continue
            if id(p) in seen:
                continue
            seen.add(id(p))
            total += p.numel()
        return total

    def parameter_report(self) -> dict:
        """Break the parameter count down by component, for the Phase 2 report.

        Reports the embedding share explicitly because that ratio is the whole
        reason Phase 1's vocabulary was revised twice: it is the number that
        decides how much of a fixed budget is left for the layers that actually
        model language.
        """
        cfg = self.cfg
        emb = self.tok_emb.weight.numel()
        pos = self.pos_emb.weight.numel()
        blocks = sum(p.numel() for b in self.blocks for p in b.parameters())
        final = sum(p.numel() for p in self.ln_f.parameters())
        head = 0 if cfg.tie_embeddings else self.lm_head.weight.numel()
        total = self.num_parameters()
        return {
            "total_trainable": total,
            "token_embedding": emb,
            "positional_embedding": pos,
            "transformer_blocks": blocks,
            "final_layernorm": final,
            "output_head": head,
            "embeddings_tied": cfg.tie_embeddings,
            "embedding_share_of_total": round(emb / total, 4),
            "config": asdict(cfg),
        }

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        greedy: bool = False,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        """Autoregressively extend ``idx`` by up to ``max_new_tokens``.

        Args:
            idx: ``(B, T)`` prompt ids.
            max_new_tokens: How many tokens to append.
            temperature: Divides the logits before softmax. Below 1 sharpens the
                distribution (more deterministic), above 1 flattens it (more
                diverse). Ignored when ``greedy``.
            greedy: Take the argmax instead of sampling.
            eos_id: Stop early once every sequence in the batch has emitted this.

        Returns:
            ``(B, T + n)`` ids including the prompt.
        """
        self.eval()
        finished = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)
        for _ in range(max_new_tokens):
            # Feed at most the last block_size tokens -- the positional table
            # has no row beyond that.
            window = idx[:, -self.cfg.block_size:]
            logits, _ = self(window)
            logits = logits[:, -1, :]                       # (B, vocab)
            if greedy:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)
            if eos_id is not None:
                # Once finished, keep emitting eos so shapes stay rectangular.
                nxt = torch.where(finished.unsqueeze(1),
                                  torch.full_like(nxt, eos_id), nxt)
                finished |= nxt.squeeze(1) == eos_id
            idx = torch.cat([idx, nxt], dim=1)
            if eos_id is not None and bool(finished.all()):
                break
        return idx
