"""Empirical checks that the from-scratch Transformer behaves as specified.

Spec section 2.1 does not merely ask for a causal mask -- it asks to *"verify
empirically that the model cannot see the future -- for example, show that
changing token t+1 does not change the logits at position t."* That is a
falsifiable claim about the implementation, so it is tested here rather than
asserted in prose, and the numeric evidence goes into the Phase 2 report.

Run directly::

    python -m common.modeling.verify_model

Every check prints its own measured evidence and returns a pass/fail, so a
regression in the attention implementation surfaces as a failing number rather
than as quietly worse training curves days later.
"""

from __future__ import annotations

import torch

from common.modeling.model import ModelConfig, TransformerLM

__all__ = ["check_causality", "check_parameter_count", "check_shapes",
           "check_tied_embeddings", "check_mask_rows_sum_to_one", "main"]


def _tiny_config() -> ModelConfig:
    """Small config: these checks probe structure, not scale."""
    return ModelConfig(vocab_size=256, block_size=32, n_layer=2, n_head=4,
                       n_embd=64, d_ff=256, dropout=0.0)


def check_causality(seed: int = 0) -> bool:
    """Changing token ``t+1`` must not change the logits at position ``t``.

    The decisive test for the causal mask. Builds a sequence, records the
    logits, then rewrites **only** the token at position ``t+1`` and recomputes.
    Positions ``0..t`` must be bit-for-bit unchanged (they cannot attend to
    ``t+1``), while position ``t+1`` itself must change (it obviously can see
    its own input). Checking only the first half would pass trivially on a model
    whose output ignored the input entirely, so both halves are asserted.

    Dropout is disabled via ``eval()`` -- otherwise the two passes would differ
    for reasons unrelated to masking and the test would be meaningless.
    """
    torch.manual_seed(seed)
    cfg = _tiny_config()
    model = TransformerLM(cfg).eval()

    T = 16
    t = 7                                        # the position we protect
    idx = torch.randint(0, cfg.vocab_size, (1, T))

    with torch.no_grad():
        logits_a, _ = model(idx)

    idx_b = idx.clone()
    # Force a genuinely different token at t+1.
    idx_b[0, t + 1] = (idx[0, t + 1] + 1) % cfg.vocab_size
    with torch.no_grad():
        logits_b, _ = model(idx_b)

    past_delta = (logits_a[0, : t + 1] - logits_b[0, : t + 1]).abs().max().item()
    changed_delta = (logits_a[0, t + 1] - logits_b[0, t + 1]).abs().max().item()

    past_ok = past_delta == 0.0
    future_ok = changed_delta > 0.0
    print(f"  positions 0..{t} max|delta| = {past_delta:.3e}  "
          f"(must be exactly 0) -> {'PASS' if past_ok else 'FAIL'}")
    print(f"  position {t+1} max|delta| = {changed_delta:.3e}  "
          f"(must be > 0)       -> {'PASS' if future_ok else 'FAIL'}")
    return past_ok and future_ok


def check_mask_rows_sum_to_one(seed: int = 0) -> bool:
    """Post-softmax attention rows must still sum to 1 after masking.

    Guards the specific bug of zeroing probabilities *after* softmax instead of
    adding ``-inf`` *before* it: that variant leaves rows summing to less than
    one, silently down-weighting every attended value. Also confirms the strict
    upper triangle is exactly zero.
    """
    torch.manual_seed(seed)
    cfg = _tiny_config()
    model = TransformerLM(cfg).eval()
    T = 12
    idx = torch.randint(0, cfg.vocab_size, (2, T))
    with torch.no_grad():
        _, _, attns = model(idx, return_attn=True)

    a = attns[0]                                  # (B, n_head, T, T), first layer
    row_sums = a.sum(dim=-1)
    sums_ok = torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)

    upper = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    future_mass = a[:, :, upper].abs().max().item()
    upper_ok = future_mass == 0.0

    print(f"  attention row sums: max|sum-1| = "
          f"{(row_sums - 1).abs().max().item():.3e} -> {'PASS' if sums_ok else 'FAIL'}")
    print(f"  strict upper triangle max weight = {future_mass:.3e} "
          f"(must be exactly 0) -> {'PASS' if upper_ok else 'FAIL'}")
    return sums_ok and upper_ok


def check_shapes(seed: int = 0) -> bool:
    """Forward pass produces the documented shapes, and the loss backpropagates."""
    torch.manual_seed(seed)
    cfg = _tiny_config()
    model = TransformerLM(cfg)
    B, T = 3, 16
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    targets = torch.randint(0, cfg.vocab_size, (B, T))

    logits, loss = model(idx, targets)
    logits_ok = tuple(logits.shape) == (B, T, cfg.vocab_size)
    loss_ok = loss.ndim == 0 and torch.isfinite(loss)

    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    grad_ok = all(g is not None and torch.isfinite(g).all() for g in grads)

    # An untrained model over V classes should sit near ln(V).
    import math
    expected = math.log(cfg.vocab_size)
    near_uniform = abs(loss.item() - expected) < 0.5

    print(f"  logits shape {tuple(logits.shape)} == (B,T,V) -> "
          f"{'PASS' if logits_ok else 'FAIL'}")
    print(f"  initial loss {loss.item():.3f} vs ln(V)={expected:.3f} -> "
          f"{'PASS' if near_uniform else 'FAIL'}")
    print(f"  all grads finite -> {'PASS' if grad_ok else 'FAIL'}")
    return logits_ok and loss_ok and grad_ok and near_uniform


def check_tied_embeddings() -> bool:
    """The output head must be the *same tensor* as the token embedding.

    ``is`` rather than ``allclose``: equal-valued copies would still double the
    parameter cost, which is exactly what tying exists to avoid.
    """
    cfg = _tiny_config()
    model = TransformerLM(cfg)
    tied = model.lm_head.weight is model.tok_emb.weight
    print(f"  lm_head.weight is tok_emb.weight -> {'PASS' if tied else 'FAIL'}")
    return tied


def check_parameter_count(target: int = 25_000_000, tol: float = 0.10) -> bool:
    """The real config must land within ``tol`` of the ~25M spec target."""
    cfg = ModelConfig.load("hindi/configs/model.json")
    model = TransformerLM(cfg)
    rep = model.parameter_report()
    n = rep["total_trainable"]
    ok = abs(n - target) / target <= tol
    print(f"  total trainable = {n:,} (target ~{target:,}, "
          f"{abs(n-target)/target:.1%} off) -> {'PASS' if ok else 'FAIL'}")
    print(f"  embedding share = {rep['embedding_share_of_total']:.1%} "
          f"| blocks = {rep['transformer_blocks']:,}")
    return ok


def check_block_size_guard() -> bool:
    """Sequences longer than ``block_size`` must raise, not silently truncate."""
    cfg = _tiny_config()
    model = TransformerLM(cfg).eval()
    too_long = torch.randint(0, cfg.vocab_size, (1, cfg.block_size + 1))
    try:
        with torch.no_grad():
            model(too_long)
    except ValueError:
        print("  over-length sequence raises ValueError -> PASS")
        return True
    print("  over-length sequence did NOT raise -> FAIL")
    return False


def main(argv: list[str] | None = None) -> int:
    """Run every check; exit non-zero if any fails."""
    checks = [
        ("causality (spec 2.1: cannot see the future)", check_causality),
        ("attention mask well-formedness", check_mask_rows_sum_to_one),
        ("forward/backward shapes and loss", check_shapes),
        ("tied embeddings share one tensor", check_tied_embeddings),
        ("block_size guard", check_block_size_guard),
        ("parameter count ~25M", check_parameter_count),
    ]
    results = []
    for title, fn in checks:
        print(f"\n=== {title} ===")
        results.append((title, bool(fn())))

    failed = [t for t, ok in results if not ok]
    print("\n" + "=" * 58)
    if failed:
        print(f"FAILED ({len(failed)}): " + "; ".join(failed))
        return 1
    print(f"ALL {len(results)} CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
