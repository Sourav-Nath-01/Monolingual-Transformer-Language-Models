"""Phase 2: the Transformer, its training loop, and its evaluation.

These six modules form a self-contained unit: they import each other and
nothing else from ``common``, and nothing else in ``common`` imports them. That
one-way boundary is what keeps the package clean -- the Phase 1 data pipeline
and the Phase 2 modelling code are genuinely independent, so this is the only
part of ``common/`` that needs no ``common.io`` at all and stays torch-only.

Note which module is deliberately *not* here:
``common/tokenization/tokenize_corpus.py``. It is the bridge between the phases
-- it reads the Phase 1 corpus through ``common.io`` and writes the token
arrays this package trains on -- so placing it here would drag a ``common.io``
dependency into an otherwise torch-only package and break that boundary.

Modules:
    model        Decoder-only Transformer built from primitive PyTorch layers.
    verify_model Empirical checks, incl. the spec's causality requirement.
    data         Memory-mapped batching over the uint16 token arrays.
    train        AdamW + cosine schedule + resume-capable checkpointing.
    evaluate     Perplexity, bits-per-byte, BLEU/chrF/ROUGE-L, diversity.
    attention    Heatmaps, per-head entropy, mean attention distance.
"""
