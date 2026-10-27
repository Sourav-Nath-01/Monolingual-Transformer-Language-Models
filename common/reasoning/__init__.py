"""Phase 3: synthetic comparative-reasoning data, finetuning, and its evaluation.

Kept separate from ``common/modeling`` because the boundary runs the same way:
these modules import ``common.modeling`` (to reuse the Transformer and the
training loop) but nothing in ``common.modeling`` imports them. Phase 2 stays
exactly as it was graded.
"""
