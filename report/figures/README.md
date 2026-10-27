# Report figures

Every figure that appears in any report in `report/`, kept here as well so that
all plots live under `report/` and nothing has to be hunted for elsewhere in the
tree — the spec asks for exactly that ("Keep every figure and table that you
want graded inside `report/`").

These are copies. The originals are written by the pipeline into
`<lang>/model/`, `<lang>/eval/` and `<lang>/train/finetuned/`, alongside the
JSON files holding the numbers each figure is drawn from.

## Phase 2 — pretraining and attention

| Figure | Report § | Source | Underlying numbers |
|---|---|---|---|
| `loss_curve_hindi.png` | P2 §2.3 | `hindi/model/` | `hindi/model/train_log.jsonl` |
| `loss_curve_assamese.png` | P2 §2.3 | `assamese/model/` | `assamese/model/train_log.jsonl` |
| `attention_heatmaps_hindi.png` | P2 §5 | `hindi/eval/` | `hindi/eval/attention_stats_hindi.json` |
| `attention_heatmaps_assamese.png` | P2 §5 | `assamese/eval/` | `assamese/eval/attention_stats_assamese.json` |
| `attention_summary_hindi.png` | P2 §5 | `hindi/eval/` | `hindi/eval/attention_stats_hindi.json` |
| `attention_summary_assamese.png` | P2 §5 | `assamese/eval/` | `assamese/eval/attention_stats_assamese.json` |

**Loss curves** — training loss logged every 20 steps, validation every 250,
over the full 6,000-step run for each language.

**Attention heatmaps** — query vs. key positions for an early layer (0) and a
late layer (9), three heads each, on a real 48-token passage from that model's
own held-out split. Axis labels are the actual tokens, in Devanagari for Hindi
and Bengali–Assamese script for Assamese. Everything above the diagonal is
exactly zero: the causal mask, visible.

**Attention summaries** — per-head normalised entropy and mean attention
distance across all 10 layers × 6 heads.

## Phase 3 — finetuning and the pretrained/finetuned comparison

| Figure | Report § | Source | Underlying numbers |
|---|---|---|---|
| `finetune_curve_hindi.png` | P3 §5.3 | `hindi/train/finetuned/` | `hindi/train/finetuned/finetune_log.jsonl` |
| `finetune_curve_assamese.png` | P3 §5.3 | `assamese/train/finetuned/` | `assamese/train/finetuned/finetune_log.jsonl` |
| `attn_ft_pretrained_hindi.png` | P3 §7 | `hindi/eval/` | `hindi/eval/attention_stats_pretrained_hindi.json` |
| `attn_ft_finetuned_hindi.png` | P3 §7 | `hindi/eval/` | `hindi/eval/attention_stats_finetuned_hindi.json` |
| `attn_ft_pretrained_assamese.png` | P3 §7 | `assamese/eval/` | `assamese/eval/attention_stats_pretrained_assamese.json` |
| `attn_ft_finetuned_assamese.png` | P3 §7 | `assamese/eval/` | `assamese/eval/attention_stats_finetuned_assamese.json` |
| `attn_ft_summary_pretrained_hindi.png` | P3 §7 | `hindi/eval/` | same as above |
| `attn_ft_summary_finetuned_hindi.png` | P3 §7 | `hindi/eval/` | same as above |
| `attn_ft_summary_pretrained_assamese.png` | P3 §7 | `assamese/eval/` | same as above |
| `attn_ft_summary_finetuned_assamese.png` | P3 §7 | `assamese/eval/` | same as above |

**Finetuning curves** — training loss and validation exact match over the
3,750-step run. The exact-match line here is teacher-forced argmax, used for
model selection; the numbers quoted in the report are stricter free generation.

**Pretrained vs. finetuned attention** — the `pretrained`/`finetuned` pairs are
probed on the **same fixed transitive prompt** (`PROMPT_INDEX = 0`) for a given
language, so the two figures are comparable cell by cell and every difference is
attributable to the weights rather than the input. The two languages use their
own prompts, as they must, since they have their own tokenizers.

## Regenerating

```bash
# Phase 2 attention
python -m common.modeling.attention --lang hindi          # and --lang assamese

# Phase 3 attention, one call per (language, checkpoint) pair
python -m common.reasoning.attention_compare --lang hindi \
    --checkpoint hindi/model/best.pt           --tag pretrained --out-dir hindi/eval
python -m common.reasoning.attention_compare --lang hindi \
    --checkpoint hindi/train/finetuned/best.pt --tag finetuned  --out-dir hindi/eval
```

Run these **locally, not on Kaggle**: Kaggle's image has no Indic font
installed, so matplotlib silently falls back to `t0, t4, …` for the axis labels
instead of rendering the actual tokens. The statistics are identical either
way — only the labels differ. The committed figures were produced locally with
Nirmala UI available.

The metric tables in the reports are generated from `<lang>/eval/*.json`, which
also hold the generated text samples and the per-example successes and failures.
