[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Q6gOCxoh)

# CL3410 — Building and Analyzing Monolingual Transformer LMs in Two Indian Languages

Two fully independent decoder-only Transformer language models, built from scratch.

| | Language | Script | Role |
|---|---|---|---|
| **Model H** | Hindi (`hi`) | Devanagari (U+0900–U+097F) | higher-resource |
| **Model L** | Assamese (`as`) | Bengali–Assamese (U+0980–U+09FF) | lower-resource |

Separate corpus, separate tokenizer, separate vocabulary, separate weights. Nothing is shared between them except code.

**Branches:** `phase-1` (data + tokenizer) · `phase-2` (pretraining) · `phase-3` (finetuning + consolidated final report) · **`bonus-ablation`** (optional bonus, built on `phase-3`)

> ### The optional bonus is done, on branch [`bonus-ablation`](../../tree/bonus-ablation)
>
> **Ablation: no positional embeddings.** Hindi retrained from scratch with the learned
> positional-embedding table removed — not zeroed, never allocated — with corpus, tokenizer, seed,
> schedule and the full 589,824,000-token budget held identical. Only 196,608 parameters differ.
>
> | | Standard | Ablated |
> |---|---|---|
> | Parameters | 24,086,016 | 23,889,408 |
> | Test bits-per-byte | 0.5595 | **0.5656** (+1.1%) |
>
> Almost nothing breaks, and a linear probe shows why: position is still recoverable from the final
> hidden state at R² 0.8121 against the standard model's 0.8183, so the ablated model keeps **99.2%
> of its positional decodability with zero positional parameters**. Causal masking already encodes
> position — a query at index *t* attends to exactly *t*+1 keys, and that count is itself a
> measurement of where it sits.
>
> Full Phase 2 evaluation suite, attention comparison, and a test that *failed* (with why it was the
> wrong measurement) in **`report/bonus_report.pdf`** on that branch.

---

## Phase 1 — final numbers

Exact, from the real selected tokenizers — see `report/phase1_report.pdf` §6:

| | Train tokens | Manual % |
|---|---|---|
| Hindi | 589,952,668 | 20.80% |
| Assamese | 591,255,819 | 23.88% |

Both languages clear the ~500M-token target and the 20% manual floor. **Vocabulary size note:** the tokenizer sweep initially selected Hindi 32k / Assamese 24k by minimizing fertility alone; course guidance on Phase 2's ~25M-parameter budget clarified that the embedding layer counts fully against it, and 32k would have consumed up to 98% of that budget. The immediate fix moved both languages to 8k (the safest point in the recommended 5k-10k range). Once that guidance was clarified as a recommendation rather than a strict rule, the choice was revisited once more on engineering merit and settled on **16k vocabulary with tied input/output embeddings** — keeping the embedding budget reasonable (24.6% of 25M tied) while recovering meaningfully better fertility and less subword fragmentation than 8k. See `report/phase1_report.pdf` §8.1 for the full three-stage correction.

## Phase 2 — model

Two independent decoder-only Transformers, built from `nn.Linear` /
`nn.Embedding` / `nn.LayerNorm` only — no `nn.Transformer*`, no HuggingFace
model classes, no pre-built attention. Identical architecture per language,
separate weights:

| | Value |
|---|---|
| Parameters | **24,086,016** (~25M target) |
| Layers / heads / `d_model` | 10 / 6 / 384 |
| Feed-forward width | 1536 (4x `d_model`) |
| Context length | 512 tokens |
| Positional encoding | learned absolute |
| Normalisation | pre-norm |
| Embeddings | **tied** input/output — 25.5% of parameters |

Tied embeddings are the reason Phase 1 settled on a 16k vocabulary: untied, the
embedding matrices alone would take ~49% of the budget. Verify the build with
`python -m common.modeling.verify_model`.

### Pretraining results

Both models saw an identical budget — 6,000 steps of 98,304 tokens = **589,824,000
tokens**, about one epoch — trained separately on a Tesla T4 in fp16.

| | Model H (Hindi) | Model L (Assamese) |
|---|---|---|
| Training time | 290 min | 263 min |
| Final train loss | 3.896 | 4.718 |
| Val CE / PPL / BPB | 3.8564 / 47.30 / 0.5537 | 4.5478 / 94.42 / 0.6204 |
| **Test CE / PPL / BPB** | **3.8864 / 48.73 / 0.5595** | **4.5556 / 95.16 / 0.6238** |

Bits-per-byte is the comparable metric across the two languages: perplexity is
per *token*, and the two tokenizers segment text differently, so a token of
Hindi and a token of Assamese are not the same unit. Note also that perplexity
is `exp(CE)`, which magnifies differences — the 0.669 nats/token gap reads as
1.95x in perplexity but 1.11x in bits-per-byte. Both describe the same gap; BPB
is the one to quote.

Test loss sits marginally above validation for both models, as it should: `best.pt`
is selected on validation, so the validation figure is slightly optimistic.

Figures and raw numbers live in `<lang>/model/` and `<lang>/eval/`; the analysis
is in `report/`.

---

## Phase 3 — reasoning finetuning

Each model is finetuned on its **own** synthetic reasoning corpus, generated
programmatically in its own language and script by
`common/reasoning/generate.py`. Three task families — two-way comparison
(`compare2`), three-way ordering (`order3`), and transitive chaining
(`transitive`, no numbers at all, pure `A<B, B<C` inference). 40,000 train /
2,000 val / 2,000 test / 1,000 `test_unseen` per language.

**Leakage is controlled two different ways, as two separate test splits**, because
they measure different things:

| Split | Relation patterns | Entity names | Measures |
|---|---|---|---|
| `test` | **held out** (0 of 3 seen in train) | shared | generalising over *phrasing* |
| `test_unseen` | shared | **held out** (0 of 14 seen in train) | generalising over *entities* |

### Reasoning results

Free generation — the model is given only the prompt, generates greedily until
EOS, and the decoded entity must match exactly. Chance is computed from the
number of candidate entities named in each prompt.

| | Model H | Model L | chance |
|---|---|---|---|
| Pretrained | 0.0000 | 0.0000 | 0.4512 / 0.4488 |
| **Finetuned** | **0.8460** | **0.7555** | |
| &nbsp;&nbsp;transitive | 1.0000 | 1.0000 | 0.5000 |
| &nbsp;&nbsp;compare2 | 0.9600 | 0.8928 | 0.5000 |
| &nbsp;&nbsp;order3 | 0.5282 | 0.3436 | 0.3333 |
| `test_unseen` | 0.0120 | 0.0000 | 0.4492 / 0.4510 |

Three findings worth calling out:

- **Both models solve transitive chaining perfectly** (1.0000 on 615 and 584
  held-out items). The resource tier is irrelevant for the task that most
  resembles multi-hop reasoning.
- **The whole 9-point gap is one task.** The models are identical on transitive
  and 6.7 points apart on `compare2`. On `order3`, Model L's 0.3436 sits on its
  0.3333 chance baseline, with Model H at 0.5282.

  That looks like Model L never learned three-way ordering, but it did. Scoring
  the **validation** split with the same free-generation method gives Model L
  **0.7719** on `order3` and Model H **0.9381**, both far above chance
  (`<lang>/eval/reasoning_val.json`). Validation reuses the training templates
  while `test` holds its templates out, so what collapses is not the skill but
  its transfer to one unseen phrasing — a 43-point drop for Model L, 41 for
  Model H. Because `test` carries a single template per task, task difficulty is
  confounded with which phrasing was held out; the Model H vs Model L
  comparison is unaffected, since both saw identical prompts.
- **Both models collapse on unfamiliar entities** (0.0120 / 0.0000 against ~0.45
  chance), while scoring 0.85/0.76 on held-out *phrasings*. The comparison
  operation is learned but welded to the memorised entity set — a capacity
  limit, not a data-tier one.

Pretrained accuracy is exactly zero because nothing in pretraining taught these
models that a question has a short answer that ends; prefix match is nonzero
(0.2100 / 0.0715), i.e. they sometimes name the right entity and keep going.

### What made the task learnable

Three fixes, each made in response to a measured failure rather than in advance:

1. **Answer-only loss** — prompt positions masked to `-100`.
2. **EOS inside the supervised span.** Without it: 0.0005 exact vs 0.5710 prefix
   on transitive — right entity, never stops.
3. **Rationale targets**, which is what actually made it work. Supervising only
   the final entity left both models at chance (0.4355 vs a 0.4512 baseline).
   Supervising the decisive value first — `22 साल, इसलिए दीपक` ("22 years,
   therefore Deepak") — moved Hindi to 0.8460. Only the final entity is scored,
   so showing correct working and then drawing the wrong conclusion still
   scores zero.

### Error analysis

`common/reasoning/error_analysis.py` scores the two halves of each prediction
independently, using the generator's entity→value map: did the model pick the
right *value*, and did it attach it to the right *owner*?

| | Model H | Model L |
|---|---|---|
| correct | 0.8460 | 0.7555 |
| binding failure (right value, wrong owner) | 0.1290 | 0.2245 |
| comparison failure (wrong value) | 0.0250 | 0.0200 |
| **binding share of all errors** | **83.8%** | **91.8%** |
| **value selected correctly** | **0.9750** | **0.9800** |

**This overturns the natural reading of the accuracy table.** Both models
perform the comparison itself at indistinguishable rates — 97.50% and 98.00%,
with the *lower*-resource model marginally ahead (40 comparison errors against
Hindi's 50). The 9-point exact-match gap is entirely in keeping track of *which
entity owns which value*, not in comparing numbers.

Conditioned on the comparison succeeding, `P(owner right | value right)`:

| Task | Model H | Model L | chance |
|---|---|---|---|
| transitive (no values to bind) | 1.0000 | 1.0000 | — |
| compare2 (2 candidates) | 0.9961 | 0.9109 | 0.5000 |
| order3 (3 candidates) | 0.5479 | **0.3576** | **0.3333** |

On `order3` Model L identifies the right value 96.1% of the time and then names
its owner 35.8% of the time against 33.3% chance — **it knows the answer and
guesses whose it is.** Performance on all three tasks, for both models, is
predicted by how many value→entity bindings the task requires: zero, one, two.

Full numbers in `<lang>/eval/error_analysis_test.json` and report §6.

### Attention, pretrained vs finetuned

`common/reasoning/attention_compare.py` probes both checkpoints on the **same
fixed** transitive prompt, so the figures are comparable cell by cell.
Finetuning made late-layer attention markedly more local and left early layers
alone:

| Mean attention distance | Model H | Model L |
|---|---|---|
| early, L0–L4 | 6.51 → 6.72 (+0.21) | 6.38 → 6.62 (+0.24) |
| **late, L5–L9** | **9.50 → 7.73 (−1.76)** | **9.86 → 6.66 (−3.20)** |
| heads that became shorter-range | 38 / 60 | 42 / 60 |

Early layers do local, positional work that reasoning finetuning has no reason
to change. The late layers had been surveying a 512-token context for general
language modelling; in a reasoning prompt everything needed is nearby and
specific, so they stop surveying and start fixating. Model L moves almost twice
as far, consistent with it being the weaker base model.

---

## Repository layout

```
common/          shared CODE only — never data, never vocabulary
  io.py          Doc dataclass, shard read/write, atomic output
  pipeline.py    orchestrates clean → filter → dedup for one source
  sources/       one module per data source (10)
  clean/         normalise, language-ID, quality filter
  corpus/        dedup, splits, decontamination
  tokenization/  SentencePiece training; corpus → uint16 arrays
  modeling/      Phase 2: Transformer, training, evaluation, attention
  reasoning/     Phase 3: data generation, finetuning, scoring, error analysis
hindi/           self-contained
  configs/       model.json — architecture hyperparameters (committed)
  tokenizer/     .model / .vocab files                (committed)
  splits/        val.ids + test.ids committed; train.ids on Drive (too large)
  stats/         generated statistics                 (committed)
  model/         training logs, loss curves, parameter report (committed);
                 .pt checkpoints on Drive (too large)
  eval/          metrics JSON, attention heatmaps and summaries, reasoning
                 scores, error analysis                (committed)
  train/         reasoning splits (val/test/test_unseen committed, train
                 regenerable), finetuning config + log + curve (committed);
                 finetuned .pt checkpoints on Drive (too large)
assamese/        same structure, fully independent
kaggle/          notebooks that run Phase 2 pretraining and Phase 3 finetuning
report/          final_report.pdf — consolidated report covering all three
                 phases; the Phase 1 and Phase 2 reports are kept alongside it
  figures/       every plot and heatmap referenced by any report, as PNGs
```

> **`common/` holds code, not artifacts.** It contains the collectors (scrapers, dump parsers, OCR) plus normalisation, filtering, dedup and statistics functions that both languages call — shared rather than duplicated per-language, since both languages run the same collectors and pipeline stages with different parameters. It contains no corpus text, no vocabulary, and nothing language-specific. The independence constraint is about *data, tokenizers, vocabularies and weights* — sharing a `normalize_nfc()` function does not violate it, and duplicating that function into two directories would only make the code worse.

`common/` is grouped into subpackages by pipeline stage. Only two modules stay at
the top level: `io.py`, which every other module builds on, and `pipeline.py`,
which chains the cleaning stages together. The dependency graph is deliberately
shallow — twenty of the twenty-two modules import nothing from `common/` except
`io`, so a stage can be read, changed or replaced on its own.

| Package | Modules | Role |
|---|---|---|
| *(top level)* | `io.py` | `Doc` dataclass, shard read/write, UTF-8 and atomic-write handling |
| *(top level)* | `pipeline.py` | runs normalise → filter → quality → nonlang → dedup for one source |
| **`sources/`** | `wikidump.py`, `scrape.py`, `scrape_sites.py`, `archive_books.py` | manually collected: Wikipedia/Wikisource, news sites, archive.org OCR |
| | `downloaded.py`, `indiccorp_v2.py`, `madlad.py`, `glot500.py`, `xahitya.py`, `assamese_mono_csv.py` | one per HuggingFace/external corpus |
| **`clean/`** | `normalize.py` → `filters.py` → `quality.py` → `nonlang.py` | the per-document cleaning chain, applied in that order |
| | `eval_language_id.py` | validates the Assamese/Bengali discriminator used by `filters.py` |
| **`corpus/`** | `dedup.py`, `splits.py`, `decontaminate.py` | corpus-wide: MinHash dedup, train/val/test assignment, leakage check |
| **`tokenization/`** | `tokenizer.py` | vocabulary sweep, training and selection |
| | `tokenize_corpus.py` | Phase 1 → 2 bridge: corpus text → `uint16` token arrays |
| **`modeling/`** | `model.py`, `verify_model.py` | Transformer, MHA, causal mask, tied head; the empirical correctness checks |
| | `data.py`, `train.py` | memmapped batching; AdamW, cosine LR, resume-capable checkpoints |
| | `evaluate.py`, `attention.py` | PPL/BPB, BLEU/chrF/ROUGE-L, diversity; heatmaps, entropy, mean distance |
| **`reasoning/`** | `generate.py` | synthetic reasoning corpus per language — all templates and entity pools in one reviewable block |
| | `data.py`, `finetune.py` | answer-masked batching; finetuning from the pretrained checkpoint, resume-capable |
| | `evaluate.py`, `error_analysis.py` | free-generation exact match vs chance; failure classification by value/owner |
| | `attention_compare.py` | pretrained vs finetuned attention on a fixed prompt, reusing `modeling/attention.py` |

### Which tokenizer files are the deliverables

`*/tokenizer/` holds the full vocabulary sweep, not just the final pick — the sweep is the evidence
behind the selection (spec §1.3: *"choose using fertility / unknown-token rate on held-out text"*).
The **selected tokenizer for each language is the 16k one**:

| File | Role |
|---|---|
| `hindi/tokenizer/hindi_bpe_16000.model` + `.vocab` | **Model H deliverable** — the tokenizer used for every token count in the report |
| `assamese/tokenizer/assamese_bpe_16000.model` + `.vocab` | **Model L deliverable** — same, for Assamese |
| `*_bpe_{8000,24000,32000,48000,64000}.*` | The other five swept sizes, kept as evidence for the fertility/UNK comparison in report §8.1 |
| `*_nofallback.*` | Variants trained with `byte_fallback=False`, used only to measure the true UNK rate (with byte-fallback on, UNK is trivially ~0% by construction) |
| `sweep_timing.json` | Per-size training times from the sweep |

Vocabularies are **not shared** between the two languages — they are separately trained on each
language's own train split, and overlap only on punctuation/digits/control tokens (~1.6%).

## What lives where

| Artifact | Location | Why |
|---|---|---|
| Reports (PDF), tables | **git** | graded; must be in the branch |
| Tokenizer `.model` / `.vocab` | **git** | ≤2 MB each, small enough to commit |
| `val.ids` / `test.ids` | **git** | small (~0.5–1 MB); Phases 2–3 must reproduce identical splits |
| Statistics JSON | **git** | graded deliverable |
| Training logs, loss curves, `config.json`, `parameter_report.json` | **git** | ~200 KB per language; spec requires *"training logs and loss curves in the repo"* |
| Metrics JSON (val + test), attention figures and stats | **git** | ~280 KB per language; graded deliverable |
| Reasoning `val` / `test` / `test_unseen` splits | **git** | ~1 MB per language; the reported scores must be reproducible exactly |
| Reasoning scores, error analysis, finetuning log + curve | **git** | graded deliverable |
| Reasoning `train` split (40,000 rows) | **neither** | regenerated deterministically by `common.reasoning.generate` in one command |
| `train.ids` | **Google Drive** | 56 MB (Hindi) / 33 MB (Assamese); regenerable as corpus − val − test |
| Corpus text (raw + cleaned) | **Google Drive** | ~6.4 GB compressed, too large for git |
| Pretrained checkpoints `best.pt` / `last.pt` | **Google Drive** | 299.7 MB each; spec forbids large binaries in git |
| Finetuned checkpoints `best.pt` / `last.pt` | **Google Drive** | 299.7 MB each; same reason |
| Tokenized `uint16` arrays (`data/tokenized/`) | **neither** | 2.3 GB, and fully regenerable from the corpus in one command |

### Google Drive links

Shared as **anyone with the link → viewer**, so it opens without requesting access.

| Artifact | Link |
|---|---|
| Hindi + Assamese corpus (raw + clean_deduped + train.ids, both languages) | [Google Drive folder](https://drive.google.com/drive/folders/1jLpgQBmY3gGCWfqxP_Kz_4MQvQHxoGJS?usp=sharing) — contains `Hindi corpus/` and `Assamese corpus/` subfolders |
| Phase 2 pretrained checkpoints (Model H and Model L) | [Google Drive folder](https://drive.google.com/drive/folders/1bIOG67ZM5iARPTvk3FGR51DxqIIxlLLn?usp=sharing) — `hindi/` and `assamese/`, each with `best.pt` and `last.pt` (299.7 MB each), plus a `README.txt` describing the payload and how to load it |
| Phase 3 **finetuned** checkpoints (Model H and Model L) | [Google Drive folder](https://drive.google.com/drive/folders/1W9mShiOj3JrejAjZgH6QvGRv99J4ksfK?usp=sharing) — `hindi/` and `assamese/`, each with `best.pt` and `last.pt` (299.7 MB each), same resume-capable format |

---

## Reproduction

### Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Windows note — this one matters.** Python defaults `open()` to the system ANSI codepage (cp1252), not UTF-8. Writing Devanagari or Bengali through a default file handle produces mojibake that looks fine until the tokenizer trains on it. Set:

```powershell
setx PYTHONUTF8 1        # permanent; reopen the shell afterwards
```

All project code additionally passes `encoding="utf-8"` explicitly, and `common/io.py` asserts a UTF-8 round-trip on every shard it writes. Both layers are deliberate.

Tesseract (OCR) needs the separate UB-Mannheim Windows installer, with `hin.traineddata` and `asm.traineddata` in `tessdata/` and `TESSDATA_PREFIX` set.

### Pipeline

Every stage is idempotent and resumable (shard + `.done`-marker based), so re-running any command
after an interruption resumes rather than restarting. Every `--lang` command is run **once per
language** — `hindi` and `assamese` are collected, cleaned, split and tokenized entirely separately.

```powershell
# 1a. Wikipedia / Wikisource -- own extractor over the official XML dumps.
#     Takes explicit dump/output paths (not --lang); run once per dump.
python -m common.sources.wikidump --dump data/dumps/hiwiki-latest-pages-articles.xml.bz2 `
                          --out data/hindi/raw/wiki --source hiwiki-20260814 --prefix hi_wiki
python -m common.sources.wikidump --dump data/dumps/aswiki-latest-pages-articles.xml.bz2 `
                          --out data/assamese/raw/wiki --source aswiki-20260814 --prefix as_wiki
#     (Wikisource: same command with the *wikisource* dump, --namespaces 250 for Hindi / 104 for Assamese)

# 1b. News scrapers (all configured sites for that language)
python -m common.sources.scrape_sites --lang hindi
python -m common.sources.scrape_sites --lang assamese

# 1c. Archive.org OCR text layer
python -m common.sources.archive_books --lang hindi
python -m common.sources.archive_books --lang assamese

# 1d. Downloaded tier -- Sangraha (both languages), then the Assamese-only sources
python -m common.sources.downloaded --lang hindi
python -m common.sources.downloaded --lang assamese
python -m common.sources.indiccorp_v2      # Assamese only
python -m common.sources.madlad            # Assamese only
python -m common.sources.glot500           # Assamese only
python -m common.sources.xahitya           # Assamese only
python -m common.sources.assamese_mono_csv # Assamese only

# 2. Clean + dedup (normalize -> language-id -> quality -> non-language-words -> dedup)
python -m common.pipeline --lang hindi --dedup
python -m common.pipeline --lang assamese --dedup

# 3. Splits (deterministic hash of document id; see common/corpus/splits.py -- library API, no CLI)
python -c "from common.corpus.splits import build_splits, SplitConfig; ..."

# 4. Decontamination (train vs val/test near-duplicate check)
python -m common.corpus.decontaminate --lang hindi
python -m common.corpus.decontaminate --lang assamese

# 5. Tokenizer: extract train text, sweep vocab sizes, measure + select
python -m common.tokenization.tokenizer --lang hindi    --extract --sweep --select
python -m common.tokenization.tokenizer --lang assamese --extract --sweep --select

# 6. Language-ID validation (Assamese/Bengali discriminator, 100-doc labelled set)
python -m common.clean.eval_language_id
```

Exact source lists, thresholds, and per-round provenance are in `sources.yaml`; the full corrective history (bugs found, decisions made, and why) is in the project's working plan document referenced from `report/phase1_report.pdf` §9.


### Phase 2 — pretraining (runs on a Kaggle GPU)

Phase 1's `### Pipeline` above is CPU work that produced the corpus. Phase 2
trains two ~24.1M-parameter models, which needs a GPU, so it runs on Kaggle
rather than locally. The only step that stays local is tokenisation.

```powershell
# 0. LOCAL: turn the corpus into compact uint16 token arrays.
#    6.4 GB of text -> 2.3 GB of ids, which is what gets uploaded to Kaggle.
python -m common.tokenization.tokenize_corpus --lang hindi
python -m common.tokenization.tokenize_corpus --lang assamese
#    Then upload data/tokenized/ as a private Kaggle Dataset named `lma-tokens`.

# 0b. LOCAL: stage the code payload and upload it as a second private Kaggle
#     Dataset named `lma-code`. This repository is private, so the notebook
#     cannot `git clone` it -- a Kaggle kernel has no credentials.
python kaggle/stage_code_payload.py

# 1. Sanity-check the implementation anywhere (CPU is fine, ~10 seconds).
#    Proves empirically that the model cannot attend to the future (spec 2.1).
python -m common.modeling.verify_model
```

Everything below runs **inside `kaggle/phase2_train_kaggle.ipynb`** — open it on
Kaggle, attach **both** the `lma-tokens` and `lma-code` datasets, enable the GPU
(*Settings → Accelerator*), set `LANG`, and run. Internet is optional: it is
needed only to fetch `sacrebleu` for BLEU/chrF; every other metric is computed
without it. The same commands work locally on a CUDA machine:

```powershell
# 2. Pretrain. Resumes automatically from last.pt if a previous run was cut short.
#    6,000 steps x 98,304 tokens = 589,824,000 tokens, about one epoch. This is
#    what the committed results were produced with (~4.8 h per model on a T4).
python -m common.modeling.train --lang hindi    --max-steps 6000
python -m common.modeling.train --lang assamese --max-steps 6000

# 3. Evaluate: cross-entropy, perplexity, bits-per-byte, BLEU-4, chrF,
#    ROUGE-L, Distinct-1/2, repetition rate, generated samples.
#    Run for both splits -- val is what model selection used, test is untouched.
python -m common.modeling.evaluate --lang hindi    --checkpoint hindi/model/best.pt --split val
python -m common.modeling.evaluate --lang hindi    --checkpoint hindi/model/best.pt --split test
python -m common.modeling.evaluate --lang assamese --checkpoint assamese/model/best.pt --split val
python -m common.modeling.evaluate --lang assamese --checkpoint assamese/model/best.pt --split test

# 4. Attention analysis: per-head heatmaps, entropy, mean attention distance.
python -m common.modeling.attention --lang hindi    --checkpoint hindi/model/best.pt
python -m common.modeling.attention --lang assamese --checkpoint assamese/model/best.pt
```

### Phase 3 — reasoning finetuning

Finetuning is only ~3 minutes per model on a T4, so unlike Phase 2 this phase is
cheap to re-run end to end. `kaggle/phase3_finetune_kaggle.ipynb` does all of it
in one pass; the equivalent CLI sequence is:

```bash
# 1. Generate the reasoning data (deterministic given the seed).
#    Writes all four splits: train / val / test / test_unseen.
python -m common.reasoning.generate --lang hindi
python -m common.reasoning.generate --lang assamese

# 2. Score the PRETRAINED model first -- this is the baseline the finetuning
#    gain is measured against, so it has to be taken before finetuning.
python -m common.reasoning.evaluate --lang hindi --pretrained hindi/model/best.pt \
    --split test --out hindi/eval/reasoning_pretrained.json

# 3. Finetune from that language's own pretrained checkpoint, its own tokenizer
#    and vocabulary held fixed. 3 epochs, 3,750 steps, ~2.9 min on a T4.
#    Resumes from last.pt if a previous run was interrupted.
python -m common.reasoning.finetune --lang hindi    --pretrained hindi/model/best.pt
python -m common.reasoning.finetune --lang assamese --pretrained assamese/model/best.pt

# 4. Score pretrained vs finetuned on both held-out splits. Free generation,
#    not teacher forcing -- exact match on the decoded entity.
python -m common.reasoning.evaluate --lang hindi --pretrained hindi/model/best.pt \
    --finetuned hindi/train/finetuned/best.pt --split test
python -m common.reasoning.evaluate --lang hindi --pretrained hindi/model/best.pt \
    --finetuned hindi/train/finetuned/best.pt --split test_unseen
#    (same two commands for assamese)

# 5. Error analysis: why the failures fail.
python -m common.reasoning.error_analysis --lang hindi    --checkpoint hindi/train/finetuned/best.pt
python -m common.reasoning.error_analysis --lang assamese --checkpoint assamese/train/finetuned/best.pt

# 6. Attention, pretrained vs finetuned, on the SAME fixed prompt so the two
#    figures are comparable cell by cell.
python -m common.reasoning.attention_compare --lang hindi \
    --checkpoint hindi/model/best.pt           --tag pretrained --out-dir hindi/eval
python -m common.reasoning.attention_compare --lang hindi \
    --checkpoint hindi/train/finetuned/best.pt --tag finetuned  --out-dir hindi/eval
#    (same two commands for assamese)
```

The committed attention figures were regenerated **locally** rather than taken
from the Kaggle run: Kaggle's image has no Indic font installed, so matplotlib
silently falls back to `t0, t4, …` for the axis labels. Running step 6 on a
machine with Nirmala UI or Noto available renders the actual tokens. The
statistics are identical either way — only the labels differ.

**Why Kaggle, and how interruption is handled.** A Kaggle session is capped at
~12 hours and can be evicted sooner, so `common/modeling/train.py` treats interruption as
routine rather than exceptional: every checkpoint carries model weights,
optimizer state, scheduler state (as the step — the cosine schedule is a pure
function of it), and both configs, written atomically via a `.tmp` rename so an
eviction mid-write cannot corrupt it. `last.pt` is overwritten on a step
interval for resuming; `best.pt` is written only when validation improves, so a
late divergence cannot destroy the best model. The notebook's markdown explains
how to chain sessions using Kaggle's notebook-output-as-input mechanism.

---

## Design notes

**Language choice.** Hindi has the largest presence in every public Indic corpus, so ~500M clean tokens is reachable. Assamese was chosen over the other allowed lower-resource languages because it shares the Bengali–Assamese script with Bengali — a far larger language — which makes language identification a *measurable* problem rather than an assumed one, and because Hindi and Assamese occupy **different Unicode blocks**, so corpus independence is structural rather than threshold-dependent.

**Assamese's token target.** For most of collection, Assamese was projected to fall short of 500M tokens — Lecture 2 slide 23 anticipates exactly this for a genuinely low-resource language, and the spec permits it provided the exact count is reported and justified. In the end, sustained manual collection (six expansion rounds, including a second archive.org pass that surfaced a real limitation in that API's search pagination — see report §9) closed the gap: Assamese finished at 591.26M tokens, above target, with manual share (23.88%) higher than Hindi's. The corpus is not padded with duplication or synthetic data at any point — see `report/phase1_report.pdf` §6 for the full corrective history.

**Token counts.** Every token figure in the statistics report is measured with that language's *final selected tokenizer*, not a proxy. Word-count proxies were used only to steer collection while the tokenizer did not yet exist.
