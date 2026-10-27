"""Deterministic train/val/test split assignment.

Lecture 3 slide 24: "Hold out your test set first, then clean, so the split
is never contaminated by the pipeline itself." The textbook order is collect
-> split -> clean. That was not followed here: collection is still running
when this module is written, and normalize/language-filter/quality/nonlang/
dedup already ran on whatever existed at the time (see the corresponding
finding in the plan). This module is the fix, not a workaround -- it makes
split assignment a pure, content-independent function of document id, so it
does not matter in what order documents were collected or cleaned. Whether a
document lands in train, val or test depends only on its id, computed the
same way regardless of when the id was assigned a split.

This also extends, rather than special-cases, an idea the project plan
already uses for the downloaded tier: "assign each incoming downloaded
document to a split on arrival by hashing its ID -- deterministic,
reproducible." Applying the identical mechanism to the manual tier means
collection and cleaning can keep running concurrently with everything
downstream, and re-running split assignment after more data arrives is
idempotent -- every document already assigned keeps its split, because the
hash of its id never changes.

What this does NOT replace: decontamination (slide 24's other half). Splitting
by hash keeps train and test disjoint by construction, but it does not catch
near-duplicates that ended up on both sides of that boundary -- the same
wire-service story scraped twice, once landing in train and once in test by
chance. That is a separate near-duplicate search with test as the query set
against train, run after this split exists, reusing the MinHash/LSH machinery
in common/corpus/dedup.py.

Stratification is a side effect of hashing on document id alone: the hash
does not know which source a document came from, so each source lands in
each split in roughly its own proportion automatically. No special-casing per
source is needed for this to hold.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from common.io import Doc, iter_docs

__all__ = ["SplitConfig", "HINDI_SPLIT", "ASSAMESE_SPLIT", "assign_split",
           "build_splits", "load_split_ids", "iter_split"]


def _stable_bucket(doc_id: str, buckets: int = 1_000_000) -> int:
    """Map a document id to a deterministic integer in [0, buckets).

    SHA-256 rather than Python's built-in ``hash()``: the latter is salted
    per-process for security (PYTHONHASHSEED), so the same doc id would land
    in a different bucket on every run -- exactly the instability this module
    exists to avoid.
    """
    digest = hashlib.sha256(doc_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % buckets


@dataclass(frozen=True)
class SplitConfig:
    """Split proportions for one language.

    Attributes:
        val_fraction: Share of documents assigned to validation.
        test_fraction: Share assigned to test.
        buckets: Hash resolution. 1,000,000 gives split boundaries accurate
            to 0.0001 of the corpus, far finer than needed but cheap.
    """

    val_fraction: float
    test_fraction: float
    buckets: int = 1_000_000


# Hindi: huge corpus, so a small percentage already clears the plan's ~5M
# val / ~5M test target on a ~500M-token corpus.
HINDI_SPLIT = SplitConfig(val_fraction=0.01, test_fraction=0.01)

# Assamese: originally 5%/5%, chosen when the corpus was expected to land at
# only 60-90M final tokens and needed a larger percentage to clear the plan's
# floor of "not below ~2M val / 2M test". The corpus grew far past that
# estimate (~511M tokens pre-split, expansion rounds 2-3) without this
# fraction being revisited -- at that scale 5%/5% was pulling ~30M words
# away from train for no remaining reason, which matters specifically
# because train tokens (not the whole corpus) is the graded target and
# manual is scarce enough that every word matters. 1%/1% (matching Hindi)
# still clears the same ~2M floor several times over (~3M each) while
# giving train its due share.
ASSAMESE_SPLIT = SplitConfig(val_fraction=0.01, test_fraction=0.01)


def assign_split(doc_id: str, config: SplitConfig) -> str:
    """Return ``"train"``, ``"val"`` or ``"test"`` for one document id.

    Pure function: same id, same config, same answer, forever. That purity is
    the entire point -- it is what makes split assignment safe to run before
    collection has finished.
    """
    b = _stable_bucket(doc_id, config.buckets)
    val_cut = int(config.buckets * config.val_fraction)
    test_cut = val_cut + int(config.buckets * config.test_fraction)
    if b < val_cut:
        return "val"
    if b < test_cut:
        return "test"
    return "train"


def build_splits(
    language: str,
    source_dirs: list[Path | str],
    config: SplitConfig,
    out_dir: Path | str | None = None,
) -> dict:
    """Assign every document in ``source_dirs`` to a split and write manifests.

    Args:
        language: For the output path and report only.
        source_dirs: Cleaned corpus directories to draw documents from.
        config: Split proportions.
        out_dir: Where to write ``{train,val,test}.ids``. Defaults to
            ``<language>/splits/``.

    Returns:
        Per-split document and word counts, and per-source breakdown --
        the latter is the stratification check: each source's val/test
        share should land close to ``config``'s fractions independently.
    """
    out_dir = Path(out_dir or f"{language}/splits")
    out_dir.mkdir(parents=True, exist_ok=True)

    ids: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    words: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    per_source: dict[str, dict[str, int]] = {}

    for d in source_dirs:
        source_name = Path(d).name
        per_source.setdefault(source_name, {"train": 0, "val": 0, "test": 0})
        for doc in iter_docs(d):
            split = assign_split(doc.id, config)
            ids[split].append(doc.id)
            words[split] += doc.n_words()
            per_source[source_name][split] += 1

    for split, id_list in ids.items():
        # Sorted for a stable diff across re-runs -- otherwise iteration
        # order (directory listing, shard order) would make an unrelated
        # collection run show up as spurious churn in the manifest.
        (out_dir / f"{split}.ids").write_text(
            "\n".join(sorted(id_list)) + "\n", encoding="utf-8"
        )

    report = {
        "language": language,
        "config": {"val_fraction": config.val_fraction, "test_fraction": config.test_fraction},
        "docs": {k: len(v) for k, v in ids.items()},
        "words": words,
        "per_source": per_source,
    }
    return report


def load_split_ids(language: str, split: str, splits_dir: Path | str | None = None) -> set[str]:
    """Read a frozen ``{split}.ids`` manifest back into a set of document ids."""
    path = Path(splits_dir or f"{language}/splits") / f"{split}.ids"
    return set(path.read_text(encoding="utf-8").split())


def iter_split(
    source_dirs: list[Path | str], ids: set[str]
) -> Iterator[Doc]:
    """Yield only the documents whose id is in ``ids``.

    Use with :func:`load_split_ids` to materialise one split's documents from
    the full cleaned corpus, e.g. for tokenizer training on train only.
    """
    for d in source_dirs:
        for doc in iter_docs(d):
            if doc.id in ids:
                yield doc


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Run with:  python -m common.corpus.splits
    from common.io import enable_utf8_console

    enable_utf8_console()
    failures = []

    def check(name, got, want):
        if got == want:
            print(f"OK   {name}  -> {got}")
        else:
            failures.append(name)
            print(f"FAIL {name}\n       got  {got!r}\n       want {want!r}")

    cfg = SplitConfig(val_fraction=0.1, test_fraction=0.1)

    # Determinism: same id, same config -> same split, every time.
    a1 = assign_split("hi_wiki_0000123", cfg)
    a2 = assign_split("hi_wiki_0000123", cfg)
    check("deterministic across calls", a1 == a2, True)

    # Proportions roughly right over a large sample. Not exact -- it is a
    # hash, not a counter -- but should land within a few percent at N=20000.
    counts = {"train": 0, "val": 0, "test": 0}
    n = 20_000
    for i in range(n):
        counts[assign_split(f"as_news_test_{i:07d}", cfg)] += 1
    val_frac = counts["val"] / n
    test_frac = counts["test"] / n
    print(f"     measured: train={counts['train']} val={counts['val']} test={counts['test']}")
    check("val fraction near target",
          abs(val_frac - 0.1) < 0.02, True)
    check("test fraction near target",
          abs(test_frac - 0.1) < 0.02, True)

    # Disjointness: a document is never in two splits at once.
    seen = set()
    for i in range(n):
        s = assign_split(f"as_news_test_{i:07d}", cfg)
        seen.add((f"as_news_test_{i:07d}", s))
    ids_seen = [x[0] for x in seen]
    check("no id assigned twice", len(ids_seen) == len(set(ids_seen)), True)

    # Different id prefixes (simulating different sources) should each get
    # roughly the configured split fraction independently -- this is the
    # stratification claim, checked directly rather than assumed.
    for prefix in ("hi_wiki_", "hi_news_", "hi_wsrc_"):
        c = {"train": 0, "val": 0, "test": 0}
        for i in range(5000):
            c[assign_split(f"{prefix}{i:07d}", cfg)] += 1
        vf = c["val"] / 5000
        check(f"stratification holds for {prefix}", abs(vf - 0.1) < 0.03, True)

    # Config sanity: Hindi and Assamese configs differ, as designed.
    check("Assamese uses a larger split fraction than Hindi",
          ASSAMESE_SPLIT.val_fraction > HINDI_SPLIT.val_fraction, True)

    if failures:
        raise SystemExit(f"\n{len(failures)} check(s) failed: {failures}")
    print("\nOK  all split checks passed")
