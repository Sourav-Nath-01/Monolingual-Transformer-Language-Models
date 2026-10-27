"""Decontamination: remove train documents that near-duplicate test (or val).

Lecture 3 slide 24's other half, distinct from `common/corpus/dedup.py`'s
corpus-internal near-dup pass: *"Hold out your test set first, then clean...
a paraphrased test item still leaks"* -- ordinary dedup collapses duplicate
clusters to one survivor without knowing which split that survivor lands in,
so a near-duplicate pair can still end up split across train and test. This
module is the targeted check for exactly that: index train, query it with
every val/test document, and drop whatever train matches.

**Reuses `common/corpus/dedup.py`'s MinHash primitives directly** -- same shingling,
same `max_shingles=3000` cap `common/pipeline.py`'s `run_dedup_pass` uses,
not `NearDupConfig`'s bare defaults (`max_shingles=20_000`) -- rather than
reimplementing near-dup matching. This was a real inconsistency the first
version of this module had despite its own stated intent: it took whatever
`NearDupConfig()` defaulted to instead of matching the corpus-wide dedup
pass's actual config, which means the two passes were judging similarity by
two different thresholds of shingle sampling without either of them saying
so. Fixed to default to the same config `run_dedup_pass` builds, so a
document judged "near-duplicate enough to collapse" in one context can't
silently be judged "distinct enough to keep" in the other for no principled
reason -- which is exactly the bug this module's own docstring already
warned against.

**Asymmetric by design.** Train is indexed once; val/test are query-only and
never modified -- a match means the *train* copy leaked and must go, not that
the held-out document is somehow at fault. This also keeps cost down: only
one corpus (train) pays the O(n) LSH insert cost, val/test pay only the O(1)
query cost per document.

**Signature caching**, same mechanism and same correctness guard (content
hash verified before trusting a cache hit) as `run_dedup_pass` -- see
`common/corpus/dedup.py`'s cache functions. Kept in a separate cache file from the
main dedup pass rather than shared, since sharing would need the two
passes's configs to be identical for every cached entry to be valid, and
this module's own thresholds may still be tuned independently of the main
pass's in the future.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from datasketch import MinHashLSH

from common.corpus.dedup import (
    NearDupConfig, _minhash_of, load_signature_cache, save_signature_cache, sha256_of,
)
from common.io import iter_docs

__all__ = ["decontaminate"]


def _default_config() -> NearDupConfig:
    """Same config `common/pipeline.py`'s `run_dedup_pass` builds -- see
    module docstring for why this must not silently drift from that one."""
    return NearDupConfig(max_shingles=3000)


def _load_ids(path: Path | str) -> set[str]:
    with Path(path).open("r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def decontaminate(
    language: str,
    corpus_dir: Path | str,
    splits_dir: Path | str,
    config: NearDupConfig | None = None,
    stats_path: Path | str | None = None,
) -> dict:
    """Index train, query with val+test, drop matched train ids from disk.

    Rewrites `splits_dir/train.ids` in place (minus leaked ids) -- this is
    the file `common/tokenization/tokenizer.py`'s `extract_train_corpus` reads, so the
    fix has to land there, not just in a report.

    Returns counts: train docs indexed, val/test docs queried, train ids
    dropped (with a val vs. test breakdown, since a val-side match is a
    weaker signal than a test-side one but still worth knowing separately).
    """
    config = config or _default_config()
    splits_dir = Path(splits_dir)
    train_ids = _load_ids(splits_dir / "train.ids")
    val_ids = _load_ids(splits_dir / "val.ids")
    test_ids = _load_ids(splits_dir / "test.ids")

    # Signature cache: same mechanism as run_dedup_pass, kept in its own
    # cache file (see module docstring for why it isn't shared with the main
    # dedup pass's cache). A document's MinHash only needs recomputing if
    # this is the first time decontamination has ever indexed it, or if its
    # text changed since the last time -- the content-hash check inside
    # index_pass-equivalent logic below guards that exactly as it does there.
    cache_path = Path(corpus_dir).parent / f"_minhash_cache_{language}_decontam.npz"
    sig_cache = load_signature_cache(cache_path, num_perm=config.num_perm)
    cache_hits = 0
    new_sigs: dict[str, object] = {}
    new_hashes: dict[str, str] = {}

    def _signature(doc) -> object:
        nonlocal cache_hits
        content_hash = sha256_of(doc.text)
        cached = sig_cache.get(doc.id)
        if cached is not None and cached[1] == content_hash:
            cache_hits += 1
            mh = cached[0]
        else:
            mh = _minhash_of(doc.text, config)
        new_sigs[doc.id] = mh
        new_hashes[doc.id] = content_hash
        return mh

    # Single pass over the corpus: every document's MinHash is computed
    # exactly once, regardless of which split it's in. Train documents go
    # straight into the LSH index; val/test signatures are held in memory
    # (small: tens of thousands of MinHash objects, a few hundred bytes each)
    # and queried after the pass closes, instead of re-reading the corpus
    # once per split -- three full decompressing passes over a 600K+ doc
    # corpus would cost as much as the indexing itself, for no benefit.
    t0 = time.time()
    lsh = MinHashLSH(threshold=config.threshold, num_perm=config.num_perm)
    train_sigs: dict[str, object] = {}
    val_sigs: dict[str, object] = {}
    test_sigs: dict[str, object] = {}

    docs_seen = {"train": 0, "val": 0, "test": 0, "other": 0}
    for doc in iter_docs(corpus_dir):
        if doc.id in train_ids:
            docs_seen["train"] += 1
            if doc.id in train_sigs:
                continue  # duplicate id inside train itself; not this pass's job
            mh = _signature(doc)
            train_sigs[doc.id] = mh
            lsh.insert(doc.id, mh)
        elif doc.id in val_ids:
            docs_seen["val"] += 1
            val_sigs[doc.id] = _signature(doc)
        elif doc.id in test_ids:
            docs_seen["test"] += 1
            test_sigs[doc.id] = _signature(doc)
        else:
            docs_seen["other"] += 1

    index_elapsed = time.time() - t0
    t1 = time.time()

    scheme = next(iter(new_sigs.values())).scheme if new_sigs else "affine32"
    save_signature_cache(cache_path, new_sigs, new_hashes, scheme=scheme)

    dropped: set[str] = set()
    dropped_by = {"val": 0, "test": 0}
    matches_found = {"val": 0, "test": 0}

    for split_name, sigs in (("val", val_sigs), ("test", test_sigs)):
        for mh in sigs.values():
            neighbours = lsh.query(mh)
            if neighbours:
                matches_found[split_name] += 1
                for n in neighbours:
                    if n not in dropped:
                        dropped.add(n)
                        dropped_by[split_name] += 1

    query_elapsed = time.time() - t1

    new_train_ids = train_ids - dropped
    (splits_dir / "train.ids").write_text(
        "\n".join(sorted(new_train_ids)) + "\n", encoding="utf-8", newline="\n"
    )

    report = {
        "language": language,
        "docs_seen": docs_seen,
        "cache_hits": cache_hits,
        "train_ids_before": len(train_ids),
        "train_ids_dropped": len(dropped),
        "train_ids_after": len(new_train_ids),
        "val_docs_with_train_match": matches_found["val"],
        "test_docs_with_train_match": matches_found["test"],
        "train_ids_dropped_via_val": dropped_by["val"],
        "train_ids_dropped_via_test": dropped_by["test"],
        "index_elapsed_seconds": round(index_elapsed, 1),
        "query_elapsed_seconds": round(query_elapsed, 1),
    }

    stats_path = Path(stats_path or f"{language}/stats/decontaminate.json")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI. Example: python -m common.corpus.decontaminate --lang hindi"""
    import argparse

    from common.io import enable_utf8_console
    enable_utf8_console()

    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("--lang", required=True, choices=("hindi", "assamese"))
    args = p.parse_args(argv)

    lang = args.lang
    r = decontaminate(lang, f"data/{lang}/clean_deduped", f"{lang}/splits")
    print(f"{lang}: train {r['train_ids_before']:,} -> {r['train_ids_after']:,} "
          f"({r['train_ids_dropped']:,} dropped: {r['train_ids_dropped_via_test']:,} via test, "
          f"{r['train_ids_dropped_via_val']:,} via val)")
    print(f"   index: {r['index_elapsed_seconds']}s, query: {r['query_elapsed_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
