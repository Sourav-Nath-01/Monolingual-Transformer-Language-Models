"""Chain the cleaning stages into one resumable pass per language.

Order is load-bearing and was fixed twice by measurement, not assumed
(§0.3.7-8 of the plan):

    normalize -> language filter -> boilerplate/quality -> non-language words
                                                          -> dedup (separate driver)

Boilerplate must run before non-language word removal, because datelines and
listing-page furniture were being counted as "foreign words" and dropping 55%
of Hindi news on furniture rather than code-mixing. Language identification
must run before non-language word removal, because stripping Latin words
first would inflate a document's in-script ratio and let English pages pass
the script gate.

Dedup is deliberately not part of this per-shard chain: paragraph and
near-duplicate dedup both need the *whole* corpus present to find repeats
across documents, which a single streaming pass over one shard at a time
cannot do. It runs as a separate, later pass over this stage's output -- see
`run_dedup_pass` below.

Each source (wiki / wikisource / news / books) gets its own thresholds,
because a threshold tuned on Wikipedia does not transfer to news datelines or
OCR noise (plan §0.0 C warning, slide 30). The mapping lives in
`SOURCE_PROFILES` below.

**ID canonicalisation runs first, before normalize.** The news scraper and
the archive.org book collector both derived ``doc.id`` from a per-run counter
that resets to 0 on every restart -- and both collectors were restarted
several times across the session (session teardown, the Task Scheduler
handoff). Measured on the raw output: 1,641 duplicate ids across 19,600
Assamese news documents, 166 across 437 books, some in triplicate. That is
invisible until something relies on id uniqueness -- it surfaced as a crash
in `NearDupDeduper.index_pass` (`ValueError: The given key already exists`
inserting into the LSH index), and it had already silently corrupted the
split manifests built before the crash, since two colliding ids could each
be written into whichever split the hash of their shared id picked.
Recomputing ``id`` from ``doc.url`` here fixes it without touching the raw
shards: a URL is stable across any number of restarts because collection
never re-fetches an already-seen URL, so it is exactly the identity the id
was supposed to carry. Applied uniformly to every source, not just the two
that were broken, since a single canonicalisation rule is easier to reason
about than one that depends on which source is being read.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterator

from common.corpus.dedup import (ExactDeduper, NearDupConfig, NearDupDeduper,
                          ParagraphDeduper)
from common.clean.filters import (AssameseBengaliClassifier, HindiDevanagariClassifier,
                            ParagraphFilter, language_filter_stage,
                            paragraph_filter_stage)
from common.io import Doc, StageStats, enable_utf8_console, iter_docs, map_shards
from common.clean.nonlang import ASSAMESE_CONFIG, HINDI_CONFIG, nonlang_stage
from common.clean.normalize import NormalizeConfig, normalize_stage
from common.clean.quality import (ASSAMESE_QUALITY, HINDI_QUALITY, QualityFilter,
                            quality_stage)

__all__ = ["SourceProfile", "SOURCE_PROFILES", "canonicalize_id",
           "run_language_pipeline", "run_dedup_pass"]


def canonicalize_id(doc: Doc) -> Doc:
    """Recompute ``doc.id`` from ``doc.url`` when present.

    A URL is content identity: the same page always has the same URL, no
    matter which run or which restart of a collector fetched it. Prefixing
    the hash with the id's own first two path segments (e.g. ``as_news``)
    keeps ids readable in logs and stats without reintroducing a counter.
    Documents with no URL (there are none in this corpus, but the check costs
    nothing) keep their existing id unchanged.
    """
    if not doc.url:
        return doc
    prefix = "_".join(doc.id.split("_")[:2]) if "_" in doc.id else doc.id
    new_id = f"{prefix}_{hashlib.sha1(doc.url.encode('utf-8')).hexdigest()[:16]}"
    if new_id == doc.id:
        return doc
    return replace(doc, id=new_id)


@dataclass(frozen=True)
class SourceProfile:
    """Per-source knobs. Wiki text is clean; news and books are not.

    Attributes:
        is_book_like: Routes through the paragraph-level language filter
            (§4.2's book/mixed-language handling) instead of whole-document
            rejection. Only Wikisource is book-like; news and Wikipedia
            articles are short and single-language, so whole-document
            rejection is both correct and far cheaper.
        min_doc_chars: Floor before a source even reaches the language
            filter. News items can be legitimately short; dictionaries are
            long but get caught by the quality filter's line-structure test,
            not by length.
    """

    is_book_like: bool
    min_doc_chars: int


SOURCE_PROFILES: dict[str, SourceProfile] = {
    "wiki": SourceProfile(is_book_like=False, min_doc_chars=200),
    "wikisource": SourceProfile(is_book_like=True, min_doc_chars=200),
    "news": SourceProfile(is_book_like=False, min_doc_chars=100),
    # Archive.org items are book-length and share Wikisource's problem: an
    # English title page or library stamp ahead of a target-language body.
    # Whole-document rejection would cost the entire book over one bad page.
    "books": SourceProfile(is_book_like=True, min_doc_chars=200),
    # Sangraha (downloaded tier). Web-crawled, single-document-per-row, so
    # whole-document rejection is appropriate -- same shape as wiki/news, not
    # book-like. Not assumed clean: a live check found a document under the
    # Assamese split that was not Assamese at all, so this source needs the
    # language filter exactly as much as anything scraped ourselves.
    "downloaded": SourceProfile(is_book_like=False, min_doc_chars=200),
}

_CLASSIFIERS = {
    "hindi": (HindiDevanagariClassifier(), "hindi"),
    "assamese": (AssameseBengaliClassifier(), "assamese"),
}
_NONLANG_CONFIGS = {"hindi": HINDI_CONFIG, "assamese": ASSAMESE_CONFIG}
_QUALITY_CONFIGS = {"hindi": HINDI_QUALITY, "assamese": ASSAMESE_QUALITY}
_SCRIPTS = {"hindi": "devanagari", "assamese": "bengali"}


def run_language_pipeline(
    language: str,
    source: str,
    in_dir: Path | str,
    out_dir: Path | str,
    stats_path: Path | str | None = None,
) -> dict:
    """Run normalize -> language filter -> quality -> non-language words
    over every shard in ``in_dir``, writing cleaned shards to ``out_dir``.

    Resumable: relies on :func:`common.io.map_shards`, which skips any shard
    whose output already carries a ``.done`` marker. Safe to interrupt and
    re-run at any point, including mid-collection -- a source directory that
    gains new shards overnight just gets those processed on the next run.

    Args:
        language: ``"hindi"`` or ``"assamese"``.
        source: Key into :data:`SOURCE_PROFILES` (``wiki``, ``wikisource``,
            ``news``, ``books``).
        in_dir: Raw shard directory, e.g. ``data/hindi/raw/news``.
        out_dir: Cleaned shard output directory.
        stats_path: Where to write the per-stage JSON report. Defaults to
            ``<language>/stats/pipeline_<source>.json``.

    Returns:
        Combined statistics across all four stages.
    """
    profile = SOURCE_PROFILES[source]
    classifier, keep_label = _CLASSIFIERS[language]
    script = _SCRIPTS[language]

    norm_stats = StageStats(stage="normalize")
    lang_stats = StageStats(stage="language_filter")
    qual_stats = StageStats(stage="quality")
    nonlang_stats = StageStats(stage="nonlang")

    qfilter = QualityFilter(_QUALITY_CONFIGS[language])
    nfilter_holder = {"f": None}  # built after normalize so counters are fresh

    def _pipeline(docs: Iterator[Doc]) -> Iterator[Doc]:
        # Stage 0: canonicalise ids from URL, before anything else touches
        # them. See module docstring -- this is the fix for the restart
        # counter collision, applied uniformly regardless of source.
        docs = (canonicalize_id(d) for d in docs)

        # Stage 1: normalise.
        docs = normalize_stage(NormalizeConfig(min_chars=profile.min_doc_chars),
                               stats=norm_stats)(docs)

        # Stage 2: language ID. Books get paragraph-level repair; everything
        # else gets whole-document accept/reject (cheaper, and correct for
        # short single-language documents).
        if profile.is_book_like:
            pfilter = ParagraphFilter(classifier=classifier, keep_label=keep_label,
                                      script=script)
            docs = paragraph_filter_stage(pfilter, stats=lang_stats)(docs)
        else:
            docs = language_filter_stage(classifier, keep_label, stats=lang_stats)(docs)

        # Stage 3: boilerplate + quality heuristics. Must precede stage 4 --
        # see module docstring.
        docs = quality_stage(qfilter, stats=qual_stats)(docs)

        # Stage 4: non-language word cascade. Built lazily so its removal
        # counter starts clean for this run.
        from common.clean.nonlang import NonLanguageWordFilter
        if nfilter_holder["f"] is None:
            nfilter_holder["f"] = NonLanguageWordFilter(_NONLANG_CONFIGS[language])
        docs = nonlang_stage(nfilter_holder["f"], stats=nonlang_stats)(docs)

        yield from docs

    # map_shards owns the ONE stats object that actually sees docs_in/docs_out
    # for the composed pipeline -- it counts input as it reads each shard and
    # output as fn(...) yields. The four per-stage StageStats above only ever
    # get their .drops counter populated by each stage's own dropped() calls;
    # their docs_in/docs_out stay at zero, since nothing else increments them.
    # That split is deliberate: overall throughput belongs to the whole-shard
    # accounting, per-stage attrition belongs to each stage's own drops.
    t0 = time.time()
    overall = map_shards(in_dir, out_dir, _pipeline, stage=f"{language}:{source}")
    elapsed = time.time() - t0

    report = {
        "language": language,
        "source": source,
        "elapsed_seconds": round(elapsed, 1),
        "overall": overall.to_dict(),
        "normalize_drops": dict(norm_stats.drops.most_common()),
        "language_filter_drops": dict(lang_stats.drops.most_common()),
        "quality": {"drops": dict(qual_stats.drops.most_common()), "detail": qfilter.report()},
        "nonlang": {
            "drops": dict(nonlang_stats.drops.most_common()),
            "detail": nfilter_holder["f"].report() if nfilter_holder["f"] else {},
        },
    }
    if overall.corrupted_shards:
        report["corrupted_shards"] = overall.corrupted_shards

    stats_path = Path(stats_path or f"{language}/stats/pipeline_{source}.json")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def run_dedup_pass(
    language: str,
    in_dirs: list[Path | str],
    out_dir: Path | str,
    stats_path: Path | str | None = None,
) -> dict:
    """Exact -> paragraph -> near-duplicate dedup across ALL cleaned sources
    for one language at once.

    Unlike :func:`run_language_pipeline`, this reads every source directory
    together and treats them as one corpus, because duplicates cross source
    boundaries -- the same news story can appear in both the scrape and,
    later, a downloaded corpus.

    **Every pass streams from disk; nothing holds the full corpus in memory.**
    This matters concretely, not just in principle: adding the downloaded
    tier took Hindi's pre-dedup corpus to 752,391 docs / 355M words, about
    4x the size of any earlier dedup run, and scaling the ~1.5GB peak
    measured on a 90M-word run gives ~5.8GB -- against a 5.9GB machine, with
    the OS and this session also wanting RAM. A first version of this
    function built one Python list holding every document's full text at
    once (`all_docs = [d for d in _iter_many(in_dirs)]`), which defeated the
    streaming design that :mod:`common.corpus.dedup`'s own classes already have --
    `ExactDeduper` holds only hashes, `NearDupDeduper` holds only MinHash
    signatures, neither ever needed the text kept around. The fix here is
    orchestration, not new dedup logic: each pass reads its input from a
    shard directory and writes survivors to the next one, so peak memory is
    bounded by hashes and signatures (small, a few hundred MB at this scale)
    plus whatever shard is currently being streamed, never the whole corpus.
    This is the plan's own S0.1 disk-backed fallback, actually built rather
    than left as a described option.

    Args:
        language: ``"hindi"`` or ``"assamese"``.
        in_dirs: Cleaned shard directories to merge and dedup.
        out_dir: Where the deduplicated corpus is written, as fresh shards.
        stats_path: Report location; defaults to
            ``<language>/stats/dedup.json``.
    """
    import shutil

    from common.io import ShardWriter, iter_docs

    # Full recomputation every call, unlike map_shards's incremental resume --
    # ShardWriter's own resumability (continue numbering after existing
    # shards) is the wrong behaviour here, since a second run would append
    # fresh output next to stale output from the first rather than replacing
    # it. Measured effect before this was guarded: 43,344 duplicate ids in
    # the final corpus, half of it an unwanted second copy of every earlier
    # document.
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)

    # Deliberately NOT tempfile.mkdtemp(): that lands under AppData\Local\Temp,
    # outside the data/ tree that has the Windows Defender real-time-scanning
    # exclusion set up for this project (plan S0.2). A smoke test on two small
    # shards took over two minutes there -- Defender scanning every shard
    # write/read is exactly the slowdown the plan already warned about, and
    # writing into an excluded path instead removes it.
    work = out_dir.parent / f"_dedup_work_{language}"
    if work.exists():
        shutil.rmtree(work)
    stage1 = work / "after_exact"
    stage2 = work / "after_paragraph"
    try:
        # Pass 1: exact dedup. ExactDeduper.seen holds one hash per unique
        # document (32 bytes each) -- never the text itself.
        ed = ExactDeduper()
        n_in = words_in = 0
        with ShardWriter(stage1, prefix="s") as w1:
            for doc in _iter_many(in_dirs):
                n_in += 1
                words_in += doc.n_words()
                if not ed.is_duplicate(doc.text):
                    w1.write(doc)

        # Pass 2: paragraph dedup. Two full passes over stage1 by design --
        # a paragraph can only be judged boilerplate after seeing how often
        # it recurs across the whole corpus, which a single streaming pass
        # cannot know in advance. Re-reading from disk between them costs
        # I/O, not RAM.
        pd = ParagraphDeduper(min_doc_count=5, min_paragraph_words=8)
        pd.count_pass(iter_docs(stage1))
        with ShardWriter(stage2, prefix="s") as w2:
            for doc in pd.filter_pass(iter_docs(stage1)):
                w2.write(doc)

        # Pass 3: near-duplicate dedup. index_pass computes one MinHash per
        # document and keeps only the signature (~800 bytes); the source
        # text is never retained once its signature exists. filter_pass
        # re-streams stage2 a second time rather than trusting anything held
        # from the index pass, so nothing here scales with document length,
        # only with document count.
        #
        # Signature cache: this function recomputes near-dup indexing over
        # the WHOLE corpus every call, by design (see the docstring above) --
        # but a document whose text hasn't changed since the last round
        # doesn't need its MinHash recomputed, only reused. Every source
        # that was already part of a previous round pays the shingling cost
        # exactly once, ever; only genuinely new or changed documents pay it
        # again. See common/corpus/dedup.py's cache functions for the correctness
        # guard (content-hash check) this relies on.
        from common.corpus.dedup import load_signature_cache, save_signature_cache
        ndup_config = NearDupConfig(max_shingles=3000)
        cache_path = out_dir.parent / f"_minhash_cache_{language}.npz"
        sig_cache = load_signature_cache(cache_path, num_perm=ndup_config.num_perm)

        nd = NearDupDeduper(ndup_config)
        nd.index_pass(iter_docs(stage2), cache=sig_cache)
        nd.resolve_clusters()

        new_sigs, new_content_hashes = nd.signatures_for_cache()
        scheme = next(iter(new_sigs.values())).scheme if new_sigs else "affine32"
        save_signature_cache(cache_path, new_sigs, new_content_hashes, scheme=scheme)

        docs_out = words_out = 0
        with ShardWriter(out_dir, prefix=f"{language}_clean") as writer:
            for doc in nd.filter_pass(iter_docs(stage2)):
                writer.write(doc)
                docs_out += 1
                words_out += doc.n_words()
    finally:
        shutil.rmtree(work, ignore_errors=True)

    report = {
        "language": language,
        "docs_in": n_in,
        "words_in": words_in,
        "exact_duplicates_removed": ed.duplicates,
        "paragraph_dedup": pd.report(),
        "near_duplicate": nd.report(),
        "docs_out": docs_out,
        "words_out": words_out,
        "word_retention": round(words_out / words_in, 4) if words_in else None,
    }
    stats_path = Path(stats_path or f"{language}/stats/dedup.json")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def _iter_many(dirs: list[Path | str]) -> Iterator[Doc]:
    for d in dirs:
        yield from iter_docs(d)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run the cleaning pipeline for one language across all its sources.

    Example::

        python -m common.pipeline --lang hindi
        python -m common.pipeline --lang assamese --dedup
    """
    import argparse

    enable_utf8_console()
    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("--lang", required=True, choices=("hindi", "assamese"))
    p.add_argument("--dedup", action="store_true",
                   help="also run the corpus-wide dedup pass after cleaning")
    p.add_argument("--sources", default="",
                   help="comma-separated subset of wiki,wikisource,news,books "
                        "(default: all). Useful for re-running just the sources "
                        "affected by a fix without redoing an expensive full pass.")
    args = p.parse_args(argv)

    lang = args.lang
    wanted = set(args.sources.split(",")) if args.sources else set(SOURCE_PROFILES)
    cleaned_dirs = []
    for source in SOURCE_PROFILES:
        if source not in wanted:
            out = Path(f"data/{lang}/clean/{source}")
            if out.exists() and any(out.glob("*.jsonl.zst")):
                cleaned_dirs.append(out)  # already-clean, just not reprocessing it
            continue
        raw = Path(f"data/{lang}/raw/{source}")
        if not raw.exists() or not any(raw.glob("*.jsonl.zst")):
            print(f"skip {lang}/{source}: no raw shards yet")
            continue
        out = Path(f"data/{lang}/clean/{source}")
        print(f"== {lang}/{source} -> {out}")
        report = run_language_pipeline(lang, source, raw, out)
        ov = report["overall"]
        print(f"   {ov['docs_in']:,} -> {ov['docs_out']:,} docs, "
              f"{ov['words_in']:,} -> {ov['words_out']:,} words  ({report['elapsed_seconds']}s)")
        print(f"   normalize drops:       {report['normalize_drops']}")
        print(f"   language_filter drops: {report['language_filter_drops']}")
        print(f"   quality drops:         {report['quality']['drops']}")
        print(f"   nonlang words removed: "
              f"{report['nonlang']['detail'].get('total_tokens_removed', 0):,}")
        if report.get("corrupted_shards"):
            print(f"   ⚠ {len(report['corrupted_shards'])} corrupted shard(s) skipped: "
                  f"{[c['shard'] for c in report['corrupted_shards']]}")
        cleaned_dirs.append(out)

    if args.dedup and cleaned_dirs:
        print(f"\n== dedup across {len(cleaned_dirs)} cleaned sources")
        dreport = run_dedup_pass(lang, cleaned_dirs, f"data/{lang}/clean_deduped")
        print(f"   {dreport['docs_in']:,} -> {dreport['docs_out']:,} docs")
        print(f"   {dreport['words_in']:,} -> {dreport['words_out']:,} words "
              f"({100*(1-dreport['word_retention']):.1f}% removed)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
