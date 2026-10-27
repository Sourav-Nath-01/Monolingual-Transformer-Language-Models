"""Deduplication: exact, then paragraph, then near-duplicate.

Three passes, cheapest first, each catching what the previous one cannot
(Lecture 3 slides 13-14, and slide 35 calls this the single highest-return
cleaning step).

**Exact.** SHA-256 of whitespace-normalised text. Catches identical documents
collected twice -- the same Wikipedia article pulled by both the dump and a
stray scrape, or a news item re-published verbatim on a mirror.

**Paragraph.** Exact match on individual paragraphs across the whole corpus.
Catches boilerplate that exact-document hashing cannot, because it differs by
document even though a chunk of it repeats: a news site's cookie notice, a
Wikisource page's running header, a book's "all rights reserved" block copied
onto every scanned page.

**Near-duplicate.** MinHash + LSH over 3-word shingles, matching the lecture's
own parameters (k=100 hashes, b=20 bands, r=5 rows, threshold ~=0.55). Catches
the case exact hashing structurally cannot: two documents that are almost the
same text -- a wire-service story picked up with a different headline, an
article republished with one paragraph edited.

RAM is the constraint driving the interface here (project plan S0.1): 10M
documents x 100 hashes x 4 bytes is about 4 GB of signatures before any bucket
index, on a machine with 5.9 GB total. Everything below streams and never
holds the full corpus in memory at once -- exact and paragraph dedup use
one-pass streaming hash tables, and near-duplicate dedup keeps only
signatures and bucket keys, never document text, until the final pass writes
survivors.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Iterator

from datasketch import MinHash, MinHashLSH

from common.io import Doc, StageStats

__all__ = [
    "normalize_for_hash", "sha256_of", "ExactDeduper", "ParagraphDeduper",
    "NearDupConfig", "NearDupDeduper", "shingles",
    "save_signature_cache", "load_signature_cache",
]


def normalize_for_hash(text: str) -> str:
    """Collapse whitespace before hashing, so formatting alone never causes
    a false unique. This is presentation normalisation for hashing only --
    it does not touch the stored document text."""
    return " ".join(text.split())


def sha256_of(text: str) -> str:
    return hashlib.sha256(normalize_for_hash(text).encode("utf-8")).hexdigest()


# --- exact -------------------------------------------------------------


@dataclass
class ExactDeduper:
    """Drop documents whose full text has been seen before.

    Holds one hash per unique document (32 bytes each), not the text, so a
    10M-document corpus costs ~320 MB -- comfortable even on this machine.
    """

    seen: set[str] = field(default_factory=set)
    duplicates: int = 0

    def is_duplicate(self, text: str) -> bool:
        h = sha256_of(text)
        if h in self.seen:
            self.duplicates += 1
            return True
        self.seen.add(h)
        return False

    def stage(self, stats: StageStats | None = None):
        def _fn(docs: Iterator[Doc]) -> Iterator[Doc]:
            for doc in docs:
                if self.is_duplicate(doc.text):
                    if stats is not None:
                        stats.dropped("dedup:exact")
                    continue
                yield doc

        return _fn


# --- paragraph -----------------------------------------------------------


@dataclass
class ParagraphDeduper:
    """Drop paragraphs that repeat across many documents (boilerplate).

    Two-pass by construction: a paragraph can only be judged boilerplate
    after seeing how often it recurs across the *whole* corpus, which is not
    known on a single streaming pass. Pass 1 counts; pass 2 filters using
    those counts. This is why paragraph dedup needs its own driver rather
    than fitting the single-pass `map_shards` stage interface directly.

    Args:
        min_doc_count: A paragraph seen in at least this many distinct
            documents is boilerplate and is dropped everywhere it occurs.
        min_paragraph_words: Paragraphs shorter than this are exempted --
            a two-word heading recurring across a book's chapters is
            structure, not boilerplate, and dropping it would fragment
            otherwise-good prose.
    """

    min_doc_count: int = 5
    min_paragraph_words: int = 8
    _counts: Counter = field(default_factory=Counter)
    paragraphs_dropped: int = 0
    docs_emptied: int = 0

    def count_pass(self, docs: Iterable[Doc]) -> None:
        """Pass 1: tally which paragraph hashes recur across documents."""
        for doc in docs:
            seen_in_doc: set[str] = set()
            for para in doc.text.split("\n\n"):
                if len(para.split()) < self.min_paragraph_words:
                    continue
                h = sha256_of(para)
                if h not in seen_in_doc:  # count each doc once per paragraph
                    self._counts[h] += 1
                    seen_in_doc.add(h)

    def filter_pass(self, docs: Iterator[Doc]) -> Iterator[Doc]:
        """Pass 2: drop paragraphs whose hash cleared the threshold.

        Yields new Doc instances rather than mutating the input. Two-pass
        dedup means the same objects can plausibly be reused (a second
        deduper run over the same in-memory batch, a caller holding onto the
        originals for a report) and a stage that mutates its input silently
        breaks the second use. common/io.py documents every stage as a pure
        function for exactly this reason.
        """
        boilerplate = {h for h, c in self._counts.items() if c >= self.min_doc_count}
        for doc in docs:
            kept = []
            for para in doc.text.split("\n\n"):
                if (len(para.split()) >= self.min_paragraph_words
                        and sha256_of(para) in boilerplate):
                    self.paragraphs_dropped += 1
                    continue
                kept.append(para)
            new_text = "\n\n".join(kept).strip()
            if not new_text:
                self.docs_emptied += 1
                continue
            yield replace(doc, text=new_text)

    def report(self) -> dict:
        return {
            "distinct_paragraphs_seen": len(self._counts),
            "boilerplate_paragraphs": sum(
                1 for c in self._counts.values() if c >= self.min_doc_count
            ),
            "paragraphs_dropped": self.paragraphs_dropped,
            "documents_emptied": self.docs_emptied,
        }


# --- near-duplicate --------------------------------------------------------


def shingles(text: str, k: int = 3, max_shingles: int | None = None) -> set[str]:
    """k-word shingles of whitespace-tokenised text.

    Args:
        max_shingles: Cap on shingles produced, via a fixed stride over
            positions rather than every position. Without this, a Wikisource
            book (measured average: 45,000 words) produces up to 45,000
            shingles, and MinHash.update() is called once per shingle --
            indexing 100 such documents measured at 30 seconds, which would
            put a full-corpus pass at several hours, competing with the
            scrapers running alongside it. A stride sample keeps the
            approximate Jaccard signature representative (MinHash is already
            an approximation) while making cost independent of document
            length. Short documents -- the news articles and encyclopedia
            entries that are the actual near-duplicate risk -- are unaffected,
            since they rarely approach the cap.
    """
    words = normalize_for_hash(text).split()
    if len(words) < k:
        return {" ".join(words)} if words else set()
    n_positions = len(words) - k + 1
    if max_shingles and n_positions > max_shingles:
        stride = max(n_positions // max_shingles, 1)
        positions = range(0, n_positions, stride)
    else:
        positions = range(n_positions)
    return {" ".join(words[i:i + k]) for i in positions}


@dataclass(frozen=True)
class NearDupConfig:
    """MinHash/LSH parameters, matching the lecture's own numbers.

    Attributes:
        shingle_size: Words per shingle.
        num_perm: Hash functions per document (k=100 in the lecture).
        bands: LSH bands (b=20).
        threshold: Target Jaccard similarity (~0.55, from b=20,r=5 via
            (1/b)^(1/r)).
        max_shingles: Per-document shingle cap; see :func:`shingles`.
    """

    shingle_size: int = 3
    num_perm: int = 100
    bands: int = 20
    threshold: float = 0.55
    max_shingles: int = 20_000

    def __post_init__(self):
        rows = self.num_perm // self.bands
        assert self.bands * rows == self.num_perm, "bands must divide num_perm"


def _minhash_of(text: str, cfg: NearDupConfig) -> MinHash:
    mh = MinHash(num_perm=cfg.num_perm)
    for sh in shingles(text, cfg.shingle_size, cfg.max_shingles):
        mh.update(sh.encode("utf-8"))
    return mh


# --- signature cache ---------------------------------------------------
#
# Every time a new source is merged in, `run_dedup_pass` recomputes near-dup
# indexing over the WHOLE corpus from scratch -- deliberately (S0.3.10c: a
# resumable ShardWriter appending fresh output next to stale output produced
# 43,344 duplicate ids, so full recomputation was chosen over incremental
# output on purpose). But "recompute the output" and "recompute every
# document's MinHash signature" are different costs, and only the first one
# actually needs to happen every time. Shingling+hashing a document's text is
# the expensive part of indexing (cost scales with document length); a
# signature that has already been computed once can be reused unchanged, as
# long as the document's text and the NearDupConfig are both unchanged --
# reconstructing a MinHash from saved hash values is a fixed-size operation
# independent of document length, measured at several times faster than
# recomputing even a short document's shingles from scratch, and the gap
# widens with real document lengths. Documents whose text never changes
# between runs (every existing source, on every round that only *adds* a new
# source) pay this cost exactly once, ever, instead of once per round.


def save_signature_cache(
    path: Path, sigs: dict[str, MinHash], content_hashes: dict[str, str], scheme: str
) -> None:
    """Persist a signature dict to disk as one compressed archive.

    `content_hashes` (sha256 of each document's exact text, same function
    `ExactDeduper` already uses) is stored alongside the signatures so a
    cache hit can be verified, not just trusted. This matters because a
    document's *text* is not actually guaranteed constant across dedup
    rounds even when its id is: paragraph-dedup's boilerplate detector is
    corpus-wide, so a paragraph that recurred in too few documents to count
    as boilerplate in round 1 can cross that threshold once a new source
    adds more copies of it in round 2 -- the same document id would then
    carry different text. Using a stale signature for changed text would be
    silently wrong rather than loudly wrong, which is worse.

    `scheme` is saved for the same reason: `MinHash` reconstruction requires
    the exact permutation scheme it was created with (`datasketch` refuses
    to compute Jaccard across two different schemes), so storing it prevents
    a silent mismatch if datasketch's default scheme ever changes between
    dependency upgrades.
    """
    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = np.array(list(sigs.keys()))
    hashvalues = np.stack([sigs[i].hashvalues for i in sigs]) if sigs else np.empty((0, 0))
    hashes = np.array([content_hashes[i] for i in sigs])
    tmp = path.with_suffix(path.suffix + ".tmp")
    # np.savez_compressed silently appends ".npz" to any path that doesn't
    # already end with it -- passed a bare "*.tmp" path, it would actually
    # write "*.tmp.npz" and leave the atomic rename below moving a file that
    # was never created. Passing an open file handle instead of a path
    # string is documented to skip that auto-append, so the tmp file lands
    # exactly where the rename expects it.
    with tmp.open("wb") as f:
        np.savez_compressed(f, ids=ids, hashvalues=hashvalues, content_hashes=hashes,
                             scheme=np.array(scheme))
    tmp.replace(path)


def load_signature_cache(path: Path, num_perm: int) -> dict[str, tuple[MinHash, str]]:
    """Load a previously-saved signature cache, or return {} if none exists
    yet (first run for this language -- there is nothing to reuse).

    Returns ``{doc_id: (minhash, content_hash)}`` -- the content hash travels
    with the signature so callers can verify a hit before trusting it
    (see `save_signature_cache`).
    """
    import numpy as np

    path = Path(path)
    if not path.exists():
        return {}
    data = np.load(path, allow_pickle=False)
    ids, hashvalues = data["ids"], data["hashvalues"]
    content_hashes, scheme = data["content_hashes"], str(data["scheme"])
    if hashvalues.shape[0] == 0:
        return {}
    return {
        doc_id: (MinHash(num_perm=num_perm, hashvalues=hv, scheme=scheme), str(ch))
        for doc_id, hv, ch in zip(ids, hashvalues, content_hashes)
    }


@dataclass
class NearDupDeduper:
    """Two-pass near-duplicate removal via MinHash + LSH.

    Pass 1 builds signatures and an LSH index over document ids; pass 2 walks
    documents again and keeps the first instance of each duplicate cluster
    (the smallest doc id, so the choice is deterministic across re-runs
    rather than depending on iteration order).

    Memory: one MinHash (num_perm x 4 bytes) plus one small doc-id string per
    document -- roughly 400 bytes/doc, so 10M docs is ~4 GB. This is the
    ceiling the project plan (S0.1) sizes Kaggle's ~30 GB RAM against; the
    disk-backed fallback there applies if this ever runs short locally.
    """

    config: NearDupConfig = field(default_factory=NearDupConfig)
    _lsh: MinHashLSH = field(init=False)
    _sigs: dict = field(default_factory=dict)
    _content_hashes: dict[str, str] = field(default_factory=dict)
    _drop: set[str] | None = None
    clusters_found: int = 0
    docs_dropped: int = 0
    duplicate_ids_skipped: int = 0
    cache_hits: int = 0

    def __post_init__(self):
        self._lsh = MinHashLSH(
            threshold=self.config.threshold, num_perm=self.config.num_perm
        )

    def index_pass(
        self, docs: Iterable[Doc], cache: dict[str, tuple[MinHash, str]] | None = None
    ) -> None:
        """Pass 1: hash every document and insert into the LSH index.

        A document whose id was already seen is skipped here rather than
        inserted a second time. `MinHashLSH.insert` raises on a duplicate key,
        and document ids are not guaranteed unique from every possible
        source: measured on Sangraha's own "verified" Hindi split, 201 of
        635,382 rows shared a `doc_id` with another row in the same split --
        not a collection bug, just the upstream corpus having a small amount
        of its own duplication. The two occurrences are, definitionally, the
        same identity; keeping the first and skipping the rest is the same
        choice `resolve_clusters` already makes for near-duplicate clusters,
        just resolved at insertion time instead of after the fact.

        Args:
            cache: Previously-computed ``{doc_id: (minhash, content_hash)}``
                (see `load_signature_cache`). A hit skips the expensive
                shingle+hash computation entirely and reuses the saved
                MinHash -- but only once the cached content hash is checked
                against this document's actual current text, since the same
                id is not guaranteed to carry the same text across dedup
                rounds (see `save_signature_cache`'s docstring). A mismatch
                falls back to a fresh computation exactly as if there had
                been no cache entry at all.
        """
        cache = cache or {}
        for doc in docs:
            if doc.id in self._sigs:
                self.duplicate_ids_skipped += 1
                continue
            content_hash = sha256_of(doc.text)
            cached = cache.get(doc.id)
            if cached is not None and cached[1] == content_hash:
                mh = cached[0]
                self.cache_hits += 1
            else:
                mh = _minhash_of(doc.text, self.config)
            self._sigs[doc.id] = mh
            self._content_hashes[doc.id] = content_hash
            self._lsh.insert(doc.id, mh)

    def resolve_clusters(self) -> None:
        """Decide, for every document, whether it is the cluster survivor.

        Must run after `index_pass` finishes and before `filter_pass` starts:
        querying the LSH index while still inserting would let insertion
        order affect which id is kept, which would make survivor choice
        depend on shard processing order rather than doc id.
        """
        resolved: set[str] = set()
        drop: set[str] = set()
        for doc_id in self._sigs:
            if doc_id in resolved:
                continue
            neighbours = set(self._lsh.query(self._sigs[doc_id]))
            neighbours.add(doc_id)
            if len(neighbours) > 1:
                self.clusters_found += 1
                survivor = min(neighbours)
                drop |= (neighbours - {survivor})
            resolved |= neighbours
        self._drop = drop
        self.docs_dropped = len(drop)

    def filter_pass(self, docs: Iterator[Doc]) -> Iterator[Doc]:
        """Pass 2: yield every document except cluster losers.

        Also collapses repeat occurrences of the same id to one: index_pass
        skips (rather than crashes on) a document whose id repeats, but
        skipping it there only stops it entering the LSH index -- the
        physical repeat is still sitting in the shard stream. Without
        tracking what has already been yielded here, that second occurrence
        would sail through (its id is not in `_drop`, since it was never
        clustered) and get written to output, silently reproducing the exact
        duplicate-id problem this whole class exists to remove.
        """
        if self._drop is None:
            raise RuntimeError("call resolve_clusters() before filter_pass()")
        yielded: set[str] = set()
        for doc in docs:
            if doc.id in self._drop or doc.id in yielded:
                continue
            yielded.add(doc.id)
            yield doc

    def report(self) -> dict:
        return {
            "documents_hashed": len(self._sigs),
            "duplicate_ids_skipped": self.duplicate_ids_skipped,
            "clusters_found": self.clusters_found,
            "documents_dropped": self.docs_dropped,
            "cache_hits": self.cache_hits,
        }

    def signatures_for_cache(self) -> tuple[dict[str, MinHash], dict[str, str]]:
        """Everything `save_signature_cache` needs: this run's signatures and
        their content hashes, ready to persist for the next round."""
        return self._sigs, self._content_hashes


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Run with:  python -m common.corpus.dedup
    from common.io import Doc, enable_utf8_console

    enable_utf8_console()
    failures = []

    def check(name, got, want):
        if got == want:
            print(f"OK   {name}  -> {got}")
        else:
            failures.append(name)
            print(f"FAIL {name}\n       got  {got!r}\n       want {want!r}")

    # --- exact ---
    ed = ExactDeduper()
    check("first copy kept", ed.is_duplicate("भारत एक देश है।"), False)
    check("exact repeat dropped", ed.is_duplicate("भारत एक देश है।"), True)
    check("whitespace-only diff still dropped",
          ed.is_duplicate("भारत   एक देश है।  "), True)
    check("different text kept", ed.is_duplicate("यह अलग वाक्य है।"), False)

    # --- paragraph ---
    boiler = "यह वेबसाइट कुकीज़ का उपयोग करती है और नीति स्वीकार करने का अनुरोध करती है।"
    docs = [
        Doc(id=f"d{i}", text=f"असली सामग्री पैराग्राफ संख्या {i} है यह अनोखा है।\n\n{boiler}",
            source="t", collection_method="manual")
        for i in range(6)
    ]
    pd = ParagraphDeduper(min_doc_count=5, min_paragraph_words=5)
    pd.count_pass(docs)
    out = list(pd.filter_pass(docs))
    check("boilerplate paragraph stripped from all docs",
          all(boiler not in d.text for d in out), True)
    check("real content survives",
          all("असली सामग्री" in d.text for d in out), True)
    check("no document emptied", pd.docs_emptied, 0)
    check("filter_pass does not mutate its input",
          any(boiler in d.text for d in docs), True)

    # A doc that is ENTIRELY boilerplate should be dropped whole. Fresh Doc
    # objects here on purpose -- `docs` above must stay untouched by
    # filter_pass for this scenario to test what it claims to.
    docs2 = [
        Doc(id=f"d{i}", text=f"असली सामग्री पैराग्राफ संख्या {i} है यह अनोखा है।\n\n{boiler}",
            source="t", collection_method="manual")
        for i in range(6)
    ]
    pure = [Doc(id="pure", text=boiler, source="t", collection_method="manual")] + docs2
    pd2 = ParagraphDeduper(min_doc_count=5, min_paragraph_words=5)
    pd2.count_pass(pure)
    out2 = list(pd2.filter_pass(pure))
    check("pure-boilerplate doc dropped entirely",
          any(d.id == "pure" for d in out2), False)

    # --- near-duplicate ---
    base = "भारत एक विशाल और विविधतापूर्ण देश है जहाँ अनेक भाषाएँ बोली जाती हैं और संस्कृति समृद्ध है।"
    near = "भारत एक विशाल और विविधतापूर्ण देश है जहाँ अनेक भाषाएँ बोली जाती हैं और संस्कृति बहुत समृद्ध है।"
    far = "मुंबई भारत का एक प्रमुख वित्तीय और व्यावसायिक केंद्र माना जाता है।"
    ndocs = [
        Doc(id="a1", text=base, source="t", collection_method="manual"),
        Doc(id="a2", text=near, source="t", collection_method="manual"),
        Doc(id="b1", text=far, source="t", collection_method="manual"),
    ]
    nd = NearDupDeduper(NearDupConfig(shingle_size=2, num_perm=64, bands=16, threshold=0.5))
    nd.index_pass(ndocs)
    nd.resolve_clusters()
    kept = {d.id for d in nd.filter_pass(ndocs)}
    check("near-duplicate pair collapses to one survivor", len(kept & {"a1", "a2"}), 1)
    check("unrelated document survives", "b1" in kept, True)
    check("survivor is deterministic (lowest id)", "a1" in kept, True)

    # Shingle count sanity.
    check("shingle count for short text", len(shingles("एक दो तीन चार", k=3)), 2)
    check("shingle fallback for text shorter than k", len(shingles("एक दो", k=3)), 1)

    # Duplicate ids in the input must not crash MinHashLSH.insert -- measured
    # on real Sangraha data: 201 of 635,382 Hindi rows repeated a doc_id.
    dup_id_docs = [
        Doc(id="same", text=base, source="t", collection_method="downloaded"),
        Doc(id="same", text="entirely unrelated text about something else here",
            source="t", collection_method="downloaded"),
        Doc(id="other", text=far, source="t", collection_method="downloaded"),
    ]
    nd2 = NearDupDeduper(NearDupConfig(shingle_size=2, num_perm=64, bands=16, threshold=0.5))
    nd2.index_pass(dup_id_docs)  # must not raise
    check("duplicate id skipped at index time, not crashed",
          nd2.duplicate_ids_skipped, 1)
    nd2.resolve_clusters()
    out = list(nd2.filter_pass(dup_id_docs))
    out_ids = [d.id for d in out]
    check("duplicate id appears exactly once in output",
          out_ids.count("same"), 1)
    check("non-duplicate id still present", "other" in out_ids, True)

    if failures:
        raise SystemExit(f"\n{len(failures)} check(s) failed: {failures}")
    print("\nOK  all dedup checks passed")
