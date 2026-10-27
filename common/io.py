"""Sharded, resumable, UTF-8-safe corpus storage.

Every stage of the pipeline reads shards and writes shards. A shard is a
zstd-compressed JSONL file holding one :class:`Doc` per line. Sessions die --
Kaggle notebooks time out, laptops sleep, scrapers get rate-limited -- so no
stage is allowed to assume it will finish in one run.

Three design decisions carry most of the weight here:

**Bytes, never str.**
    Python on Windows defaults ``open()`` to the system ANSI codepage
    (cp1252), not UTF-8. A single default-encoded file handle silently turns
    Devanagari into mojibake that survives every later stage and is only
    discovered when the tokenizer has already trained on it. This module
    never opens a file in text mode: JSON is serialised straight to UTF-8
    bytes by ``orjson`` and written to a binary handle. Line endings are
    written as explicit ``b"\\n"``, so Windows CRLF translation cannot corrupt
    the byte counts that the dataset statistics report.

**Atomic shard completion.**
    A shard is written to ``<name>.tmp``, renamed into place with
    ``os.replace`` (atomic on Windows and POSIX), and only then marked with a
    ``<name>.done`` sidecar. A shard without its ``.done`` marker is
    unfinished by definition, so a killed run leaves no half-written file that
    a later run would mistake for complete.

**Few large shards.**
    Windows Defender scanning millions of small files dominates runtime, so
    shards default to ~128 MB of uncompressed JSON rather than one file per
    document.
"""

from __future__ import annotations

import io as _io
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator

import orjson
import zstandard as zstd

__all__ = [
    "Doc",
    "StageStats",
    "ShardWriter",
    "read_shard",
    "iter_docs",
    "shard_paths",
    "is_done",
    "map_shards",
    "enable_utf8_console",
]

SHARD_SUFFIX = ".jsonl.zst"
DONE_SUFFIX = ".done"
DEFAULT_TARGET_BYTES = 128 * 1024 * 1024  # uncompressed JSON per shard
DEFAULT_LEVEL = 3  # zstd level; 3 is ~3.5x on Indic text at high speed

_VALID_METHODS = frozenset({"manual", "downloaded"})


# ---------------------------------------------------------------------------
# Console
# ---------------------------------------------------------------------------


def enable_utf8_console() -> None:
    """Make ``print()`` safe for Indic text on a Windows console.

    Shard files are written as bytes and are therefore immune to the platform
    encoding, but ``sys.stdout`` is not: on Windows it defaults to cp1252 and
    raises ``UnicodeEncodeError`` the moment a script prints Devanagari or
    Assamese. That matters because inspecting what a filter removed -- reading
    twenty rejected documents per filter -- is assigned coursework, and those
    scripts print corpus text by definition.

    Call this at the top of any script that prints document text.
    """
    import sys

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Doc:
    """One corpus document.

    Attributes:
        id: Stable unique identifier, e.g. ``hi_wiki_000123``. Used for split
            assignment and for the independence assertion between corpora, so
            it must never be reused across languages.
        text: The document text itself.
        source: Source key matching an entry in ``sources.yaml``.
        collection_method: Either ``"manual"`` or ``"downloaded"``. The
            project requires at least 20% of final training tokens to be
            manually collected, and this field is the only thing that
            distinguishes them -- hence the hard validation below.
        url: Where it came from, where applicable.
        retrieved: ISO date of retrieval, for the provenance record.
        licence: Licence string, for the provenance record.
        meta: Free-form per-source extras (title, section, OCR confidence...).
    """

    id: str
    text: str
    source: str
    collection_method: str
    url: str | None = None
    retrieved: str | None = None
    licence: str | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject anything that would corrupt the manual-fraction accounting."""
        if self.collection_method not in _VALID_METHODS:
            raise ValueError(
                f"collection_method must be one of {sorted(_VALID_METHODS)}, "
                f"got {self.collection_method!r} (doc id {self.id!r})"
            )
        if not self.id:
            raise ValueError("Doc.id must be non-empty")

    def to_bytes(self) -> bytes:
        """Serialise to a UTF-8 JSON line, without the trailing newline.

        ``orjson`` emits UTF-8 directly and does not escape non-ASCII, so
        Indic text stays readable in the shard rather than becoming a wall of
        ``\\u0928`` escapes.
        """
        return orjson.dumps(asdict(self))

    @classmethod
    def from_bytes(cls, raw: bytes) -> "Doc":
        """Parse one UTF-8 JSON line back into a :class:`Doc`."""
        return cls(**orjson.loads(raw))

    def n_words(self) -> int:
        """Whitespace word count -- the collection-time proxy for tokens.

        Real token counts require the final tokenizer, which does not exist
        until the train split is frozen. This proxy steers collection; it must
        not appear in the statistics report.
        """
        return len(self.text.split())


# ---------------------------------------------------------------------------
# Stage statistics
# ---------------------------------------------------------------------------


@dataclass
class StageStats:
    """Counts of what a pipeline stage kept and what it threw away.

    Every filter has to report not just how much it dropped but *why*, because
    the report needs a per-stage token-loss table and because reading a sample
    of what each filter removed is assigned coursework.
    """

    stage: str
    docs_in: int = 0
    docs_out: int = 0
    words_in: int = 0
    words_out: int = 0
    drops: Counter = field(default_factory=Counter)
    corrupted_shards: list = field(default_factory=list)

    def dropped(self, reason: str, doc: Doc | None = None) -> None:
        """Record one dropped document under a named reason."""
        self.drops[reason] += 1

    @property
    def docs_removed(self) -> int:
        return self.docs_in - self.docs_out

    def to_dict(self) -> dict:
        """Flatten for JSON output into ``<language>/stats/``."""
        return {
            "stage": self.stage,
            "docs_in": self.docs_in,
            "docs_out": self.docs_out,
            "docs_removed": self.docs_removed,
            "words_in": self.words_in,
            "words_out": self.words_out,
            "word_retention": (self.words_out / self.words_in) if self.words_in else None,
            "drops": dict(self.drops.most_common()),
            "corrupted_shards": list(self.corrupted_shards),
        }


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


class ShardWriter:
    """Writes documents into rotating, atomically-completed shards.

    Use as a context manager::

        with ShardWriter(Path("data/hi/raw"), prefix="hiwiki") as w:
            for doc in docs:
                w.write(doc)

    Resumes safely: if the output directory already holds completed shards,
    numbering continues after the highest existing index rather than
    overwriting them.
    """

    def __init__(
        self,
        out_dir: Path | str,
        prefix: str = "shard",
        target_bytes: int = DEFAULT_TARGET_BYTES,
        level: int = DEFAULT_LEVEL,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.target_bytes = target_bytes
        self.level = level

        self._index = self._next_index()
        self._fh = None
        self._writer = None
        self._path: Path | None = None
        self._tmp_path: Path | None = None
        self._bytes_in_shard = 0
        self._docs_in_shard = 0
        self._checked_roundtrip = False

        self.docs_written = 0
        self.words_written = 0
        self.shards_written: list[Path] = []

    # -- internals ----------------------------------------------------------

    def _next_index(self) -> int:
        """Continue numbering after any shards already present."""
        existing = sorted(self.out_dir.glob(f"{self.prefix}_*{SHARD_SUFFIX}"))
        if not existing:
            return 0
        last = existing[-1].name[len(self.prefix) + 1 :].split(".")[0]
        return int(last) + 1

    def _open_shard(self) -> None:
        name = f"{self.prefix}_{self._index:04d}{SHARD_SUFFIX}"
        self._path = self.out_dir / name
        self._tmp_path = self.out_dir / (name + ".tmp")
        self._fh = open(self._tmp_path, "wb")
        self._writer = zstd.ZstdCompressor(level=self.level).stream_writer(self._fh)
        self._bytes_in_shard = 0
        self._docs_in_shard = 0

    def _close_shard(self) -> None:
        """Finish the current shard: flush, rename atomically, mark done."""
        if self._writer is None:
            return
        self._writer.flush(zstd.FLUSH_FRAME)
        self._writer.close()
        self._fh.close()

        os.replace(self._tmp_path, self._path)  # atomic
        _write_done(
            self._path,
            {
                "docs": self._docs_in_shard,
                "uncompressed_bytes": self._bytes_in_shard,
                "compressed_bytes": self._path.stat().st_size,
                "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
        )

        self.shards_written.append(self._path)
        self._writer = None
        self._fh = None
        self._index += 1

    def _verify_roundtrip(self, doc: Doc, raw: bytes) -> None:
        """Prove the first document of the run survives serialisation intact.

        This is the guard against the cp1252 failure mode. It runs once per
        writer -- negligible cost, and it fails loudly at the start of a long
        job rather than silently poisoning a corpus that only looks wrong days
        later.
        """
        back = Doc.from_bytes(raw)
        if back.text != doc.text or back.id != doc.id:
            raise RuntimeError(
                "UTF-8 round-trip failed: text changed during serialisation. "
                "Check that PYTHONUTF8=1 and that nothing upstream opened a "
                "file in text mode."
            )
        self._checked_roundtrip = True

    # -- public -------------------------------------------------------------

    def write(self, doc: Doc) -> None:
        """Append one document, rotating to a new shard when the target is hit."""
        if self._writer is None:
            self._open_shard()

        raw = doc.to_bytes()
        if not self._checked_roundtrip:
            self._verify_roundtrip(doc, raw)

        self._writer.write(raw)
        self._writer.write(b"\n")  # explicit: never CRLF, on any platform

        n = len(raw) + 1
        self._bytes_in_shard += n
        self._docs_in_shard += 1
        self.docs_written += 1
        self.words_written += doc.n_words()

        if self._bytes_in_shard >= self.target_bytes:
            self._close_shard()

    def close(self) -> None:
        """Finish the open shard, if any."""
        self._close_shard()

    def __enter__(self) -> "ShardWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
        else:
            # Leave the .tmp behind on failure: no .done marker means the
            # shard is unfinished, and the next run will redo it cleanly.
            if self._writer is not None:
                self._writer.close()
                self._fh.close()


def _write_done(shard: Path, payload: dict) -> None:
    """Write the ``.done`` sidecar that marks a shard as complete."""
    done = shard.with_name(shard.name + DONE_SUFFIX)
    done.write_bytes(orjson.dumps(payload))


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def is_done(shard: Path) -> bool:
    """True if the shard has its completion marker."""
    return shard.with_name(shard.name + DONE_SUFFIX).exists()


def shard_paths(directory: Path | str, prefix: str = "", only_done: bool = True) -> list[Path]:
    """List shards in a directory, sorted, optionally only completed ones."""
    directory = Path(directory)
    if not directory.exists():
        return []
    pattern = f"{prefix}*{SHARD_SUFFIX}" if prefix else f"*{SHARD_SUFFIX}"
    paths = sorted(p for p in directory.glob(pattern) if not p.name.endswith(".tmp"))
    return [p for p in paths if is_done(p)] if only_done else paths


def read_shard(path: Path | str) -> Iterator[Doc]:
    """Stream documents out of one shard.

    Memory footprint is one document, regardless of shard or corpus size --
    which is what lets the cleaning stages run on a 5.9 GB machine.
    """
    path = Path(path)
    dctx = zstd.ZstdDecompressor()
    with open(path, "rb") as fh:
        with dctx.stream_reader(fh) as reader:
            for line in _io.BufferedReader(reader):
                line = line.strip()
                if line:
                    yield Doc.from_bytes(line)


def iter_docs(directory: Path | str, prefix: str = "") -> Iterator[Doc]:
    """Stream every document across every completed shard in a directory."""
    for shard in shard_paths(directory, prefix):
        yield from read_shard(shard)


# ---------------------------------------------------------------------------
# Resumable stage runner
# ---------------------------------------------------------------------------


def map_shards(
    in_dir: Path | str,
    out_dir: Path | str,
    fn: Callable[[Iterator[Doc]], Iterable[Doc]],
    stage: str = "map",
    stats: StageStats | None = None,
    level: int = DEFAULT_LEVEL,
) -> StageStats:
    """Apply a stage function to every input shard, one shard at a time.

    Output shards mirror input shard names one-to-one, which is what makes the
    stage resumable: a shard whose output already carries a ``.done`` marker is
    skipped, so re-running after a session dies costs only the work that had
    not finished.

    Args:
        in_dir: Directory of completed input shards.
        out_dir: Where to write processed shards.
        fn: Pure function over documents. Receives an iterator of input docs
            and yields the docs to keep. Drops are simply not yielded; call
            ``stats.dropped(reason)`` inside ``fn`` to record why.
        stage: Name recorded in the returned statistics.
        stats: Existing accumulator, so a resumed run keeps earlier counts.
        level: zstd compression level.

    Returns:
        :class:`StageStats` covering the shards processed in this call.
    """
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = stats or StageStats(stage=stage)

    for shard in shard_paths(in_dir):
        out_path = out_dir / shard.name
        if is_done(out_path):
            continue  # already processed in an earlier run

        tmp = out_path.with_name(out_path.name + ".tmp")
        cctx = zstd.ZstdCompressor(level=level)
        docs_out = bytes_out = 0

        def _counted(src: Path = shard) -> Iterator[Doc]:
            """Count inputs as they stream past, without buffering them."""
            for doc in read_shard(src):
                stats.docs_in += 1
                stats.words_in += doc.n_words()
                yield doc

        try:
            with open(tmp, "wb") as fh:
                writer = cctx.stream_writer(fh)
                for doc in fn(_counted()):
                    raw = doc.to_bytes()
                    writer.write(raw)
                    writer.write(b"\n")
                    docs_out += 1
                    bytes_out += len(raw) + 1
                    stats.docs_out += 1
                    stats.words_out += doc.n_words()
                writer.flush(zstd.FLUSH_FRAME)
                writer.close()
        except Exception as exc:
            # A shard can be corrupted despite carrying a .done marker: two
            # writers racing on the same shard index during a scraper
            # handoff can interleave their compressed streams into garbage
            # that decompresses to nothing meaningful, or raises partway
            # through (observed: a zstd "Unknown frame descriptor" on one
            # Assamese news shard after the collector was moved to a
            # Scheduled Task mid-run). One bad shard out of tens of thousands
            # should not lose a multi-hour cleaning pass over everything
            # else, so it is quarantined and skipped rather than raised.
            tmp.unlink(missing_ok=True)
            stats.corrupted_shards.append({
                "shard": str(shard),
                "error": f"{type(exc).__name__}: {exc}",
            })
            continue

        os.replace(tmp, out_path)
        _write_done(
            out_path,
            {
                "docs": docs_out,
                "uncompressed_bytes": bytes_out,
                "compressed_bytes": out_path.stat().st_size,
                "stage": stage,
                "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
        )

    return stats


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Run with:  python -m common.io
    #
    # Round-trips real Hindi and Assamese text through the full write/read
    # path. If the encoding setup on this machine is wrong, this fails here
    # rather than three days into a collection run.
    import shutil
    import tempfile

    enable_utf8_console()

    samples = [
        ("hi_test_0001", "भारत एक विशाल देश है। यहाँ २०२४ में लगभग १४० करोड़ लोग रहते हैं।"),
        ("as_test_0001", "অসম ভাৰতৰ উত্তৰ-পূৰ্বাঞ্চলৰ এখন ৰাজ্য। ইয়াত বহুতো মানুহে বাস কৰে।"),
        ("hi_test_0002", "ज़्यादा — विशिष्ट “उद्धरण” और ZWJ: क्‍ष, तथा emoji 🇮🇳"),
    ]

    tmp = Path(tempfile.mkdtemp(prefix="lma_io_check_"))
    try:
        with ShardWriter(tmp, prefix="check", target_bytes=1024) as w:
            for doc_id, text in samples:
                w.write(
                    Doc(
                        id=doc_id,
                        text=text,
                        source="self-check",
                        collection_method="manual",
                    )
                )

        back = list(iter_docs(tmp, prefix="check"))
        assert len(back) == len(samples), f"expected {len(samples)} docs, got {len(back)}"
        for (doc_id, text), doc in zip(samples, back):
            assert doc.id == doc_id, f"id mismatch: {doc.id!r} != {doc_id!r}"
            assert doc.text == text, f"TEXT CORRUPTED for {doc_id}:\n  {doc.text!r}\n  {text!r}"

        # collection_method is load-bearing for the 20% requirement
        try:
            Doc(id="x", text="y", source="s", collection_method="scraped")
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError("invalid collection_method was accepted")

        print(f"OK  {len(back)} docs round-tripped, Devanagari and Assamese intact")
        print(f"OK  {len(shard_paths(tmp, 'check'))} shards written and marked done")
        print("OK  invalid collection_method rejected")
        # Printing corpus text is itself a test: this line raises
        # UnicodeEncodeError on a Windows console without enable_utf8_console().
        print(f"OK  console prints Indic text: {back[1].text[:34]}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
