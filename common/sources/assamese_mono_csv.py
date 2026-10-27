"""Stream the MWirelabs/jintz0 "assamese-monolingual-corpus" (identical
mirrors on HuggingFace) as a sixth downloaded-tier source for Assamese.

**Why this source.** Same motivation as `common/sources/xahitya.py`: every prior
downloaded source is exhausted or fully drawn, and Assamese's manual ceiling
is reached. This is a single CSV (`assamese_monolingual_sentences_final_
cleaned.csv`, one `text` column, ~655MB) of pre-segmented sentences --
verified by sampling real content before building anything (news/current-
affairs prose, not degenerate or templated text).

**Document granularity.** Rows are single sentences, the same wrinkle as
`common/sources/indiccorp_v2.py`'s flat source -- too short individually to clear
`min_doc_chars=200` after normal punctuation/whitespace overhead. Consecutive
rows are batched into pseudo-documents the same way, up to a target
character count.

**CSV parsing over a live stream, not line-splitting.** A "row" is not
always one line -- a text field containing an embedded newline inside CSV
quoting would break naive line-based splitting. `csv.reader` is fed from a
generator that decodes and yields lines from the HTTP stream, so embedded
newlines inside quoted fields are handled correctly rather than assumed away.
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from common.io import Doc, ShardWriter, enable_utf8_console

__all__ = ["AssameseMonoCsvConfig", "stream_assamese_mono_csv"]

_URL = ("https://huggingface.co/datasets/MWirelabs/assamese-monolingual-corpus/"
        "resolve/main/assamese_monolingual_sentences_final_cleaned.csv")
_TARGET_CHARS = 2000


@dataclass
class AssameseMonoCsvConfig:
    word_cap: int
    min_chars: int = 200
    target_chars: int = _TARGET_CHARS


def _iter_csv_lines(url: str, chunk_size: int = 1 << 20):
    """Decode an HTTP byte stream into text lines, for feeding csv.reader."""
    buf = b""
    with httpx.stream("GET", url, follow_redirects=True, timeout=60.0) as resp:
        resp.raise_for_status()
        for chunk in resp.iter_bytes(chunk_size):
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                yield line.decode("utf-8", errors="ignore")
        if buf.strip():
            yield buf.decode("utf-8", errors="ignore")


def stream_assamese_mono_csv(
    language: str,
    out_dir: Path | str,
    config: AssameseMonoCsvConfig,
    stats_path: Path | str | None = None,
) -> dict:
    """Stream the monolingual-sentences CSV, batching rows into pseudo-documents."""
    import shutil

    assert language == "assamese", "this collector is only wired for Assamese"

    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = {"words": 0, "docs": 0, "rows_seen": 0}
    t0 = time.time()

    reader = csv.reader(_iter_csv_lines(_URL))
    header = next(reader, None)  # "text"

    def flush(writer: ShardWriter, buf_lines: list[str]) -> int:
        text = "\n".join(buf_lines).strip()
        if len(text) < config.min_chars:
            return 0
        doc_id = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
        doc = Doc(
            id=f"as_monocsv_{doc_id}",
            text=text,
            source="assamese-monolingual-csv",
            collection_method="downloaded",
            url=None,
            retrieved=time.strftime("%Y-%m-%d"),
            licence="see MWirelabs/assamese-monolingual-corpus on HuggingFace",
            meta={},
        )
        writer.write(doc)
        n = doc.n_words()
        counts["words"] += n
        counts["docs"] += 1
        return n

    with ShardWriter(out_dir, prefix="as_monocsv") as writer:
        pending: list[str] = []
        pending_chars = 0
        for row in reader:
            counts["rows_seen"] += 1
            if counts["words"] >= config.word_cap:
                break
            if not row:
                continue
            line = row[0].strip()
            if not line:
                continue
            pending.append(line)
            pending_chars += len(line)
            if pending_chars >= config.target_chars:
                flush(writer, pending)
                pending, pending_chars = [], 0

        if pending and counts["words"] < config.word_cap:
            flush(writer, pending)

    counts["elapsed_seconds"] = round(time.time() - t0, 1)
    counts["word_cap"] = config.word_cap

    stats_path = Path(stats_path or f"{language}/stats/collect_monocsv.json")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(counts, indent=2, ensure_ascii=False), encoding="utf-8")
    return counts


def main(argv: list[str] | None = None) -> int:
    """CLI. Example: python -m common.sources.assamese_mono_csv --word-cap 999000000"""
    import argparse

    enable_utf8_console()
    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("--word-cap", type=int, required=True)
    p.add_argument("--min-chars", type=int, default=200)
    args = p.parse_args(argv)

    cfg = AssameseMonoCsvConfig(word_cap=args.word_cap, min_chars=args.min_chars)
    result = stream_assamese_mono_csv("assamese", "data/assamese/raw/monocsv", cfg)
    print(f"assamese/monocsv: {result['docs']:,} docs, {result['words']:,} words "
          f"in {result['elapsed_seconds']}s (rows_seen={result['rows_seen']:,})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
