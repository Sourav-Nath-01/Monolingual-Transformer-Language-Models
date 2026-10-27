"""Stream IndicCorpV2 (ai4bharat) as a third downloaded-tier source for Assamese.

**Why this source.** Same motivation as `common/sources/madlad.py`: Sangraha's
non-synthetic Assamese supply is naturally exhausted (157.46M raw words), so
reaching a materially larger total needs a different corpus. IndicCorpV2 is
AI4Bharat's own successor to Sangraha's underlying data -- same lab, same
general provenance standard -- and ships a flat `data/as.txt`, 1.18GB.

**Document granularity is the one real wrinkle here.** Unlike Sangraha and
MADLAD-400, `as.txt` is one *sentence* per line, not one document per line --
measured directly: an 8000-line sample sits at roughly 368 bytes/line, which
in this script (measured elsewhere in this project at ~17.7 bytes/word) is
on the order of 20 words -- far under the pipeline's `min_doc_chars=200` for
the downloaded profile (`common/pipeline.py`). Treating each line as its own
`Doc` would have nearly all of it rejected by the length filter before
language ID or quality even ran.

The fix: accumulate consecutive lines into a pseudo-document until it clears
a target size, then flush it as one `Doc`. This is not lossy in any way that
matters for a monolingual LM corpus -- the model consumes token sequences,
and grouping originally-adjacent sentences back into paragraph-sized chunks
is closer to natural document structure than one-sentence-per-doc would have
been, not further from it.

**Streamed, not downloaded whole.** Same reasoning as `common/sources/madlad.py`:
1.18GB uncompressed is diskable but not worth materialising up front when an
HTTP chunk stream can be buffered and split on newlines directly.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from common.io import Doc, ShardWriter, enable_utf8_console

__all__ = ["IndicCorpConfig", "stream_indiccorp_v2"]

_URL = "https://huggingface.co/datasets/ai4bharat/IndicCorpV2/resolve/main/data/{lang}.txt"

# Target pseudo-document size. Large enough to comfortably clear
# min_doc_chars=200 with margin, small enough that a handful of sentences
# from the same crawled source don't dominate paragraph-dedup's boilerplate
# detector (which looks for repetition across many documents, not size).
_TARGET_CHARS = 2000


@dataclass
class IndicCorpConfig:
    word_cap: int
    min_chars: int = 200
    target_chars: int = _TARGET_CHARS


def _iter_text_lines(url: str, chunk_size: int = 1 << 20):
    """Stream a remote plain-text file, yielding one decoded line at a time."""
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


def stream_indiccorp_v2(
    language: str,
    out_dir: Path | str,
    config: IndicCorpConfig,
    stats_path: Path | str | None = None,
) -> dict:
    """Stream IndicCorpV2's as.txt, batching lines into pseudo-documents.

    Same clear-output-dir-first discipline as the other downloaded-tier
    collectors: no resume logic, so a stale partial run would collide with a
    fresh one on rerun.
    """
    import shutil

    assert language == "assamese", "IndicCorpV2 collector only wired for Assamese so far"

    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = {"words": 0, "docs": 0, "lines_seen": 0, "elapsed_seconds": 0.0}
    t0 = time.time()
    url = _URL.format(lang="as")

    def flush(writer: ShardWriter, buf_lines: list[str]) -> int:
        text = "\n".join(buf_lines).strip()
        if len(text) < config.min_chars:
            return 0
        doc_id = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
        doc = Doc(
            id=f"as_indiccorpv2_{doc_id}",
            text=text,
            source="indiccorp-v2",
            collection_method="downloaded",
            url=None,
            retrieved=time.strftime("%Y-%m-%d"),
            licence="see ai4bharat/IndicCorpV2 on HuggingFace (CC0)",
            meta={},
        )
        writer.write(doc)
        n = doc.n_words()
        counts["words"] += n
        counts["docs"] += 1
        return n

    with ShardWriter(out_dir, prefix="as_indiccorpv2") as writer:
        pending: list[str] = []
        pending_chars = 0
        for line in _iter_text_lines(url):
            counts["lines_seen"] += 1
            line = line.strip()
            if not line:
                continue
            pending.append(line)
            pending_chars += len(line)
            if pending_chars >= config.target_chars:
                flush(writer, pending)
                pending = []
                pending_chars = 0
                if counts["words"] >= config.word_cap:
                    break
        if pending and counts["words"] < config.word_cap:
            flush(writer, pending)

    counts["elapsed_seconds"] = round(time.time() - t0, 1)
    counts["word_cap"] = config.word_cap

    stats_path = Path(stats_path or f"{language}/stats/collect_indiccorpv2.json")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(counts, indent=2, ensure_ascii=False), encoding="utf-8")
    return counts


def main(argv: list[str] | None = None) -> int:
    """CLI. Example: python -m common.sources.indiccorp_v2 --word-cap 60000000"""
    import argparse

    enable_utf8_console()
    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("--word-cap", type=int, required=True)
    p.add_argument("--min-chars", type=int, default=200)
    args = p.parse_args(argv)

    cfg = IndicCorpConfig(word_cap=args.word_cap, min_chars=args.min_chars)
    result = stream_indiccorp_v2("assamese", "data/assamese/raw/indiccorpv2", cfg)
    print(f"assamese/indiccorpv2: {result['docs']:,} docs, {result['words']:,} words "
          f"in {result['elapsed_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
