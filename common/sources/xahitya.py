"""Stream the xahitya-assamese-corpus (Ranjit89, HuggingFace) as a fifth
downloaded-tier source for Assamese.

**Why this source.** Every prior downloaded source is now either exhausted
(Sangraha) or fully drawn (IndicCorpV2, MADLAD, Glot500's useful tags), and
Assamese's manual ceiling is reached (archive.org, news, Wikiquote all
checked -- see common/sources/archive_books.py and the project plan). Xahitya.org is
an Assamese literature site; this corpus is someone else's scrape of it
already organized on HuggingFace, which makes it downloaded rather than
manual under the course ruling, but it is genuinely new content -- verified
directly by sampling real article text (an editorial titled "এটা ভাষাৰ মৃত্যু
যন্ত্ৰণা", with a real xahitya.org URL and timestamp) before building
anything to consume it.

**Article boundaries, not naive line-batching.** The source file
(`xahitya_dump/corpus.txt`) is a flat concatenation of articles, each
shaped as: title line, ISO-8601 timestamp line, URL line, title line again,
blank, then paragraphs separated by blank lines. Treating this as
undifferentiated text (the way `common/sources/indiccorp_v2.py` batches its flat,
already-sentence-level source) would interleave titles/timestamps/URLs into
the middle of pseudo-documents. Instead this module detects real article
boundaries via the timestamp line (an unambiguous anchor: `\\d{4}-\\d{2}-\\d{2}
T\\d{2}:\\d{2}:\\d{2}`) and reconstructs one Doc per article, with the URL
carried into `Doc.url` properly rather than left in the body text.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from common.io import Doc, ShardWriter, enable_utf8_console

__all__ = ["XahityaConfig", "stream_xahitya"]

_URL = "https://huggingface.co/datasets/Ranjit89/xahitya-assamese-corpus/resolve/main/xahitya_dump/corpus.txt"
_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$")


@dataclass
class XahityaConfig:
    word_cap: int
    min_chars: int = 200


def _iter_text_lines(url: str, chunk_size: int = 1 << 20):
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


def _iter_articles(lines):
    """Reconstruct (title, timestamp, url, body_lines) tuples from the flat
    line stream, anchored on the timestamp line -- see module docstring."""
    window: list[str] = []  # last few lines, to recover the title before a timestamp match
    pending_title = pending_ts = pending_url = None
    body: list[str] = []
    seen_header = False

    for line in lines:
        if _TIMESTAMP_RE.match(line.strip()):
            # Flush whatever article was being built.
            if seen_header and body:
                yield pending_title, pending_ts, pending_url, body
            pending_ts = line.strip()
            pending_title = window[-1] if window else ""
            pending_url = None
            body = []
            seen_header = True
            continue
        if seen_header and pending_url is None:
            pending_url = line.strip()
            continue
        if seen_header:
            # Skip the repeated title line and the blank right after it.
            if line.strip() == pending_title or not line.strip():
                if line.strip() == pending_title:
                    continue
                continue
            body.append(line)
        window.append(line)
        if len(window) > 3:
            window.pop(0)

    if seen_header and body:
        yield pending_title, pending_ts, pending_url, body


def stream_xahitya(
    language: str,
    out_dir: Path | str,
    config: XahityaConfig,
    stats_path: Path | str | None = None,
) -> dict:
    """Stream xahitya-assamese-corpus, one Doc per reconstructed article."""
    import shutil

    assert language == "assamese", "xahitya collector only wired for Assamese"

    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = {"words": 0, "docs": 0, "articles_seen": 0, "too_short": 0}
    t0 = time.time()

    with ShardWriter(out_dir, prefix="as_xahitya") as writer:
        for title, ts, url, body_lines in _iter_articles(_iter_text_lines(_URL)):
            counts["articles_seen"] += 1
            if counts["words"] >= config.word_cap:
                break
            text = "\n\n".join(l for l in body_lines if l.strip()).strip()
            if len(text) < config.min_chars:
                counts["too_short"] += 1
                continue
            doc_id = hashlib.sha1((url or text).encode("utf-8")).hexdigest()[:16]
            doc = Doc(
                id=f"as_xahitya_{doc_id}",
                text=text,
                source="xahitya-assamese-corpus",
                collection_method="downloaded",
                url=url,
                retrieved=time.strftime("%Y-%m-%d"),
                licence="see Ranjit89/xahitya-assamese-corpus on HuggingFace",
                meta={"title": (title or "")[:200], "published": ts},
            )
            writer.write(doc)
            n = doc.n_words()
            counts["words"] += n
            counts["docs"] += 1

    counts["elapsed_seconds"] = round(time.time() - t0, 1)
    counts["word_cap"] = config.word_cap

    stats_path = Path(stats_path or f"{language}/stats/collect_xahitya.json")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(counts, indent=2, ensure_ascii=False), encoding="utf-8")
    return counts


def main(argv: list[str] | None = None) -> int:
    """CLI. Example: python -m common.sources.xahitya --word-cap 999000000"""
    import argparse

    enable_utf8_console()
    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("--word-cap", type=int, required=True)
    p.add_argument("--min-chars", type=int, default=200)
    args = p.parse_args(argv)

    cfg = XahityaConfig(word_cap=args.word_cap, min_chars=args.min_chars)
    result = stream_xahitya("assamese", "data/assamese/raw/xahitya", cfg)
    print(f"assamese/xahitya: {result['docs']:,} docs, {result['words']:,} words "
          f"in {result['elapsed_seconds']}s (articles_seen={result['articles_seen']:,})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
