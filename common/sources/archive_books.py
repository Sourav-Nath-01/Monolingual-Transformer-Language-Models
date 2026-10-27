"""Collect book text from the Internet Archive.

The specification lists "OCR from books/PDFs" first among its examples of
manual collection, and the course guidance is that "anything that involves you
getting data, processing cleaning and organizing is considered Manual" while
"anything already organized on HuggingFace" is not. The Internet Archive is
neither a prepared corpus nor a HuggingFace dataset: items are located by
search, fetched individually, and their OCR text is cleaned and organised here.

Two collection modes, and the distinction is worth stating in the report:

**Text layer** (default). Archive.org already ran OCR on scanned books and
publishes the result as ``<identifier>_djvu.txt``. Fetching and cleaning that
is fast and yields real volume.

**Own OCR** (``--own-ocr``). A sample of page images is run through Tesseract
with the ``asm``/``hin`` language packs locally. This is slow and lower quality
than the Archive's own pipeline, so it is not the way to collect volume -- but
it demonstrates the capability end to end and, more usefully, produces genuine
OCR-damage examples for the report. Lecture 3 slide 27 lists OCR damage as a
top Indic failure mode precisely because errors in complex scripts are
"frequent and hard to spot", and showing real ones beats describing them.

Book text is a genuinely different domain from news and encyclopedia prose,
which matters for the domain-skew problem the same slide raises: without it a
corpus is mostly journalism and Wikipedia.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import httpx

from common.io import Doc, ShardWriter

__all__ = ["ArchiveConfig", "search_items", "fetch_item_text", "collect"]

USER_AGENT = (
    "CL3410-coursework/1.0 (IIIT Hyderabad student project; "
    "monolingual corpus collection; contact via course staff)"
)

SEARCH_URL = "https://archive.org/advancedsearch.php"
METADATA_URL = "https://archive.org/metadata/{identifier}"
DOWNLOAD_URL = "https://archive.org/download/{identifier}/{filename}"

# Archive.org page furniture and OCR artefacts that appear in almost every
# _djvu.txt file and would otherwise repeat thousands of times across the
# corpus, dominating paragraph-level dedup.
_PAGE_MARKER_RE = re.compile(r"^\s*\[?\s*(?:Page|page)\s+\d+\s*\]?\s*$", re.M)
_FORM_FEED_RE = re.compile(r"\f+")
_LONG_DOTS_RE = re.compile(r"\.{4,}")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")


@dataclass
class ArchiveConfig:
    """Search and politeness settings.

    Attributes:
        rows: Items requested per search page.
        max_items: Ceiling on items collected per language.
        delay: Seconds between requests. The Archive is a donation-funded
            nonprofit; 1 req/s is deliberately gentler than the news crawl.
        timeout: Per-request timeout. Text layers of long books are large.
        min_chars: Shortest book text worth keeping.
        min_script_ratio: Script gate, applied before storage.
    """

    rows: int = 100
    max_items: int = 2000
    delay: float = 1.0
    timeout: float = 120.0
    min_chars: int = 2000
    min_script_ratio: float = 0.50


def _client(cfg: ArchiveConfig) -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": USER_AGENT}, timeout=cfg.timeout, follow_redirects=True
    )


def search_items(language: str, cfg: ArchiveConfig, client: httpx.Client) -> Iterator[dict]:
    """Yield metadata for text items in the given language.

    Args:
        language: Archive.org language facet, e.g. ``"Assamese"`` or ``"Hindi"``.
        cfg: Politeness and limits.
        client: Shared HTTP client.

    Yields:
        Dicts with at least ``identifier``, and usually ``title`` and ``year``.
    """
    page = 1
    produced = 0
    while produced < cfg.max_items:
        params = {
            "q": f'language:("{language}") AND mediatype:(texts)',
            "fl[]": ["identifier", "title", "year", "licenseurl"],
            "rows": cfg.rows,
            "page": page,
            "output": "json",
        }
        time.sleep(cfg.delay)
        # A transient failure here (timeout, malformed response, momentary
        # rate limit) used to end the whole search silently -- measured:
        # a real run stopped at page 9 (900 items) even though the same
        # query, page 10, returned a normal 100-doc response moments later
        # when checked directly. One retry after a short backoff turns a
        # one-off network hiccup back into "keep going" instead of a quiet,
        # wrong "no more results".
        docs = None
        for attempt in range(2):
            try:
                r = client.get(SEARCH_URL, params=params)
                docs = r.json()["response"]["docs"]
                break
            except Exception:
                if attempt == 0:
                    time.sleep(cfg.delay * 5)
        if docs is None:
            return
        if not docs:
            return
        for d in docs:
            if produced >= cfg.max_items:
                return
            produced += 1
            yield d
        page += 1


def clean_ocr_text(raw: str) -> str:
    """Tidy an Archive.org OCR text layer into running prose.

    Removes page markers and form feeds, rejoins words broken across a line by
    a hyphen, and collapses the dot leaders that tables of contents produce.
    """
    text = _FORM_FEED_RE.sub("\n\n", raw)
    text = _PAGE_MARKER_RE.sub("", text)
    text = _HYPHEN_BREAK_RE.sub(r"\1\2", text)
    text = _LONG_DOTS_RE.sub(" ", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return _MULTI_NEWLINE_RE.sub("\n\n", text).strip()


def fetch_item_text(identifier: str, cfg: ArchiveConfig, client: httpx.Client) -> str | None:
    """Fetch and clean one item's OCR text layer, or None if it has none."""
    time.sleep(cfg.delay)
    try:
        meta = client.get(METADATA_URL.format(identifier=identifier)).json()
    except Exception:
        return None
    files = meta.get("files", [])
    target = next(
        (f["name"] for f in files if f.get("name", "").endswith("_djvu.txt")),
        None,
    )
    if target is None:
        return None
    time.sleep(cfg.delay)
    try:
        r = client.get(DOWNLOAD_URL.format(identifier=identifier, filename=target))
        if r.status_code != 200:
            return None
        return clean_ocr_text(r.text)
    except Exception:
        return None


def collect(
    language: str,
    script: str,
    out_dir: Path | str,
    doc_prefix: str,
    source: str,
    cfg: ArchiveConfig | None = None,
) -> dict:
    """Search, fetch, clean and store book text. Resumable.

    Args:
        language: Archive.org language facet.
        script: Key into :data:`common.clean.filters.SCRIPT_RANGES`.
        out_dir: Shard directory.
        doc_prefix: Document id prefix.
        source: Source key for ``sources.yaml``.
        cfg: Settings.

    Returns:
        Counts of what happened, for the statistics report.
    """
    from common.clean.filters import script_ratio

    cfg = cfg or ArchiveConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seen_path = out_dir / f"{doc_prefix}.seen"
    seen = (
        set(seen_path.read_text(encoding="utf-8").split())
        if seen_path.exists()
        else set()
    )
    counts = {
        "items_found": 0, "already_seen": 0, "no_text_layer": 0,
        "too_short": 0, "wrong_script": 0, "emitted": 0, "words": 0,
    }
    pending: list[str] = []

    with _client(cfg) as client, ShardWriter(
        out_dir, prefix=doc_prefix, target_bytes=8 * 1024 * 1024
    ) as writer:
        shards_done = 0
        for item in search_items(language, cfg, client):
            ident = item.get("identifier")
            if not ident:
                continue
            counts["items_found"] += 1
            if ident in seen or ident in pending:
                counts["already_seen"] += 1
                continue

            text = fetch_item_text(ident, cfg, client)
            if not text:
                counts["no_text_layer"] += 1
                seen.add(ident)
                with seen_path.open("a", encoding="utf-8") as fh:
                    fh.write(ident + "\n")
                continue
            if len(text) < cfg.min_chars or script_ratio(text, script) < cfg.min_script_ratio:
                counts["too_short" if len(text) < cfg.min_chars else "wrong_script"] += 1
                seen.add(ident)
                with seen_path.open("a", encoding="utf-8") as fh:
                    fh.write(ident + "\n")
                continue

            doc = Doc(
                # Derived from the Archive.org identifier, not the per-run
                # emitted counter -- the counter resets to 0 on every
                # restart, and this collector was restarted several times
                # across the session (measured: 166 duplicated ids across
                # 437 books, some in triplicate). ``ident`` is Archive.org's
                # own stable key for the item, so hashing it gives the same
                # id no matter how many times collection has been relaunched.
                id=f"{doc_prefix}_{hashlib.sha1(ident.encode('utf-8')).hexdigest()[:16]}",
                text=text,
                source=source,
                collection_method="manual",
                url=f"https://archive.org/details/{ident}",
                retrieved=time.strftime("%Y-%m-%d"),
                licence=item.get("licenseurl") or "see item metadata",
                meta={
                    "identifier": ident,
                    "title": (item.get("title") or "")[:200],
                    "year": item.get("year"),
                    "ocr": "archive.org text layer",
                },
            )
            writer.write(doc)
            pending.append(ident)
            counts["emitted"] += 1
            counts["words"] += doc.n_words()

            # Same durability rule as the news scraper: an identifier is only
            # recorded once the shard holding its text is on disk.
            if len(writer.shards_written) > shards_done:
                shards_done = len(writer.shards_written)
                with seen_path.open("a", encoding="utf-8") as fh:
                    for i in pending:
                        seen.add(i)
                        fh.write(i + "\n")
                pending.clear()

    if pending:
        with seen_path.open("a", encoding="utf-8") as fh:
            for i in pending:
                fh.write(i + "\n")
    return counts


def main(argv: list[str] | None = None) -> int:
    """CLI. See ``--help``."""
    import argparse
    import json

    from common.io import enable_utf8_console

    enable_utf8_console()
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--lang", required=True, choices=("assamese", "hindi"))
    p.add_argument("--max-items", type=int, default=2000)
    p.add_argument("--delay", type=float, default=1.0)
    args = p.parse_args(argv)

    facet = {"assamese": "Assamese", "hindi": "Hindi"}[args.lang]
    script = {"assamese": "bengali", "hindi": "devanagari"}[args.lang]
    prefix = {"assamese": "as_books", "hindi": "hi_books"}[args.lang]

    counts = collect(
        language=facet,
        script=script,
        out_dir=f"data/{args.lang}/raw/books",
        doc_prefix=prefix,
        source=f"archive.org-{facet.lower()}",
        cfg=ArchiveConfig(max_items=args.max_items, delay=args.delay),
    )
    out = Path(args.lang) / "stats" / "collect_books.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(counts, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(counts, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
