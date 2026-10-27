"""Stream MADLAD-400 (allenai) as a second downloaded-tier source for Assamese.

**Why a second downloaded source, this late.** Sangraha's non-synthetic
configs are naturally exhausted for Assamese at 157.46M raw words (see
`common/sources/downloaded.py`) -- there is no more of that specific corpus to draw,
and "synthetic" stays banned per the TA ruling. Reaching a materially larger
Assamese total needs a *different* corpus, not a bigger draw from the same
one. MADLAD-400 (web-crawl-derived, AllenAI) is one; `common/sources/indiccorp_v2.py`
is the other.

**Two quality tiers, both used.** MADLAD-400 ships `as_clean` and `as_noisy`
splits per language. `as_clean` is drawn first, matching the
verified-before-unverified precedence already established for Sangraha.
`as_noisy` is included too rather than skipped, because after §0.1a's
ceiling-measurement history on this project, leaving tokens on the table
because a raw-size estimate merely *sounds* risky has not been the right call
before -- the quality filter (script gate, language ID, quality heuristics,
non-language word removal, dedup) is the same one every other source flows
through, so a noisier tier is filtered harder, not exempted from filtering.

**Streamed, not downloaded whole.** The two files are ~100MB and ~150MB
gzip-compressed, several hundred MB uncompressed -- comfortably diskable, but
there is no reason to materialise the whole thing when a byte-stream gzip
decompressor can hand back one JSON line at a time. `zlib.decompressobj` with
the gzip window bit (`MAX_WBITS | 16`) does this without needing a seekable
file handle, which a plain chunked HTTP stream is not.

**Document granularity.** Each JSONL line is one already-segmented web
document (`{"text": "..."}`, no other fields) -- unlike IndicCorpV2's flat
one-line-per-sentence format, no re-chunking is needed here.
"""

from __future__ import annotations

import hashlib
import json
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

import httpx

from common.io import Doc, ShardWriter, enable_utf8_console

__all__ = ["MadladConfig", "stream_madlad"]

_BASE_URL = "https://huggingface.co/datasets/allenai/MADLAD-400/resolve/main/data/{lang}/{split}_0000.jsonl.gz"

# clean before noisy, mirroring Sangraha's verified-before-unverified order.
_SPLITS = ("as_clean", "as_noisy")


@dataclass
class MadladConfig:
    word_cap: int
    min_chars: int = 200
    splits: tuple[str, ...] = _SPLITS


def _iter_gzip_jsonl_lines(url: str, chunk_size: int = 1 << 20):
    """Stream-decompress a remote .jsonl.gz, yielding one decoded line at a time.

    Never holds the full decompressed file in memory -- the decompressor's
    internal state is the only thing carried across chunks, so peak memory is
    the chunk size plus one line, regardless of how large the file is.
    """
    decompressor = zlib.decompressobj(zlib.MAX_WBITS | 16)
    buf = b""
    with httpx.stream("GET", url, follow_redirects=True, timeout=60.0) as resp:
        resp.raise_for_status()
        for chunk in resp.iter_bytes(chunk_size):
            buf += decompressor.decompress(chunk)
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if line.strip():
                    yield line
        buf += decompressor.flush()
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            if line.strip():
                yield line
        if buf.strip():
            yield buf


def stream_madlad(
    language: str,
    out_dir: Path | str,
    config: MadladConfig,
    stats_path: Path | str | None = None,
) -> dict:
    """Stream MADLAD-400's as_clean/as_noisy splits until the word cap is reached.

    Same clear-output-dir-first discipline as `stream_sangraha`: this
    collector has no resume logic, so a stale partial run left behind by an
    earlier attempt would re-emit and collide with a fresh run's ids.
    """
    import shutil

    assert language == "assamese", "MADLAD-400 collector only wired for Assamese so far"

    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = {"words": 0, "docs": 0, "by_split": {}, "too_short": 0}
    t0 = time.time()

    with ShardWriter(out_dir, prefix="as_madlad") as writer:
        for split in config.splits:
            if counts["words"] >= config.word_cap:
                break
            split_words = 0
            url = _BASE_URL.format(lang="as", split=split)
            try:
                lines = _iter_gzip_jsonl_lines(url)
            except Exception as exc:
                counts["by_split"][split] = f"unavailable: {exc}"
                continue

            for line in lines:
                if counts["words"] >= config.word_cap:
                    break
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = (row.get("text") or "").strip()
                if len(text) < config.min_chars:
                    counts["too_short"] += 1
                    continue

                doc_id = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
                doc = Doc(
                    id=f"as_madlad_{split}_{doc_id}",
                    text=text,
                    source=f"madlad400-{split}",
                    collection_method="downloaded",
                    url=None,
                    retrieved=time.strftime("%Y-%m-%d"),
                    licence="see allenai/MADLAD-400 on HuggingFace (ODC-BY)",
                    meta={},
                )
                writer.write(doc)
                n = doc.n_words()
                counts["words"] += n
                split_words += n
                counts["docs"] += 1

            counts["by_split"][split] = split_words

    counts["elapsed_seconds"] = round(time.time() - t0, 1)
    counts["word_cap"] = config.word_cap

    stats_path = Path(stats_path or f"{language}/stats/collect_madlad.json")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(counts, indent=2, ensure_ascii=False), encoding="utf-8")
    return counts


def main(argv: list[str] | None = None) -> int:
    """CLI. Example: python -m common.sources.madlad --word-cap 60000000"""
    import argparse

    enable_utf8_console()
    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("--word-cap", type=int, required=True)
    p.add_argument("--min-chars", type=int, default=200)
    args = p.parse_args(argv)

    cfg = MadladConfig(word_cap=args.word_cap, min_chars=args.min_chars)
    result = stream_madlad("assamese", "data/assamese/raw/madlad", cfg)
    print(f"assamese/madlad: {result['docs']:,} docs, {result['words']:,} words "
          f"in {result['elapsed_seconds']}s")
    print(f"  by split: {result['by_split']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
