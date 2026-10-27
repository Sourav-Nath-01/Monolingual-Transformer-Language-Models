"""Stream the downloaded tier: Sangraha, capped at 4xM.

This is the *other* half of the token budget. The manual tier (Wikipedia,
Wikisource, news, archive.org books) is what we collected and extracted
ourselves; everything here is a pre-existing corpus we did not build,
identified per the course ruling as "already organized on HuggingFace" and
therefore explicitly NOT manual (plan S2, S3.2).

**Why Sangraha, and why "verified" before "unverified".**
    Lecture 3 slide 26: "For the larger Indian languages, start here rather
    than with a raw crawl." Sangraha (AI4Bharat) ships three configs on
    HuggingFace: verified, unverified, synthetic. "synthetic" is
    machine-translated/generated text -- exactly what the TA ruled out
    ("machine translated data nahh"), so it is never touched here. verified
    is drawn first, unverified only if verified alone cannot fill the cap.

**Streaming, not downloading.**
    `datasets` streaming mode reads parquet shards lazily and never
    materialises the full split on disk -- necessary given Sangraha Hindi
    alone runs into many GB. Rows are counted against a word-proxy cap
    (matching the collection-time proxy convention in plan S2.1) and the
    stream stops the moment the cap is crossed, not after finishing the
    current shard.

**The 4xM cap, and why it is enforced here rather than trusted to yield.**
    S2's rule: draw exactly 4 x M downloaded tokens, where M is the cleaned
    manual-tier token count, capped at 400M for Hindi. That keeps the 20%
    manual floor true by construction rather than by hoping. The cap passed
    in here is the *word* form of that token cap (tokens / measured
    fertility), because collection time has no tokenizer yet to count tokens
    directly (S2.1).

**Mislabelled rows are expected, not a bug to route around.**
    A live check of Sangraha's own "asm" verified split returned a document
    in a language that was not Assamese at all in its very first sample.
    Downloaded corpora are not assumed clean -- every row collected here
    still gets tagged `collection_method="downloaded"` and flows through the
    exact same language-identification stage (common/clean/filters.py) as
    everything else before it counts as anything. This module's only job is
    to fetch and store; filtering happens downstream, uniformly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from common.io import Doc, ShardWriter, enable_utf8_console

__all__ = ["SangrahaConfig", "LANG_ISO", "stream_sangraha"]

# Sangraha's own split naming is ISO 639-3, not the two-letter codes used
# elsewhere in this project.
LANG_ISO = {"hindi": "hin", "assamese": "asm"}

# Machine-generated text is deliberately excluded; see module docstring.
_ALLOWED_CONFIGS = ("verified", "unverified")


@dataclass
class SangrahaConfig:
    """Streaming and stopping-condition settings.

    Attributes:
        word_cap: Stop once this many raw proxy words have been collected.
            This is the word-form of the plan's `4 x M` token cap.
        min_chars: Rows shorter than this are not worth storing -- Sangraha
            includes short web snippets that would just be dropped by the
            quality filter downstream anyway.
        configs: Which Sangraha configs to draw from, in order. "synthetic"
            must never appear here.
    """

    word_cap: int
    min_chars: int = 200
    configs: tuple[str, ...] = _ALLOWED_CONFIGS

    def __post_init__(self):
        assert "synthetic" not in self.configs, (
            "synthetic is machine-generated text; the TA ruling excludes it"
        )


def stream_sangraha(
    language: str,
    out_dir: Path | str,
    config: SangrahaConfig,
    stats_path: Path | str | None = None,
) -> dict:
    """Stream Sangraha for one language until the word cap is reached.

    Args:
        language: ``"hindi"`` or ``"assamese"``.
        out_dir: Shard output directory, e.g. ``data/hindi/raw/downloaded``.
        config: Cap and source settings.
        stats_path: Where to write the collection report. Defaults to
            ``<language>/stats/collect_downloaded.json``.

    Returns:
        Counts: words collected per Sangraha config, documents emitted,
        elapsed time.
    """
    import shutil

    from datasets import load_dataset

    iso = LANG_ISO[language]
    out_dir = Path(out_dir)
    # Always start clean. This collector has no resume/skip mechanism -- each
    # call re-streams the dataset from the beginning -- so any pre-existing
    # shard here (a prior smoke test, an aborted run) would have its
    # documents re-emitted by this run too. Same doc_id in, same constructed
    # Doc.id out, so the leftover and the fresh copy collide once dedup runs.
    # Measured: a 13-document, 5,209-word smoke test left one small shard
    # behind; the real 157M-word run re-streamed from row zero and reproduced
    # those same 13 documents partway through, producing 32 duplicate ids
    # that only surfaced as a MinHash/LSH insert crash much later. Clearing
    # first makes "re-run this collector" always safe.
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = {"words": 0, "docs": 0, "by_config": {}, "too_short": 0}
    t0 = time.time()

    with ShardWriter(out_dir, prefix=f"{language[:2]}_sangraha") as writer:
        for cfg_name in config.configs:
            if counts["words"] >= config.word_cap:
                break
            cfg_words = 0
            try:
                ds = load_dataset(
                    "ai4bharat/sangraha",
                    data_files={iso: f"{cfg_name}/{iso}/*.parquet"},
                    split=iso,
                    streaming=True,
                )
            except Exception as exc:
                # A config with no data for this language (e.g. unverified
                # not covering every language) is not an error -- just move
                # on to the next config in priority order.
                counts["by_config"][cfg_name] = f"unavailable: {exc}"
                continue

            for row in ds:
                if counts["words"] >= config.word_cap:
                    break
                text = (row.get("text") or "").strip()
                if len(text) < config.min_chars:
                    counts["too_short"] += 1
                    continue

                doc = Doc(
                    id=f"{language[:2]}_sangraha_{cfg_name}_{row.get('doc_id', counts['docs'])}",
                    text=text,
                    source=f"sangraha-{cfg_name}",
                    collection_method="downloaded",
                    url=None,
                    retrieved=time.strftime("%Y-%m-%d"),
                    licence="see ai4bharat/sangraha on HuggingFace",
                    meta={"sangraha_type": row.get("type", "")},
                )
                writer.write(doc)
                n = doc.n_words()
                counts["words"] += n
                cfg_words += n
                counts["docs"] += 1

            counts["by_config"][cfg_name] = cfg_words

    counts["elapsed_seconds"] = round(time.time() - t0, 1)
    counts["word_cap"] = config.word_cap

    import json

    stats_path = Path(stats_path or f"{language}/stats/collect_downloaded.json")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(counts, indent=2, ensure_ascii=False), encoding="utf-8")
    return counts


def main(argv: list[str] | None = None) -> int:
    """CLI. Example: python -m common.sources.downloaded --lang hindi --word-cap 5000000"""
    import argparse

    enable_utf8_console()
    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("--lang", required=True, choices=("hindi", "assamese"))
    p.add_argument("--word-cap", type=int, required=True)
    p.add_argument("--min-chars", type=int, default=200)
    args = p.parse_args(argv)

    cfg = SangrahaConfig(word_cap=args.word_cap, min_chars=args.min_chars)
    result = stream_sangraha(args.lang, f"data/{args.lang}/raw/downloaded", cfg)
    print(f"{args.lang}: {result['docs']:,} docs, {result['words']:,} words "
          f"in {result['elapsed_seconds']}s")
    print(f"  by config: {result['by_config']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
