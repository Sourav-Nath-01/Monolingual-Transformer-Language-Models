"""Stream Glot500 (cis-lmu) as a fourth downloaded-tier source for Assamese.

**Why a fourth source, and why filtered rather than taken whole.** After
IndicCorpV2 and MADLAD-400, Assamese sat at ~463M tokens -- close to 500M but
short, and the 20% manual floor (measured at 20.72%, thin margin) blocks
further *unfiltered* downloaded draws (plan §0.0/§2). Glot500 is an
aggregation of many underlying corpora, tagged per-row by its own `dataset`
column. Checked directly against this project's Assamese portion: **64.5% of
rows are tagged "Indiccorp"** -- almost certainly the same IndicCorp lineage
already pulled in via `common/sources/indiccorp_v2.py`, so processing it would spend
real compute on data the dedup pass would mostly discard anyway. Two more
tags (`Flores200`, `Tatoeba`) are tiny and not worth the risk: Flores200 is
a translation *evaluation* benchmark, and training on eval-benchmark text is
a contamination risk worth avoiding on principle even before any Phase 2/3
work touches it.

**What's kept:** `Oscar22`, `CC100`, `mtdata`, `Wikipedia`, `Leipzig_wikipedia`,
`Leipzig_community`, `Earthlings`, `wiki` -- about 35% of rows. Two of these
(`Oscar22`, `CC100`) are notable for a different reason: this project tried
both directly (`common/sources/downloaded.py`'s module docstring era of probing, and
a direct CC100 host check) and found OSCAR gated behind HF authentication
and CC100 no longer hosted at its documented path. Glot500 is, concretely,
how this project actually gets that data.

**Document granularity.** Same wrinkle as IndicCorpV2: ~459 bytes/row
average (measured: 862MB / 1,876,644 rows), well under `min_doc_chars=200`
after accounting for UTF-8 multi-byte Assamese characters bringing typical
rows close to the floor. Consecutive same-source rows are batched into
pseudo-documents the same way `common/sources/indiccorp_v2.py` does, and never mixed
across `dataset` tags -- concatenating an OSCAR row with a Wikipedia row
into one pseudo-document would blur the provenance the `source` field is
supposed to record.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

from common.io import Doc, ShardWriter, enable_utf8_console

__all__ = ["Glot500Config", "stream_glot500"]

# See module docstring for why each tag is included or excluded. "Indiccorp"
# was originally excluded here on the assumption it mostly duplicates
# common/sources/indiccorp_v2.py's own draw -- reconsidered: the corpus-wide dedup
# pass is specifically built to discard true duplicates safely (that is its
# whole job), so pre-filtering it out only saves compute time, it does not
# change correctness. With the priority being maximum genuine yield, it is
# included and left for dedup to sort out, same as every other overlapping
# source this pipeline has already handled (Sangraha vs. MADLAD vs.
# IndicCorpV2 all overlap each other too).
_KEEP_TAGS = (
    "Oscar22", "CC100", "mtdata", "Wikipedia", "Indiccorp",
    "Leipzig_wikipedia", "Leipzig_community", "Earthlings", "wiki",
)
_TARGET_CHARS = 2000


@dataclass
class Glot500Config:
    word_cap: int
    min_chars: int = 200
    target_chars: int = _TARGET_CHARS
    keep_tags: tuple[str, ...] = _KEEP_TAGS


def stream_glot500(
    language: str,
    out_dir: Path | str,
    config: Glot500Config,
    stats_path: Path | str | None = None,
) -> dict:
    """Load Glot500's Assamese split, keep only low-overlap-risk tags, batch
    into pseudo-documents, write shards.

    Uses the `datasets` library's own Arrow loading (memory-mapped, not
    materialised into RAM) rather than a manual HTTP stream -- Glot500 ships
    genuine Arrow files, not a flat text/jsonl source, so there is no reason
    to reimplement what `datasets` already does correctly.
    """
    import shutil
    from datasets import load_dataset

    assert language == "assamese", "Glot500 collector only wired for Assamese so far"

    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("cis-lmu/Glot500", "asm_Beng", split="train")

    counts = {"words": 0, "docs": 0, "rows_seen": 0, "rows_kept_by_tag": {}, "rows_skipped": 0}
    t0 = time.time()

    def flush(writer: ShardWriter, tag: str, buf_lines: list[str]) -> int:
        text = "\n".join(buf_lines).strip()
        if len(text) < config.min_chars:
            return 0
        doc_id = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
        doc = Doc(
            id=f"as_glot500_{doc_id}",
            text=text,
            source=f"glot500-{tag}",
            collection_method="downloaded",
            url=None,
            retrieved=time.strftime("%Y-%m-%d"),
            licence="see cis-lmu/Glot500 on HuggingFace (per-subcorpus licences)",
            meta={"glot500_tag": tag},
        )
        writer.write(doc)
        n = doc.n_words()
        counts["words"] += n
        counts["docs"] += 1
        return n

    with ShardWriter(out_dir, prefix="as_glot500") as writer:
        pending: dict[str, list[str]] = {}
        pending_chars: dict[str, int] = {}
        for row in ds:
            counts["rows_seen"] += 1
            if counts["words"] >= config.word_cap:
                break
            tag = row.get("dataset", "")
            if tag not in config.keep_tags:
                counts["rows_skipped"] += 1
                continue
            line = (row.get("text") or "").strip()
            if not line:
                continue
            pending.setdefault(tag, [])
            pending_chars.setdefault(tag, 0)
            pending[tag].append(line)
            pending_chars[tag] += len(line)
            if pending_chars[tag] >= config.target_chars:
                n = flush(writer, tag, pending[tag])
                counts["rows_kept_by_tag"][tag] = counts["rows_kept_by_tag"].get(tag, 0) + n
                pending[tag] = []
                pending_chars[tag] = 0

        if counts["words"] < config.word_cap:
            for tag, lines in pending.items():
                if lines:
                    n = flush(writer, tag, lines)
                    counts["rows_kept_by_tag"][tag] = counts["rows_kept_by_tag"].get(tag, 0) + n

    counts["elapsed_seconds"] = round(time.time() - t0, 1)
    counts["word_cap"] = config.word_cap

    stats_path = Path(stats_path or f"{language}/stats/collect_glot500.json")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(counts, indent=2, ensure_ascii=False), encoding="utf-8")
    return counts


def main(argv: list[str] | None = None) -> int:
    """CLI. Example: python -m common.sources.glot500 --word-cap 999000000"""
    import argparse

    enable_utf8_console()
    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("--word-cap", type=int, required=True)
    p.add_argument("--min-chars", type=int, default=200)
    args = p.parse_args(argv)

    cfg = Glot500Config(word_cap=args.word_cap, min_chars=args.min_chars)
    result = stream_glot500("assamese", "data/assamese/raw/glot500", cfg)
    print(f"assamese/glot500: {result['docs']:,} docs, {result['words']:,} words "
          f"in {result['elapsed_seconds']}s (rows_seen={result['rows_seen']:,}, "
          f"rows_skipped={result['rows_skipped']:,})")
    print(f"  by tag: {result['rows_kept_by_tag']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
