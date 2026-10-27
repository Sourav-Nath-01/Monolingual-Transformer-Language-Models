"""Precision/recall of the Assamese/Bengali discriminator against a
source-provenance ground-truth set (spec §1.2, plan §4.2).

**Why source provenance, not re-derived labels.** Labelling a sample using
the same script-codepoint/function-word signals the classifier itself uses
would be circular -- it would show the classifier agrees with itself, not
that it is correct. Ground truth here comes from *where the text was
collected*, independent of the classifier: Assamese-label documents are
drawn from this project's own Assamese corpus (built and filtered as
Assamese elsewhere in the pipeline); Bengali-label documents are drawn
directly from IndicCorpV2's `bn.txt`, a source never touched by the Assamese
collection. Neither draw runs through `AssameseBengaliClassifier` before
being treated as ground truth, so a correct classification is a genuine
agreement, not the same signal checking itself.

**Same corpus family on both sides, deliberately.** Both samples come from
IndicCorpV2 ( Assamese via the project's normal downloaded-tier draw,
Bengali via a small held-out fetch here) rather than mixing Assamese-from-
Wikipedia against Bengali-from-news, so the comparison isolates *language*
rather than picking up a genre or source artifact as a shortcut signal.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from common.clean.filters import AssameseBengaliClassifier
from common.io import Doc, iter_docs, enable_utf8_console

__all__ = ["fetch_bengali_sample", "sample_assamese", "evaluate"]

_URL = "https://huggingface.co/datasets/ai4bharat/IndicCorpV2/resolve/main/data/bn.txt"
_TARGET_CHARS = 2000  # same pseudo-document chunking as common/sources/indiccorp_v2.py


def _iter_text_lines(url: str, chunk_size: int = 1 << 20):
    buf = b""
    with httpx.stream("GET", url, follow_redirects=True, timeout=60.0) as resp:
        resp.raise_for_status()
        for chunk in resp.iter_bytes(chunk_size):
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                yield line.decode("utf-8", errors="ignore")


def fetch_bengali_sample(n_docs: int, min_chars: int = 200) -> list[Doc]:
    """Pull a small held-out Bengali sample directly from IndicCorpV2's
    bn.txt -- never written to data/, never enters any Assamese-labelled
    directory, purely in-memory for this evaluation."""
    docs = []
    pending: list[str] = []
    pending_chars = 0
    for line in _iter_text_lines(_URL):
        line = line.strip()
        if not line:
            continue
        pending.append(line)
        pending_chars += len(line)
        if pending_chars >= _TARGET_CHARS:
            text = "\n".join(pending).strip()
            if len(text) >= min_chars:
                doc_id = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
                docs.append(Doc(id=f"bn_eval_{doc_id}", text=text, source="indiccorp-v2-bn-eval",
                                 collection_method="downloaded"))
            pending, pending_chars = [], 0
            if len(docs) >= n_docs:
                break
    return docs


def sample_assamese(corpus_dir: Path | str, n_docs: int, seed: int = 42) -> list[Doc]:
    """Reservoir-sample n_docs from the project's own final Assamese corpus."""
    rng = random.Random(seed)
    reservoir: list[Doc] = []
    for i, doc in enumerate(iter_docs(corpus_dir)):
        if len(reservoir) < n_docs:
            reservoir.append(doc)
        else:
            j = rng.randint(0, i)
            if j < n_docs:
                reservoir[j] = doc
    return reservoir


def evaluate(n_per_class: int = 50, corpus_dir: str = "data/assamese/clean_deduped") -> dict:
    clf = AssameseBengaliClassifier()

    t0 = time.time()
    as_docs = sample_assamese(corpus_dir, n_per_class)
    bn_docs = fetch_bengali_sample(n_per_class)

    rows = []
    tp = fp = fn = tn = 0  # positive class = "assamese"
    for doc, truth in [(d, "assamese") for d in as_docs] + [(d, "bengali") for d in bn_docs]:
        decision = clf.classify(doc.text)
        pred = decision.label
        rows.append({"id": doc.id, "truth": truth, "predicted": pred,
                      "evidence": decision.evidence, "text_sample": doc.text[:150]})
        if truth == "assamese" and pred == "assamese":
            tp += 1
        elif truth == "bengali" and pred == "assamese":
            fp += 1
        elif truth == "assamese" and pred != "assamese":
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall else None

    return {
        "n_assamese": len(as_docs),
        "n_bengali": len(bn_docs),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "elapsed_seconds": round(time.time() - t0, 1),
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI. Example: python -m common.clean.eval_language_id --n 50"""
    import argparse

    enable_utf8_console()
    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("--n", type=int, default=50, help="documents per class")
    args = p.parse_args(argv)

    r = evaluate(args.n)
    print(f"n_assamese={r['n_assamese']} n_bengali={r['n_bengali']} "
          f"precision={r['precision']:.3f} recall={r['recall']:.3f} f1={r['f1']:.3f}")
    print(f"confusion: {r['confusion']}")

    out = Path("assamese/stats/language_id_eval.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
