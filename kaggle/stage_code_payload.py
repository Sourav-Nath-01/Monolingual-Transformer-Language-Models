"""Stage the minimal code payload that the Kaggle notebook needs.

The GitHub Classroom repository is **private**, so the notebook cannot
``git clone`` it: a Kaggle kernel has no credentials and the clone fails with
"Repository not found". Rather than embedding a personal access token in a
notebook (which would leak the moment the notebook is shared or committed), the
code is shipped the same way the data is -- as a Kaggle Dataset.

Kept deliberately separate from the ``lma-tokens`` data dataset: the token
arrays are 2.3 GB and stable, whereas the code is a few MB and will change
between runs. One combined dataset would mean re-uploading 2.3 GB to fix a
typo.

What the Kaggle run actually needs, and nothing else:

* ``common/`` -- the pipeline and modelling packages
* ``<lang>/configs/model.json`` -- architecture hyperparameters
* ``<lang>/tokenizer/<lang>_bpe_16000.model`` -- for decoding during evaluation
  (bits-per-byte needs real text, and generation samples must be readable)

Excluded on purpose: the corpus, the split manifests (the arrays are already
tokenised and split), the unused sweep tokenizers, and the reports.

Run ``python kaggle/stage_code_payload.py`` and upload the resulting
``kaggle/code_payload/`` directory as a private Kaggle Dataset named
``lma-code``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

VOCAB = 16000
LANGS = ("hindi", "assamese")
OUT = Path(__file__).parent / "code_payload"


def stage() -> Path:
    """Copy the needed files into a clean payload directory. Returns its path."""
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    # common/, minus caches and any stray notebooks
    shutil.copytree(
        "common", OUT / "common",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".ipynb_checkpoints"),
    )

    for lang in LANGS:
        (OUT / lang / "configs").mkdir(parents=True, exist_ok=True)
        (OUT / lang / "tokenizer").mkdir(parents=True, exist_ok=True)
        shutil.copy2(f"{lang}/configs/model.json", OUT / lang / "configs" / "model.json")
        tok = f"{lang}/tokenizer/{lang}_bpe_{VOCAB}.model"
        shutil.copy2(tok, OUT / lang / "tokenizer" / Path(tok).name)

    _write_version(OUT)
    return OUT


def _write_version(root: Path) -> None:
    """Stamp the payload with the commit and time it was built from.

    A Kaggle Dataset is a snapshot, not a live checkout: fixing a bug in the
    repo changes nothing on Kaggle until the dataset gets a new version and the
    notebook is pointed at it. Without a stamp that staleness is invisible --
    the notebook runs happily on month-old code and the only clue is behaviour
    that contradicts the source in front of you. Printing this at startup makes
    "did my fix actually ship" a one-line answer instead of a guess.
    """
    import subprocess
    from datetime import datetime, timezone

    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"],
                               capture_output=True, text=True, check=True).stdout.strip()
        commit = f"{sha}{'+local-changes' if dirty else ''}"
    except Exception:
        commit = "unknown (not a git checkout)"

    built = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    (root / "PAYLOAD_VERSION.txt").write_text(
        f"commit: {commit}\nbuilt:  {built}\n", encoding="utf-8")


def report(root: Path) -> None:
    """Print the payload contents and total size."""
    files = sorted(p for p in root.rglob("*") if p.is_file())
    total = sum(p.stat().st_size for p in files)
    print(f"staged {len(files)} files, {total/1e6:.1f} MB total, at {root}")
    for p in files:
        if p.suffix in (".json", ".model"):
            print(f"  {p.stat().st_size/1e6:7.2f} MB  {p.relative_to(root)}")
    n_py = sum(1 for p in files if p.suffix == ".py")
    print(f"  {'':7s}     + {n_py} Python modules under common/")


if __name__ == "__main__":
    report(stage())
