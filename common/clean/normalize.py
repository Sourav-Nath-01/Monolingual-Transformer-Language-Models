"""Unicode normalisation for Devanagari and Bengali-Assamese text.

Stage 1 of the cleaning pipeline. Runs before language identification, because
every later stage compares characters and codepoints -- and two visually
identical strings that differ in encoding would otherwise be treated as
different words by the tokenizer, fragmenting the vocabulary.

Four decisions here are worth defending in the report:

**NFC, not NFKC.**
    NFKC applies compatibility folding, which collapses distinctions that
    matter in Indic scripts. NFC is canonical-only and is what the project
    specification asks for. A useful side effect: the precomposed nukta
    characters (Devanagari U+0958-U+095F, Bengali U+09DC/U+09DD/U+09DF) are
    composition exclusions, so NFC rewrites them to base + nukta. That is the
    behaviour we want -- the same sound ends up with one representation
    instead of two.

**Zero-width joiners are edited, not deleted.**
    ZWJ (U+200D) and ZWNJ (U+200C) are orthographically load-bearing in Indic
    scripts: after a virama they select between a ligature and an explicit
    half-form. Deleting them wholesale changes how words render. But most
    joiners in scraped text are strays from copy-paste and markup, and they
    fragment the vocabulary because the same word appears with and without
    them. So the rule is positional: a joiner immediately after a virama is
    doing orthographic work and is kept; every other joiner is a stray and is
    removed. Both counts are recorded, so the report can state how many of
    each were found rather than asserting the rule was right.

**Native digits are folded to ASCII.**
    The course ruling is that digits stay ("digits are fine but do remove non
    language words"). Nothing says they must stay in their native form, and
    leaving both forms means 2024 and the Devanagari spelling of the same year
    occupy separate vocabulary entries. Folding is configurable and counted.

**Paragraph structure survives.**
    Whitespace is collapsed, but blank lines are not, because paragraph-level
    deduplication and the line-structure quality heuristics both need
    paragraph boundaries to still exist.

Source convention: every invisible or ambiguous codepoint below is built with
``chr()`` from its numeric value. Literal zero-width and control characters
would be invisible in review, unreviewable in a diff, and in the case of the
C0 controls would make this file unparseable.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Iterator

from ftfy import TextFixerConfig, fix_text

from common.io import Doc, StageStats

__all__ = ["NormalizeConfig", "Normalizer", "normalize_stage"]


# --- codepoints -------------------------------------------------------------

ZWNJ = chr(0x200C)
ZWJ = chr(0x200D)
VIRAMA_DEVANAGARI = chr(0x094D)
VIRAMA_BENGALI = chr(0x09CD)

DEVANAGARI_DIGITS = "".join(chr(c) for c in range(0x0966, 0x0970))
BENGALI_DIGITS = "".join(chr(c) for c in range(0x09E6, 0x09F0))
ASCII_DIGITS = "0123456789"

_DEV_DIGIT_MAP = str.maketrans(DEVANAGARI_DIGITS, ASCII_DIGITS)
_BEN_DIGIT_MAP = str.maketrans(BENGALI_DIGITS, ASCII_DIGITS)

_VIRAMAS = VIRAMA_DEVANAGARI + VIRAMA_BENGALI
_JOINERS = ZWNJ + ZWJ

# Stray joiners: any ZWJ/ZWNJ NOT immediately preceded by a virama.
_STRAY_JOINER_RE = re.compile(f"(?<![{_VIRAMAS}])[{_JOINERS}]")
_ANY_JOINER_RE = re.compile(f"[{_JOINERS}]")
_ORTHO_JOINER_RE = re.compile(f"[{_VIRAMAS}][{_JOINERS}]")

# Control characters (Cc) except tab and newline, plus invisible format (Cf)
# characters that carry no meaning in running text. ZWJ/ZWNJ are deliberately
# absent -- they are handled positionally above.
_JUNK_CODEPOINTS = (
    list(range(0x00, 0x09))          # NUL..BS   (keeps TAB 0x09)
    + [0x0B, 0x0C]                   # VT, FF    (keeps LF 0x0A)
    + list(range(0x0E, 0x20))        # SO..US    (keeps CR 0x0D, folded below)
    + list(range(0x7F, 0xA0))        # DEL and the C1 controls
    + [0x00AD]                       # soft hyphen
    + [0x200B, 0x200E, 0x200F]       # zero-width space, LTR/RTL marks
    + list(range(0x202A, 0x202F))    # bidi embedding and override
    + list(range(0x2060, 0x2065))    # word joiner, invisible operators
    + [0xFEFF]                       # BOM / zero-width no-break space
)
_JUNK_TABLE = {c: None for c in _JUNK_CODEPOINTS}

# Unicode spaces that should behave as a plain space.
_SPACE_CODEPOINTS = (
    [0x00A0, 0x1680]
    + list(range(0x2000, 0x200B))
    + [0x202F, 0x205F, 0x3000]
)
_SPACE_TABLE = {c: " " for c in _SPACE_CODEPOINTS}

_HORIZ_WS_RE = re.compile(r"[ \t]+")
_BLANKLINE_RE = re.compile(r"\n{3,}")

# ftfy defaults do Latin-centric work we do not want. Encoding repair and
# control-character removal are the useful parts; ligature folding and width
# folding are compatibility operations that would undo the NFC-not-NFKC choice.
_FTFY = TextFixerConfig(
    unescape_html=True,
    remove_terminal_escapes=True,
    fix_encoding=True,
    fix_latin_ligatures=False,
    fix_character_width=False,
    uncurl_quotes=False,
    fix_line_breaks=True,
    fix_surrogates=True,
    remove_control_chars=True,
    normalization="NFC",
)


# --- configuration ----------------------------------------------------------


@dataclass(frozen=True)
class NormalizeConfig:
    """Switches for stage 1. Defaults are the choices argued for above.

    Attributes:
        fix_mojibake: Run ftfy encoding repair. Disable only if it is ever
            observed damaging correct Indic text.
        strip_stray_joiners: Remove ZWJ/ZWNJ that are not after a virama.
        strip_orthographic_joiners: Also remove the post-virama joiners. Off by
            default: those change rendering. Turn on only with a measured
            reason, and say so in the report.
        digits_to_ascii: Fold Devanagari/Bengali digits to ASCII.
        min_chars: Documents shorter than this after normalisation are dropped
            as empty. Deliberately tiny -- real length filtering is stage 4's
            job, and stage 1 should not silently make length decisions.
    """

    fix_mojibake: bool = True
    strip_stray_joiners: bool = True
    strip_orthographic_joiners: bool = False
    digits_to_ascii: bool = True
    min_chars: int = 1


# --- normaliser -------------------------------------------------------------


@dataclass
class Normalizer:
    """Applies stage-1 normalisation and records what it changed.

    The counters exist so the report can quantify each transformation rather
    than assert it happened.
    """

    config: NormalizeConfig = field(default_factory=NormalizeConfig)
    counts: Counter = field(default_factory=Counter)

    def __call__(self, text: str) -> str:
        """Normalise one string. Pure: same input always gives same output."""
        if not text:
            return ""

        original = text

        # 1. Encoding repair. Also applies NFC as configured above.
        if self.config.fix_mojibake:
            fixed = fix_text(text, config=_FTFY)
            if fixed != text:
                self.counts["mojibake_or_html_fixed"] += 1
            text = fixed

        # 2. Canonical composition, stated explicitly rather than relying on
        #    ftfy's setting. Idempotent, so the duplication is free.
        composed = unicodedata.normalize("NFC", text)
        if composed != text:
            self.counts["nfc_changed"] += 1
        text = composed

        # 3. Invisible junk.
        stripped = text.translate(_JUNK_TABLE)
        if len(stripped) != len(text):
            self.counts["control_chars_removed"] += len(text) - len(stripped)
        text = stripped

        # 4. Joiners, positionally.
        n_ortho = len(_ORTHO_JOINER_RE.findall(text))
        if self.config.strip_orthographic_joiners:
            text, n = _ANY_JOINER_RE.subn("", text)
            if n:
                self.counts["joiners_removed_all"] += n
        elif self.config.strip_stray_joiners:
            text, n = _STRAY_JOINER_RE.subn("", text)
            if n:
                self.counts["stray_joiners_removed"] += n
            if n_ortho:
                self.counts["orthographic_joiners_kept"] += n_ortho

        # 5. Digits.
        if self.config.digits_to_ascii:
            if any(c in DEVANAGARI_DIGITS for c in text):
                text = text.translate(_DEV_DIGIT_MAP)
                self.counts["devanagari_digits_folded"] += 1
            if any(c in BENGALI_DIGITS for c in text):
                text = text.translate(_BEN_DIGIT_MAP)
                self.counts["bengali_digits_folded"] += 1

        # 6. Whitespace -- collapsed, but paragraph breaks preserved.
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = text.translate(_SPACE_TABLE)
        text = _HORIZ_WS_RE.sub(" ", text)
        text = "\n".join(line.strip() for line in text.split("\n"))
        text = _BLANKLINE_RE.sub("\n\n", text).strip()

        if text != original:
            self.counts["docs_changed"] += 1
        return text

    def normalize_doc(self, doc: Doc) -> Doc:
        """Return a copy of ``doc`` with normalised text."""
        return Doc(
            id=doc.id,
            text=self(doc.text),
            source=doc.source,
            collection_method=doc.collection_method,
            url=doc.url,
            retrieved=doc.retrieved,
            licence=doc.licence,
            meta=doc.meta,
        )

    def report(self) -> dict:
        """Counter contents, for the statistics report."""
        return dict(self.counts.most_common())


def normalize_stage(
    config: NormalizeConfig | None = None,
    stats: StageStats | None = None,
    normalizer: Normalizer | None = None,
) -> Callable[[Iterator[Doc]], Iterator[Doc]]:
    """Build a stage function for :func:`common.io.map_shards`.

    Documents that normalise to nothing are dropped and counted, since an
    empty document would otherwise reach the tokenizer as a zero-length
    training example.

    Args:
        config: Normalisation switches.
        stats: Accumulator for drop reasons.
        normalizer: Reuse an existing normaliser so its counters accumulate
            across shards. One is created if omitted.

    Returns:
        A function suitable for ``map_shards(..., fn=...)``.
    """
    norm = normalizer or Normalizer(config or NormalizeConfig())

    def _stage(docs: Iterator[Doc]) -> Iterator[Doc]:
        for doc in docs:
            out = norm.normalize_doc(doc)
            if len(out.text) < norm.config.min_chars:
                if stats is not None:
                    stats.dropped("empty_after_normalize")
                continue
            yield out

    return _stage


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Run with:  python -m common.clean.normalize
    from common.io import enable_utf8_console

    enable_utf8_console()
    norm = Normalizer()
    failures = []

    def check(name: str, got, want) -> None:
        if got == want:
            print(f"OK   {name}")
        else:
            failures.append(name)
            print(f"FAIL {name}\n       got  {got!r}\n       want {want!r}")

    NBSP = chr(0x00A0)
    ZWSP = chr(0x200B)

    # -- Assamese discriminators must survive untouched. The entire
    #    Assamese/Bengali language filter depends on these two codepoints.
    check("Assamese ra U+09F0 preserved", norm(chr(0x09F0)), chr(0x09F0))
    check("Assamese wa U+09F1 preserved", norm(chr(0x09F1)), chr(0x09F1))
    check("Assamese word aru intact", norm("আৰু"), "আৰু")

    # -- NFC decomposes precomposed nukta forms (composition exclusions),
    #    giving one representation per sound.
    check(
        "Devanagari qa U+0958 -> ka + nukta",
        norm(chr(0x0958)),
        chr(0x0915) + chr(0x093C),
    )
    check(
        "Bengali rra U+09DC -> dda + nukta",
        norm(chr(0x09DC)),
        chr(0x09A1) + chr(0x09BC),
    )
    check(
        "already-decomposed nukta unchanged",
        norm(chr(0x0915) + chr(0x093C)),
        chr(0x0915) + chr(0x093C),
    )

    # -- Joiners: strays go, orthographic ones stay.
    check("stray ZWNJ removed", norm("भारत" + ZWNJ), "भारत")
    check(
        "post-virama ZWNJ kept",
        norm("क" + VIRAMA_DEVANAGARI + ZWNJ + "ष"),
        "क" + VIRAMA_DEVANAGARI + ZWNJ + "ष",
    )
    aggressive = Normalizer(NormalizeConfig(strip_orthographic_joiners=True))
    check(
        "post-virama ZWNJ removed when configured",
        aggressive("क" + VIRAMA_DEVANAGARI + ZWNJ + "ष"),
        "क" + VIRAMA_DEVANAGARI + "ष",
    )

    # -- Digits fold; configurable.
    check("Devanagari digits -> ASCII", norm(DEVANAGARI_DIGITS), ASCII_DIGITS)
    check("Bengali digits -> ASCII", norm(BENGALI_DIGITS), ASCII_DIGITS)
    keep = Normalizer(NormalizeConfig(digits_to_ascii=False))
    check("digits kept when configured", keep(DEVANAGARI_DIGITS), DEVANAGARI_DIGITS)

    # -- Whitespace collapses; paragraphs survive.
    check("spaces collapsed", norm("भारत    एक"), "भारत एक")
    check("NBSP -> space", norm("भारत" + NBSP + "एक"), "भारत एक")
    check("paragraph break preserved", norm("पहला।\n\n\n\nदूसरा।"), "पहला।\n\nदूसरा।")
    check("zero-width space removed", norm("भा" + ZWSP + "रत"), "भारत")
    check("danda preserved", norm("यह है।"), "यह है।")

    # -- Empty documents are dropped by the stage, not silently emitted.
    st = StageStats(stage="normalize")
    stage = normalize_stage(stats=st)
    docs = [
        Doc(id="a", text="भारत एक देश है।", source="t", collection_method="manual"),
        Doc(id="b", text="  " + ZWSP + "  ", source="t", collection_method="manual"),
    ]
    kept = list(stage(iter(docs)))
    check("empty doc dropped", (len(kept), st.drops["empty_after_normalize"]), (1, 1))

    print("\ncounters:", norm.report())
    if failures:
        raise SystemExit(f"\n{len(failures)} check(s) failed: {failures}")
    print("\nOK  all normalisation checks passed")
