"""Boilerplate removal and heuristic quality filtering.

Stage 4, and it must run **before** non-language word removal (§4.3). That
ordering was wrong in the original plan and measurement caught it: running the
non-language cascade first destroyed 64% of Hindi news documents, because the
words it was removing were ``IST``, ``Aug``, ``pm`` and ``am`` -- datelines and
bylines, not code-mixing. A short article carries enough furniture to push it
past the document-loss threshold on boilerplate alone. The cascade is meant to
measure code-mixing, so it must only ever see article prose.

Two parts, in order:

**Boilerplate stripping** removes lines rather than documents. News pages carry
datelines, "Also Read" teasers, share prompts and follow-us appeals; scanned
books carry library stamps and page furniture. These repeat across thousands of
documents, so leaving them in would also dominate paragraph-level dedup later.

**Heuristic filters** are the rules from Lecture 3 slide 15: length bounds,
symbol-to-letter ratio, line structure, intra-document repetition, and PII.
Every threshold is set by reading what it removes, in the language (slide 30),
and every rejection is recorded with a reason so ``inspect_rejects`` can print
samples per filter.

**The line-structure filter earns its place here specifically.** Slide 15
describes it as catching menus; in this corpus it catches *dictionaries*. The
largest single "books" in both Wikisource corpora are lexicons -- Hindi's
संस्कृत-हिन्दी कोश at 509k words, Assamese's হেমকোষ at 434k -- and they are word
lists, not prose. Training on them would teach the model that text is a sequence
of headword-gloss pairs.

**Thresholds are asymmetric between languages**, as in §4.3. Hindi filters hard
because it has a surplus and precision is worth more than recall; Assamese
filters gently because every document discarded is data that cannot be
replaced.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Iterator

import regex

from common.io import Doc, StageStats

__all__ = [
    "QualityConfig", "HINDI_QUALITY", "ASSAMESE_QUALITY",
    "strip_boilerplate", "QualityFilter", "quality_stage",
]

# --- boilerplate ------------------------------------------------------------

# Datelines and timestamps. Measured as the single largest source of spurious
# "non-language words" in Hindi news: IST appeared 22,805 times and Aug 21,022
# times across 4,000 articles.
_MONTHS = ("Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec")
_BOILERPLATE_LINE_RES = [
    # "Aug 13, 2026 12:16 pm IST" and variants, alone on a line
    regex.compile(rf"^\W*(?:{_MONTHS})\w*\s+\d{{1,2}},?\s*\d{{4}}.*$", regex.I),
    regex.compile(r"^\W*\d{1,2}:\d{2}\s*(?:am|pm)?\s*(?:IST|GMT|UTC)?\W*$", regex.I),
    regex.compile(r"^\W*(?:IST|GMT|UTC)\W*$", regex.I),
    # English UI furniture that survives extraction
    regex.compile(r"^\W*(?:Also Read|Read More|Read Also|New Update|Share this|"
                  r"Follow us|Subscribe|Advertisement|Trending|Top Stories|"
                  r"Published|Updated|Last Updated|By\s+\w+)\b.*$", regex.I),
    # Library and publisher stamps in scanned books
    regex.compile(r"^\W*(?:All rights? reserved|OSMANIA UNIVERSITY|Digitized by|"
                  r"Google|Printed (?:at|by)|Published by)\b.*$", regex.I),
]

# Datelines are not always on their own line. Live Hindustan appends one to
# every headline inside listing pages: "...सात टुकड़ियों ने दी सलामी Aug 14,
# 2026 01:34 pm IST". A line-anchored pattern misses those entirely, which is
# why the first version of this module failed to change the drop rate at all.
# This one matches anywhere, and swallows the trailing timezone even when the
# extractor has glued the next word onto it ("ISTAbhinav").
_INLINE_DATELINE_RE = regex.compile(
    rf"(?:{_MONTHS})[a-z]*\.?\s+\d{{1,2}},?\s*\d{{4}}"      # Aug 14, 2026
    rf"(?:\s+\d{{1,2}}:\d{{2}}\s*(?:am|pm)?)?"              # 01:34 pm
    rf"\s*(?:IST|GMT|UTC)?",                                # IST
    regex.I,
)
# A page carrying several inline datelines is an index of headlines, not an
# article. Extracting it yields a list of titles with no prose.
_LISTING_DATELINE_THRESHOLD = 3

# PII. Anonymised rather than deleted, so sentence structure survives.
_EMAIL_RE = regex.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE_RE = regex.compile(r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)")
_URL_RE = regex.compile(r"https?://\S+|www\.\S+")

# Script content is letters AND combining marks. Using \p{L} alone is wrong for
# Indic text and wrong in a way that looks plausible: भारत is भ + ा + र + त,
# where ा is a mark, not a letter. Counting only letters scored ordinary Hindi
# prose at 0.55 alphabetic content and rejected it as symbol soup. This is the
# same trap as the \w+ tokenisation bug in common/clean/filters.py -- combining marks
# are part of the writing system, and every character-class test has to say so.
_ALPHA_RE = regex.compile(r"[\p{L}\p{M}]")
_SYMBOL_RE = regex.compile(r"[^\p{L}\p{M}\p{N}\s]")


def is_listing_page(text: str) -> bool:
    """True if this looks like an index of headlines rather than an article."""
    return len(_INLINE_DATELINE_RE.findall(text)) >= _LISTING_DATELINE_THRESHOLD


def strip_boilerplate(text: str) -> tuple[str, int]:
    """Drop boilerplate, strip inline datelines and anonymise PII.

    Line-level removal handles furniture that sits alone; the inline dateline
    substitution handles the far more common case of a timestamp glued to the
    end of a real sentence.

    Returns:
        ``(cleaned_text, items_removed)``.
    """
    kept: list[str] = []
    removed = 0
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            kept.append("")
            continue
        if any(r.match(stripped) for r in _BOILERPLATE_LINE_RES):
            removed += 1
            continue
        kept.append(stripped)

    out = "\n".join(kept)
    out, n_inline = _INLINE_DATELINE_RE.subn(" ", out)
    removed += n_inline
    out = _URL_RE.sub(" ", out)
    out = _EMAIL_RE.sub("[EMAIL]", out)
    out = _PHONE_RE.sub("[PHONE]", out)
    # Collapse the blank runs and doubled spaces left behind.
    out = regex.sub(r"[ \t]{2,}", " ", out)
    out = regex.sub(r"\n{3,}", "\n\n", out)
    return out.strip(), removed


# --- quality ----------------------------------------------------------------


@dataclass(frozen=True)
class QualityConfig:
    """Heuristic thresholds (slide 15), tuned per language (slide 30).

    Attributes:
        min_words: Documents shorter than this carry too little language.
        max_words: Absurdly long documents are usually concatenated dumps.
        max_symbol_ratio: Symbols against letters; catches garbled OCR and
            tables.
        min_alpha_ratio: Share of characters that are letters at all.
        min_mean_line_words: Mean words per line. **The dictionary filter** --
            a lexicon averages two or three words per line where prose averages
            fifteen.
        max_short_line_fraction: Share of lines under ``short_line_words``.
        short_line_words: What counts as a short line.
        max_line_repetition: Share of lines that are exact duplicates of
            another line in the same document.
        max_top_ngram_share: Share of 5-grams taken by the single most common
            one; catches a paragraph repeated throughout.
    """

    min_words: int
    max_words: int
    max_symbol_ratio: float
    min_alpha_ratio: float
    min_mean_line_words: float
    max_short_line_fraction: float
    short_line_words: int
    max_line_repetition: float
    max_top_ngram_share: float


# Hindi: abundance. Filter hard, favour precision.
HINDI_QUALITY = QualityConfig(
    min_words=40,
    max_words=200_000,
    max_symbol_ratio=0.20,
    min_alpha_ratio=0.55,
    min_mean_line_words=5.0,
    max_short_line_fraction=0.60,
    short_line_words=4,
    max_line_repetition=0.30,
    max_top_ngram_share=0.05,
)

# Assamese: scarcity. Every filter costs data that cannot be replaced, so each
# threshold is loosened and the reason recorded. This is the slide 34 answer.
ASSAMESE_QUALITY = QualityConfig(
    min_words=25,               # keep short news items
    max_words=400_000,
    max_symbol_ratio=0.30,      # OCR noise tolerated rather than discarded
    min_alpha_ratio=0.45,
    min_mean_line_words=3.5,    # still catches lexicons, keeps verse
    max_short_line_fraction=0.75,
    short_line_words=3,
    max_line_repetition=0.45,
    max_top_ngram_share=0.08,
)


@dataclass
class QualityFilter:
    """Apply the heuristics, recording why each document was rejected."""

    config: QualityConfig
    rejects: Counter = field(default_factory=Counter)
    samples: dict = field(default_factory=dict)
    keep_samples: int = 3

    def _record(self, reason: str, doc: Doc | None, text: str) -> str:
        self.rejects[reason] += 1
        bucket = self.samples.setdefault(reason, [])
        if len(bucket) < self.keep_samples:
            bucket.append({
                "id": getattr(doc, "id", "?"),
                "title": (getattr(doc, "meta", {}) or {}).get("title", ""),
                "excerpt": text[:300],
            })
        return reason

    def check(self, text: str, doc: Doc | None = None, raw: str | None = None) -> str | None:
        """Return a rejection reason, or None if the document passes.

        Args:
            text: The cleaned text to judge.
            doc: Source document, for reject samples.
            raw: The text *before* boilerplate stripping. Listing pages are
                detected here, because stripping removes the very datelines
                that identify them.
        """
        if raw is not None and is_listing_page(raw):
            return self._record("listing_page", doc, raw)

        words = text.split()
        n = len(words)
        if n < self.config.min_words:
            return self._record("too_short", doc, text)
        if n > self.config.max_words:
            return self._record("too_long", doc, text)

        # Symbol ratio is checked first because symbol-soup text necessarily
        # also has a low alphabetic ratio, and "symbol_heavy" is the more
        # informative reason to show when reading rejects. low_alpha_ratio is
        # then left to catch what is neither letters nor symbols -- documents
        # that are mostly digits, or mostly whitespace.
        alpha = len(_ALPHA_RE.findall(text))
        symbols = len(_SYMBOL_RE.findall(text))
        if symbols / max(alpha, 1) > self.config.max_symbol_ratio:
            return self._record("symbol_heavy", doc, text)
        if alpha / max(len(text), 1) < self.config.min_alpha_ratio:
            return self._record("low_alpha_ratio", doc, text)

        lines = [ln for ln in text.split("\n") if ln.strip()]
        if lines:
            mean_line = sum(len(ln.split()) for ln in lines) / len(lines)
            if mean_line < self.config.min_mean_line_words:
                # This is what removes dictionaries and lexicons.
                return self._record("line_structure_wordlist", doc, text)
            short = sum(1 for ln in lines
                        if len(ln.split()) <= self.config.short_line_words)
            if short / len(lines) > self.config.max_short_line_fraction:
                return self._record("mostly_short_lines", doc, text)
            dupes = len(lines) - len(set(lines))
            if dupes / len(lines) > self.config.max_line_repetition:
                return self._record("repeated_lines", doc, text)

        if n >= 100:
            grams = Counter(tuple(words[i:i + 5]) for i in range(n - 4))
            if grams:
                top = grams.most_common(1)[0][1]
                if top / max(n - 4, 1) > self.config.max_top_ngram_share:
                    return self._record("ngram_repetition", doc, text)
        return None

    def report(self) -> dict:
        """Rejection counts and sample excerpts, for the statistics report."""
        return {
            "rejects": dict(self.rejects.most_common()),
            "total_rejected": sum(self.rejects.values()),
            "samples": self.samples,
        }


def quality_stage(
    qfilter: QualityFilter,
    stats: StageStats | None = None,
    do_boilerplate: bool = True,
) -> Callable[[Iterator[Doc]], Iterator[Doc]]:
    """Build a :func:`common.io.map_shards` stage: boilerplate, then heuristics."""

    def _stage(docs: Iterator[Doc]) -> Iterator[Doc]:
        for doc in docs:
            raw = doc.text
            text = raw
            if do_boilerplate:
                text, _removed = strip_boilerplate(text)
            reason = qfilter.check(text, doc, raw=raw if do_boilerplate else None)
            if reason is not None:
                if stats is not None:
                    stats.dropped(f"quality:{reason}")
                continue
            doc.text = text
            yield doc

    return _stage


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Run with:  python -m common.clean.quality
    from common.io import enable_utf8_console

    enable_utf8_console()
    failures = []

    def check(name, got, want):
        if got == want:
            print(f"OK   {name}  -> {got}")
        else:
            failures.append(name)
            print(f"FAIL {name}\n       got  {got!r}\n       want {want!r}")

    # Boilerplate: the exact patterns that broke the Hindi news cascade.
    news = ("Aug 13, 2026 12:16 pm IST\n"
            "भारत सरकार ने आज एक नई योजना की घोषणा की है।\n"
            "Also Read: Top stories of the day\n"
            "इस योजना से लाखों लोगों को लाभ मिलेगा।\n"
            "Follow us on social media\n")
    cleaned, removed = strip_boilerplate(news)
    check("boilerplate lines removed", removed, 3)
    check("prose survives", "भारत सरकार" in cleaned and "Also Read" not in cleaned, True)

    check("email anonymised", "[EMAIL]" in strip_boilerplate("संपर्क: a.b@c.com")[0], True)
    check("phone anonymised", "[PHONE]" in strip_boilerplate("फोन 9876543210 पर")[0], True)

    q = QualityFilter(HINDI_QUALITY)

    prose = ("भारत एक विशाल देश है और यहाँ अनेक भाषाएँ बोली जाती हैं। " * 12 + "\n" +
             "दिल्ली इसकी राजधानी है और यह शहर बहुत पुराना है। " * 12)
    check("good prose passes", q.check(prose), None)

    # A dictionary: many very short lines of ordinary Devanagari. This is the
    # filter that removes संस्कृत-हिन्दी कोश and হেমকোষ, and it has to fire on
    # line structure rather than on any character-level signal, because a
    # lexicon's characters look exactly like prose.
    entries = ["अग्नि आग ज्वाला", "जल पानी नीर", "वायु हवा पवन",
               "पृथ्वी धरती भूमि", "आकाश गगन नभ", "सूर्य रवि भानु"]
    lexicon = "\n".join(entries[i % len(entries)] for i in range(300))
    check("dictionary rejected", q.check(lexicon), "line_structure_wordlist")

    check("too short rejected", q.check("बहुत छोटा दस्तावेज़ है यह।"), "too_short")
    check("symbol soup rejected",
          q.check("॥ ॥ ॥ #### $$$$ %%%% &&&& @@@@ !!!! ???? ==== ++++ " * 20),
          "symbol_heavy")

    repeated = "यह वाक्य बार बार दोहराया गया है और कुछ नहीं है। " * 60
    check("ngram repetition rejected", q.check(repeated), "ngram_repetition")

    # Asymmetry: a short Assamese news item that Hindi thresholds would drop.
    short_as = ("অসমৰ গুৱাহাটীত আজি এক নতুন আঁচনিৰ শুভাৰম্ভ কৰা হ'ল। "
                "এই আঁচনিৰ জৰিয়তে বহুতো লোকে উপকৃত হ'ব বুলি আশা কৰা হৈছে। "
                "ৰাজ্য চৰকাৰে ইয়াৰ বাবে বিশেষ ব্যৱস্থা গ্ৰহণ কৰিছে।")
    hi_verdict = QualityFilter(HINDI_QUALITY).check(short_as)
    as_verdict = QualityFilter(ASSAMESE_QUALITY).check(short_as)
    print(f"     same text: hindi-thresholds={hi_verdict}  assamese-thresholds={as_verdict}")
    check("loosened Assamese keeps a short item Hindi drops",
          (hi_verdict is not None, as_verdict), (True, None))

    if failures:
        raise SystemExit(f"\n{len(failures)} check(s) failed: {failures}")
    print("\nOK  all quality checks passed")
