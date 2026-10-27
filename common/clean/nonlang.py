"""Remove non-language words, per the course ruling.

The ruling: *"Digits are fine but do remove non language words."* So टाटा
stays, Tata goes, 2024 stays.

**Order matters.** This runs *after* language identification, never before.
Stripping Latin words first inflates a document's in-script ratio and lets a
mostly-English page slip through the script gate -- the filter would be
sabotaging the filter upstream of it.

**Word surgery, not document rejection.** Dropping every article containing
"IRCTC" would cost enormous amounts of Hindi and is simply unaffordable for
Assamese. Instead a cascade, each level catching what the one below cannot:

1. **Word** -- a token whose letters are mostly Latin is removed. Digits,
   punctuation and in-script words survive untouched.
2. **Sentence** -- if a sentence lost too large a share of its words it is
   dropped whole. What remains after heavy stripping is not a sentence with a
   few loanwords missing; it is disfluent wreckage that would teach the model
   broken syntax.
3. **Document** -- if a document lost too much, it was never really
   monolingual and goes entirely.

**The thresholds are asymmetric between the two languages, deliberately.**
Lecture 3 slide 30: with abundant text, filter hard and favour precision;
with a scarce corpus, "every filter you add costs you data you cannot
replace... repair what you can rather than discarding it." Hindi therefore
uses tight thresholds and Assamese loosened ones, and the report should say
so -- slide 34 asks explicitly for "one filtering decision I made differently
for the lower-resource language."

**Everything removed is counted.** The frequency table of stripped tokens is
one table in the report, it evidences the filter working, and it is the
"read what your filter removes" exercise from slide 36.

**Consequence to state in the report:** words removed here never reach the
tokenizer, so the Phase 2 model can never generate "IRCTC". That is the
intended effect of the ruling, and it is better noted now than discovered
during Phase 2 generation.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Iterator

import regex

from common.io import Doc, StageStats

__all__ = ["NonLanguageConfig", "HINDI_CONFIG", "ASSAMESE_CONFIG",
           "NonLanguageWordFilter", "nonlang_stage"]

SCRIPT_RANGES = {
    "devanagari": (0x0900, 0x097F),
    "bengali": (0x0980, 0x09FF),
}

_TOKEN_RE = regex.compile(r"\S+")
_LETTER_RE = regex.compile(r"\p{L}")
_LATIN_RE = regex.compile(r"\p{Script=Latin}")
# Indic sentence enders: danda, double danda, plus Western punctuation, which
# appears throughout modern Hindi and Assamese prose.
_SENT_SPLIT_RE = regex.compile(r"(?<=[।॥.!?])\s+")


@dataclass(frozen=True)
class NonLanguageConfig:
    """Cascade thresholds.

    Attributes:
        script: Which Unicode block counts as "the language".
        sentence_loss: Drop a sentence that lost more than this share of its
            words.
        document_loss: Drop a document that lost more than this share.
        min_sentence_words: Sentences shorter than this are exempt from the
            ratio test -- a two-word heading that loses one word is at 50%
            without being damaged.
        keep_acronym_len: Latin runs this short are kept, so measurement units
            and single letters inside otherwise-Indic text survive.
    """

    script: str
    sentence_loss: float
    document_loss: float
    min_sentence_words: int = 4
    keep_acronym_len: int = 0


# Hindi has a surplus: filter hard, favour precision (slide 30).
HINDI_CONFIG = NonLanguageConfig(
    script="devanagari", sentence_loss=0.30, document_loss=0.20
)

# Assamese has no surplus. Both thresholds are loosened, and short Latin runs
# are kept, because borrowed terms are more frequent in Assamese news and the
# data cannot be replaced. This is the asymmetry slide 34 asks us to report.
ASSAMESE_CONFIG = NonLanguageConfig(
    script="bengali", sentence_loss=0.50, document_loss=0.40, keep_acronym_len=2
)


@dataclass
class NonLanguageWordFilter:
    """Apply the word → sentence → document cascade.

    Attributes:
        config: Thresholds for this language.
        removed: Frequency table of stripped tokens, for the report.
        counts: Cascade-level tallies.
    """

    config: NonLanguageConfig
    removed: Counter = field(default_factory=Counter)
    counts: Counter = field(default_factory=Counter)

    def _is_non_language(self, token: str) -> bool:
        """True if a whitespace token is a foreign-script word.

        Tokens with no letters at all -- numbers, punctuation, dates -- are
        never non-language, which is how the ruling's "digits are fine"
        survives contact with the implementation.
        """
        letters = _LETTER_RE.findall(token)
        if not letters:
            return False
        lo, hi = SCRIPT_RANGES[self.config.script]
        in_script = sum(1 for ch in letters if lo <= ord(ch) <= hi)
        latin = len(_LATIN_RE.findall(token))
        if latin <= self.config.keep_acronym_len and in_script == 0 and latin > 0:
            return False  # short Latin run, kept by configuration
        return latin > in_script

    def _filter_sentence(self, sentence: str) -> str | None:
        """Strip foreign words; return None if the sentence should be dropped."""
        tokens = _TOKEN_RE.findall(sentence)
        if not tokens:
            return None
        kept = []
        dropped = 0
        for tok in tokens:
            if self._is_non_language(tok):
                self.removed[tok.strip(".,;:!?()[]\"'")] += 1
                dropped += 1
            else:
                kept.append(tok)
        if not kept:
            self.counts["sentences_dropped"] += 1
            return None
        if len(tokens) >= self.config.min_sentence_words:
            if dropped / len(tokens) > self.config.sentence_loss:
                self.counts["sentences_dropped"] += 1
                return None
        self.counts["words_removed"] += dropped
        return " ".join(kept)

    def apply(self, text: str) -> tuple[str, dict]:
        """Run the cascade over one document.

        Returns:
            ``(cleaned_text, info)``. ``cleaned_text`` is empty when the
            document-level threshold rejected it.
        """
        words_in = len(text.split())
        out_paragraphs = []
        for para in text.split("\n\n"):
            sentences = _SENT_SPLIT_RE.split(para)
            kept = [s for s in (self._filter_sentence(x) for x in sentences) if s]
            if kept:
                out_paragraphs.append(" ".join(kept))

        cleaned = "\n\n".join(out_paragraphs).strip()
        words_out = len(cleaned.split())
        loss = 1.0 - (words_out / words_in) if words_in else 1.0
        info = {
            "words_in": words_in,
            "words_out": words_out,
            "loss": round(loss, 4),
        }
        if not cleaned or loss > self.config.document_loss:
            self.counts["documents_dropped"] += 1
            info["verdict"] = "document_dropped"
            return "", info
        info["verdict"] = "kept"
        return cleaned, info

    def report(self, top: int = 50) -> dict:
        """Summary for the statistics report."""
        return {
            "counts": dict(self.counts),
            "distinct_tokens_removed": len(self.removed),
            "total_tokens_removed": sum(self.removed.values()),
            "top_removed": dict(self.removed.most_common(top)),
        }


def nonlang_stage(
    nfilter: NonLanguageWordFilter,
    stats: StageStats | None = None,
) -> Callable[[Iterator[Doc]], Iterator[Doc]]:
    """Build a :func:`common.io.map_shards` stage from the cascade."""

    def _stage(docs: Iterator[Doc]) -> Iterator[Doc]:
        for doc in docs:
            cleaned, info = nfilter.apply(doc.text)
            if not cleaned:
                if stats is not None:
                    stats.dropped("nonlang:document_dropped")
                continue
            doc.text = cleaned
            yield doc

    return _stage


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Run with:  python -m common.clean.nonlang
    from common.io import enable_utf8_console

    enable_utf8_console()
    failures = []

    def check(name, got, want):
        if got == want:
            print(f"OK   {name}")
        else:
            failures.append(name)
            print(f"FAIL {name}\n       got  {got!r}\n       want {want!r}")

    f = NonLanguageWordFilter(HINDI_CONFIG)

    # The ruling's own examples.
    check("Latin word removed, Indic kept",
          f._filter_sentence("टाटा कंपनी Tata Motors ने कहा है कि"),
          "टाटा कंपनी ने कहा है कि")
    check("digits survive",
          f._filter_sentence("वर्ष 2024 में लगभग 140 करोड़ लोग रहते हैं यहाँ"),
          "वर्ष 2024 में लगभग 140 करोड़ लोग रहते हैं यहाँ")
    check("acronym removed for Hindi",
          f._filter_sentence("IRCTC ने नई सुविधा शुरू की है अब यात्री"),
          "ने नई सुविधा शुरू की है अब यात्री")

    # Code-mixed sentence loses too much and goes entirely.
    check("code-mixed sentence dropped",
          f._filter_sentence("The company said that its quarterly revenue बढ़ा"),
          None)

    # Whole-document behaviour.
    hindi_doc = ("भारत एक विशाल देश है। यहाँ 2024 में अनेक भाषाएँ बोली जाती हैं।\n\n"
                 "दिल्ली इसकी राजधानी है। यह शहर बहुत पुराना और ऐतिहासिक है।")
    cleaned, info = NonLanguageWordFilter(HINDI_CONFIG).apply(hindi_doc)
    check("clean Hindi document untouched", (info["verdict"], info["loss"]), ("kept", 0.0))

    english_doc = "This document is entirely in English and should not survive at all."
    _, info2 = NonLanguageWordFilter(HINDI_CONFIG).apply(english_doc)
    check("English document dropped", info2["verdict"], "document_dropped")

    # Asymmetry, the point of slide 30: identical text, two postures.
    # 9 tokens, 5 of them Latin. Strict loses 5/9 = 56% and drops the
    # sentence; loosened Assamese keeps "IT" as a short acronym, so it loses
    # 4/9 = 44%, under its 50% threshold, and the sentence survives.
    mixed_as = "অসমৰ Guwahati চহৰত IT park আৰু software company হৈছে"
    strict = NonLanguageWordFilter(NonLanguageConfig("bengali", 0.30, 0.20))
    loose = NonLanguageWordFilter(ASSAMESE_CONFIG)
    s_out = strict._filter_sentence(mixed_as)
    l_out = loose._filter_sentence(mixed_as)
    print(f"     strict(0.30, no acronyms) -> {s_out!r}")
    print(f"     loose (0.50, acronyms<=2) -> {l_out!r}")
    check("loosened Assamese keeps a sentence strict discards",
          (s_out is None, l_out is not None), (True, True))

    f2 = NonLanguageWordFilter(HINDI_CONFIG)
    f2._filter_sentence("टाटा कंपनी Tata Motors ने कहा है कि")
    f2._filter_sentence("IRCTC ने Tata को नई सुविधा दी है अब")
    print(f"     removal table: {f2.report(5)['top_removed']}")
    check("removals are logged", f2.report()["total_tokens_removed"] >= 4, True)

    if failures:
        raise SystemExit(f"\n{len(failures)} check(s) failed: {failures}")
    print("\nOK  all non-language word removal checks passed")
