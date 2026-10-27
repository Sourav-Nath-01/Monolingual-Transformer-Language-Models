"""Language identification for script-sharing Indic languages.

Off-the-shelf language detectors are the weak point for both of this
project's languages, for the reason Lecture 3 slide 27 gives: Hindi, Marathi,
Nepali and Bhojpuri all share Devanagari, and detectors confuse them, so the
corpus quietly fills with a neighbouring language. Assamese has the same
problem against Bengali, a language with roughly twenty times more speakers
and vastly more web text.

Identification runs in two stages.

**Stage 1 -- script gate.**
    Fraction of *letters* in the target Unicode block. Digits and punctuation
    are excluded from the denominator, so an article full of numbers is not
    penalised for them. This kills romanised text and English pages cheaply,
    before any per-language work.

**Stage 2 -- discrimination among script sharers.**
    Function-word profiles plus, for Assamese, an orthographic signal that is
    close to decisive.

The Assamese case is the one worth understanding, because it is nearly free
and very reliable. Assamese and Bengali share a Unicode block but not an
alphabet:

    Assamese writes /r/ as U+09F0 RA WITH MIDDLE DIAGONAL
    Bengali  writes /r/ as U+09B0 RA

/r/ is among the most frequent consonants in both languages, so any document
of reasonable length contains many of them, and they are *mutually exclusive*
in correct orthography. A document with hundreds of U+09B0 and no U+09F0 is
Bengali, whatever else it contains. That single ratio does most of the work;
function words are used to break ties and to catch documents too short for
the ratio to be stable.

Every classifier returns its evidence, not just a label, because thresholds
here have to be set by reading what they reject (slide 30) and because the
report needs measured precision and recall rather than an assertion that the
filter works.
"""

from __future__ import annotations

import regex
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Iterator

from common.io import Doc, StageStats

__all__ = [
    "script_ratio",
    "LangDecision",
    "AssameseBengaliClassifier",
    "HindiDevanagariClassifier",
    "language_filter_stage",
    "evaluate",
]

# --- scripts ----------------------------------------------------------------

SCRIPT_RANGES: dict[str, tuple[int, int]] = {
    "devanagari": (0x0900, 0x097F),
    "bengali": (0x0980, 0x09FF),  # the Bengali-Assamese block
}

# The orthographic discriminators. Written as codepoints because U+09F0 and
# U+09B0 are near-identical on screen and the whole filter turns on telling
# them apart -- a reviewer must be able to see which is which.
ASSAMESE_RA = chr(0x09F0)  # ৰ
BENGALI_RA = chr(0x09B0)  # র
ASSAMESE_WA = chr(0x09F1)  # ৱ
BENGALI_BA = chr(0x09AC)  # ব

# Function words. Chosen to be frequent and, as far as possible, exclusive.
# Note that some obvious candidates are shared -- এই ("this") occurs in both
# Assamese and Bengali -- so shared words are deliberately absent.
ASSAMESE_MARKERS = (
    "আৰু", "হৈছে", "কৰি", "বুলি", "তেওঁ", "নাই",
    "পৰা", "লগত", "বাবে", "কৰা", "হোৱা", "গৈ",
)
BENGALI_MARKERS = (
    "এবং", "করে", "হয়েছে", "তিনি", "থেকে", "জন্য",
    "সঙ্গে", "তার", "করেছেন", "হয়ে", "গিয়ে", "নয়",
)

HINDI_MARKERS = (
    "है", "हैं", "का", "की", "के", "को", "में", "से",
    "और", "नहीं", "यह", "लिए", "पर", "हुआ",
)
MARATHI_MARKERS = (
    "आहे", "चा", "ची", "चे", "मध्ये", "आणि",
    "या", "ला", "नाही", "होते", "त्या",
)
NEPALI_MARKERS = (
    "छ", "छन्", "मा", "भएको", "गर्ने", "हुन्",
    "लागि", "तर", "गरेको", "हो",
)
BHOJPURI_MARKERS = ("बा", "बाड़", "रउआ", "कइल", "भइल", "हऽ")
SANSKRIT_MARKERS = ("इति", "अस्ति", "भवति", "एव", "तु", "यत्", "सः")

# U+0933 LETTER LLA is standard in Marathi and essentially absent from Hindi,
# so it works as a character-level signal the way U+09F0 does for Assamese.
MARATHI_LLA = chr(0x0933)  # ळ

# Word tokenisation for Indic scripts.
#
# The obvious pattern, r"[^\W\d_]+", is WRONG here and silently so. Devanagari
# and Bengali vowel signs, anusvara and virama are combining marks (Unicode
# categories Mn/Mc), not letters, so that pattern treats them as separators:
# आहे splits into आ + ह, and परंपरा splits into पर + पर. The first destroys
# every multi-syllable function word before it can be matched; the second
# fabricates matches that were never in the text -- in testing it produced two
# false "पर" hits in a Marathi document and pushed it over the Hindi
# threshold. A word count came out at 113 for a 60-word sentence.
#
# \p{L}\p{M} keeps letters together with the marks that belong to them, which
# is why the `regex` module is a dependency rather than stdlib `re`.
_WORD_RE = regex.compile(r"[\p{L}\p{M}]+")


def script_ratio(text: str, script: str) -> float:
    """Fraction of alphabetic characters lying in the target Unicode block.

    Args:
        text: Document text.
        script: Key into :data:`SCRIPT_RANGES`.

    Returns:
        0.0 to 1.0. Text with no letters at all returns 0.0.
    """
    lo, hi = SCRIPT_RANGES[script]
    total = in_script = 0
    for ch in text:
        if ch.isalpha():
            total += 1
            if lo <= ord(ch) <= hi:
                in_script += 1
    return in_script / total if total else 0.0


def _marker_rate(words: list[str], markers: tuple[str, ...]) -> float:
    """Share of tokens that are one of ``markers``."""
    if not words:
        return 0.0
    marker_set = set(markers)
    return sum(1 for w in words if w in marker_set) / len(words)


@dataclass
class LangDecision:
    """Outcome of one classification.

    Attributes:
        label: Predicted language, or ``"unknown"`` when evidence is too thin.
        score: Signed confidence. Positive favours the target language.
        evidence: Raw measurements, kept so thresholds can be re-tuned from
            logged rejects without re-running classification.
    """

    label: str
    score: float
    evidence: dict = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return self.label != "unknown"


# --- Assamese vs Bengali ----------------------------------------------------


@dataclass
class AssameseBengaliClassifier:
    """Separate Assamese from Bengali within the shared Unicode block.

    Args:
        min_script_ratio: Script-gate threshold. Loosened relative to Hindi
            because Assamese data is scarce and borrowed Latin terms are more
            frequent -- discarding a borderline Assamese document costs data
            that cannot be replaced.
        ra_margin: How decisively Assamese RA must outnumber Bengali RA.
            0.0 means a simple majority.
        min_ra_total: Below this many /r/ characters the ratio is noise, so
            the decision falls back to function words.
    """

    min_script_ratio: float = 0.60
    ra_margin: float = 0.0
    min_ra_total: int = 8

    def classify(self, text: str) -> LangDecision:
        """Label ``text`` as ``assamese``, ``bengali`` or ``unknown``."""
        sr = script_ratio(text, "bengali")
        ev: dict = {"script_ratio": round(sr, 4)}
        if sr < self.min_script_ratio:
            return LangDecision("unknown", -1.0, ev | {"reason": "script_gate"})

        n_as_ra = text.count(ASSAMESE_RA)
        n_bn_ra = text.count(BENGALI_RA)
        n_as_wa = text.count(ASSAMESE_WA)
        total_ra = n_as_ra + n_bn_ra

        words = _WORD_RE.findall(text)
        as_rate = _marker_rate(words, ASSAMESE_MARKERS)
        bn_rate = _marker_rate(words, BENGALI_MARKERS)

        ev.update(
            assamese_ra=n_as_ra, bengali_ra=n_bn_ra, assamese_wa=n_as_wa,
            assamese_marker_rate=round(as_rate, 5),
            bengali_marker_rate=round(bn_rate, 5),
            words=len(words),
        )

        if total_ra >= self.min_ra_total:
            # The orthographic signal, mapped to [-1, 1].
            ra_score = (n_as_ra - n_bn_ra) / total_ra
            ev["ra_score"] = round(ra_score, 4)
            # Function words agree or disagree; they adjust, never override,
            # because the character signal is far more reliable.
            word_score = as_rate - bn_rate
            score = ra_score + 2.0 * word_score
            label = "assamese" if score > self.ra_margin else "bengali"
            return LangDecision(label, round(score, 4), ev)

        # Too few /r/ characters for the ratio to mean anything.
        score = (as_rate - bn_rate) * 10.0
        ev["reason"] = "short_text_function_words"
        if abs(score) < 0.01 and n_as_ra == 0 and n_as_wa == 0:
            return LangDecision("unknown", round(score, 4), ev)
        return LangDecision(
            "assamese" if score > 0 else "bengali", round(score, 4), ev
        )


# --- Hindi vs its Devanagari neighbours -------------------------------------


@dataclass
class HindiDevanagariClassifier:
    """Separate Hindi from Marathi, Nepali, Bhojpuri and Sanskrit.

    There is no single-character signal as clean as Assamese RA, so this
    leans on function-word rates, with Marathi's LLA as a supporting cue.

    Args:
        min_script_ratio: Script-gate threshold. Tighter than the Assamese
            classifier, because Hindi has a surplus and precision is worth
            more than recall (Lecture 3 slide 30).
        min_words: Documents shorter than this cannot support a rate estimate.
        min_margin: Hindi's marker rate must exceed the best competitor's by
            this much.
    """

    min_script_ratio: float = 0.70
    min_words: int = 25
    min_margin: float = 0.005

    COMPETITORS = {
        "marathi": MARATHI_MARKERS,
        "nepali": NEPALI_MARKERS,
        "bhojpuri": BHOJPURI_MARKERS,
        "sanskrit": SANSKRIT_MARKERS,
    }

    def classify(self, text: str) -> LangDecision:
        """Label ``text`` as ``hindi``, a competitor language, or ``unknown``."""
        sr = script_ratio(text, "devanagari")
        ev: dict = {"script_ratio": round(sr, 4)}
        if sr < self.min_script_ratio:
            return LangDecision("unknown", -1.0, ev | {"reason": "script_gate"})

        words = _WORD_RE.findall(text)
        if len(words) < self.min_words:
            return LangDecision("unknown", 0.0, ev | {"reason": "too_short"})

        hi_rate = _marker_rate(words, HINDI_MARKERS)
        rates = {name: _marker_rate(words, m) for name, m in self.COMPETITORS.items()}
        # Marathi LLA is nearly absent from Hindi; treat its presence as
        # additional Marathi evidence proportional to how often it occurs.
        lla_rate = text.count(MARATHI_LLA) / max(len(text), 1)
        rates["marathi"] += lla_rate * 5.0

        best_rival, best_rate = max(rates.items(), key=lambda kv: kv[1])
        margin = hi_rate - best_rate
        ev.update(
            hindi_marker_rate=round(hi_rate, 5),
            competitor_rates={k: round(v, 5) for k, v in rates.items()},
            marathi_lla_rate=round(lla_rate, 6),
            words=len(words),
        )

        if margin >= self.min_margin:
            return LangDecision("hindi", round(margin, 5), ev)
        if best_rate > hi_rate:
            return LangDecision(best_rival, round(margin, 5), ev)
        return LangDecision("unknown", round(margin, 5), ev | {"reason": "no_margin"})


# --- paragraph-level repair -------------------------------------------------


@dataclass
class ParagraphFilter:
    """Keep the target-language paragraphs of a mixed-language document.

    Document-level rejection is the right call for a web page, which is one
    thing in one language. It is the wrong call for a scanned book, which
    routinely opens with an English title page, a publisher's imprint and a
    library stamp before several hundred pages of the target language. Judging
    such a book as a whole throws away the body to punish the front matter.

    Measured cost of getting this wrong: dropping whole documents removed 1.35M
    words from Assamese Wikisource -- 12.5% of it -- including a book whose
    first line is "(All right reserved) The Assamese language is the Easternmost
    member of that family" followed by a full Assamese text. For the language
    with no surplus, that is unaffordable, and Lecture 3 slide 30 is explicit
    that a low-resource corpus should be repaired rather than discarded.

    Short paragraphs are judged on script alone. A three-word line carries too
    little evidence for function-word rates, and headings and chapter numbers
    are legitimately short -- classifying them as foreign would punch holes in
    otherwise clean text.

    Attributes:
        classifier: Anything with ``classify(text) -> LangDecision``.
        keep_label: Language to keep.
        script: Key into :data:`SCRIPT_RANGES`, used for short paragraphs.
        min_chars_to_classify: Below this, fall back to the script test.
        short_script_ratio: Script threshold for those short paragraphs.
        min_keep_fraction: If less than this share of characters survives, the
            document was not really in the target language and is dropped
            whole.
    """

    classifier: object
    keep_label: str
    script: str
    min_chars_to_classify: int = 120
    short_script_ratio: float = 0.50
    min_keep_fraction: float = 0.25

    def apply(self, text: str) -> tuple[str, dict]:
        """Filter ``text`` paragraph by paragraph.

        Returns:
            ``(kept_text, info)``. ``kept_text`` is empty when too little
            survived. ``info`` records what happened, for the statistics
            report and for reading rejects.
        """
        paragraphs = [p for p in text.split("\n\n")]
        kept: list[str] = []
        info = {
            "paragraphs": len(paragraphs),
            "kept": 0,
            "dropped_lang": 0,
            "dropped_script": 0,
            "chars_in": len(text),
            "chars_kept": 0,
        }

        for para in paragraphs:
            stripped = para.strip()
            if not stripped:
                continue
            if len(stripped) < self.min_chars_to_classify:
                # Too short to classify; judge on script alone.
                if script_ratio(stripped, self.script) >= self.short_script_ratio:
                    kept.append(stripped)
                    info["kept"] += 1
                    info["chars_kept"] += len(stripped)
                else:
                    info["dropped_script"] += 1
                continue

            if self.classifier.classify(stripped).label == self.keep_label:
                kept.append(stripped)
                info["kept"] += 1
                info["chars_kept"] += len(stripped)
            else:
                info["dropped_lang"] += 1

        info["keep_fraction"] = (
            info["chars_kept"] / info["chars_in"] if info["chars_in"] else 0.0
        )
        if info["keep_fraction"] < self.min_keep_fraction:
            info["verdict"] = "document_dropped"
            return "", info
        info["verdict"] = "kept"
        return "\n\n".join(kept), info


def paragraph_filter_stage(
    pfilter: ParagraphFilter,
    stats: StageStats | None = None,
) -> Callable[[Iterator[Doc]], Iterator[Doc]]:
    """Build a :func:`common.io.map_shards` stage from a :class:`ParagraphFilter`.

    Use this for book-like sources -- Wikisource and scanned books -- where a
    document is long and may mix languages. Use
    :func:`language_filter_stage` for web pages, which are short and
    single-language.
    """

    def _stage(docs: Iterator[Doc]) -> Iterator[Doc]:
        for doc in docs:
            kept, info = pfilter.apply(doc.text)
            if not kept:
                if stats is not None:
                    stats.dropped("paragraph_filter:document_dropped")
                continue
            if stats is not None and info["dropped_lang"] + info["dropped_script"]:
                stats.dropped("paragraph_filter:paragraphs_removed")
            doc.text = kept
            doc.meta = dict(doc.meta or {})
            doc.meta["para_keep_fraction"] = round(info["keep_fraction"], 3)
            yield doc

    return _stage


# --- pipeline stage ---------------------------------------------------------


def language_filter_stage(
    classifier,
    keep_label: str,
    stats: StageStats | None = None,
    record_evidence: bool = True,
) -> Callable[[Iterator[Doc]], Iterator[Doc]]:
    """Build a :func:`common.io.map_shards` stage that keeps one language.

    Args:
        classifier: Anything with ``classify(text) -> LangDecision``.
        keep_label: The label to keep; every other outcome is dropped.
        stats: Accumulator; drop reasons are recorded as ``lang:<label>``.
        record_evidence: Store the decision on the surviving document's
            ``meta``, so later stages and the report can inspect borderline
            cases without reclassifying.

    Returns:
        A stage function.
    """

    def _stage(docs: Iterator[Doc]) -> Iterator[Doc]:
        for doc in docs:
            decision = classifier.classify(doc.text)
            if decision.label != keep_label:
                if stats is not None:
                    reason = decision.evidence.get("reason", decision.label)
                    stats.dropped(f"lang:{reason}")
                continue
            if record_evidence:
                doc.meta = dict(doc.meta or {})
                doc.meta["lang_score"] = decision.score
            yield doc

    return _stage


# --- evaluation -------------------------------------------------------------


def evaluate(classifier, labelled: list[tuple[str, str]], positive: str) -> dict:
    """Precision, recall and F1 against hand-labelled documents.

    This is the gradeable artifact for the Assamese/Bengali filter: the report
    needs a measured number, not a claim that the discriminator works.

    Args:
        classifier: Anything with ``classify(text) -> LangDecision``.
        labelled: ``(text, gold_label)`` pairs.
        positive: Which label counts as the positive class.

    Returns:
        Counts, precision, recall, F1 and a full confusion matrix.
    """
    tp = fp = fn = tn = 0
    confusion: Counter = Counter()
    for text, gold in labelled:
        pred = classifier.classify(text).label
        confusion[(gold, pred)] += 1
        if pred == positive and gold == positive:
            tp += 1
        elif pred == positive and gold != positive:
            fp += 1
        elif pred != positive and gold == positive:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "n": len(labelled),
        "positive_class": positive,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round((tp + tn) / len(labelled), 4) if labelled else 0.0,
        "confusion": {f"gold={g},pred={p}": c for (g, p), c in confusion.most_common()},
    }


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Run with:  python -m common.clean.filters
    from common.io import enable_utf8_console

    enable_utf8_console()
    failures = []

    def check(name, got, want):
        if got == want:
            print(f"OK   {name}  -> {got}")
        else:
            failures.append(name)
            print(f"FAIL {name}\n       got  {got!r}\n       want {want!r}")

    ab = AssameseBengaliClassifier()

    # Real Assamese: note ৰ throughout, and আৰু.
    assamese = (
        "অসম ভাৰতৰ উত্তৰ-পূব দিশত অৱস্থিত এখন ৰাজ্য। ইয়াৰ ৰাজধানী দিছপুৰ। "
        "অসমৰ সংস্কৃতি আৰু ঐতিহ্য অতি চহকী। ব্ৰহ্মপুত্ৰ নদী অসমৰ মাজেৰে বৈ গৈছে "
        "আৰু ই ৰাজ্যখনৰ কৃষিৰ বাবে অতি গুৰুত্বপূৰ্ণ। অসমীয়া ভাষা ইয়াৰ ৰাজ্যিক ভাষা।"
    )
    # Real Bengali: র throughout, এবং / থেকে / তার.
    bengali = (
        "পশ্চিমবঙ্গ ভারতের পূর্ব দিকে অবস্থিত একটি রাজ্য। এর রাজধানী কলকাতা। "
        "বাংলার সংস্কৃতি এবং ঐতিহ্য অত্যন্ত সমৃদ্ধ। গঙ্গা নদী রাজ্যের মধ্য দিয়ে "
        "বয়ে গিয়েছে এবং তার জল কৃষির জন্য খুবই গুরুত্বপূর্ণ। বাংলা ভাষা এখানকার রাজ্য ভাষা।"
    )
    check("assamese classified", ab.classify(assamese).label, "assamese")
    check("bengali classified", ab.classify(bengali).label, "bengali")
    check("english rejected by script gate",
          ab.classify("This is a plain English sentence about Assam.").label, "unknown")

    d_as = ab.classify(assamese).evidence
    d_bn = ab.classify(bengali).evidence
    print(f"     assamese evidence: ra={d_as['assamese_ra']}/{d_bn['bengali_ra']}"
          f" ra_score={d_as.get('ra_score')}")
    print(f"     bengali  evidence: as_ra={d_bn['assamese_ra']} bn_ra={d_bn['bengali_ra']}"
          f" ra_score={d_bn.get('ra_score')}")

    hd = HindiDevanagariClassifier()
    hindi = (
        "भारत दक्षिण एशिया में स्थित एक देश है। इसकी राजधानी नई दिल्ली है। "
        "भारत की संस्कृति और परंपरा बहुत समृद्ध है। गंगा नदी देश के मध्य से होकर "
        "बहती है और इसका जल कृषि के लिए बहुत महत्वपूर्ण है। हिन्दी यहाँ की प्रमुख भाषा है "
        "और इसे बड़ी संख्या में लोग बोलते हैं। यह देश अनेक भाषाओं का घर है।"
    )
    marathi = (
        "महाराष्ट्र हे भारताच्या पश्चिम भागात असलेले एक राज्य आहे. याची राजधानी मुंबई आहे. "
        "महाराष्ट्राची संस्कृती आणि परंपरा खूप समृद्ध आहे. गोदावरी नदी या राज्याच्या "
        "मध्यातून वाहते आणि तिचे पाणी शेतीसाठी महत्त्वाचे आहे. मराठी ही येथील प्रमुख भाषा आहे "
        "आणि ती अनेक लोक बोलतात. या राज्यात अनेक किल्ले आहेत."
    )
    check("hindi classified", hd.classify(hindi).label, "hindi")
    check("marathi not accepted as hindi", hd.classify(marathi).label != "hindi", True)
    print(f"     marathi -> {hd.classify(marathi).label}, "
          f"lla_rate={hd.classify(marathi).evidence['marathi_lla_rate']}")

    ev = evaluate(ab, [(assamese, "assamese"), (bengali, "bengali")], "assamese")
    check("evaluate harness", (ev["tp"], ev["tn"], ev["precision"]), (1, 1, 1.0))

    if failures:
        raise SystemExit(f"\n{len(failures)} check(s) failed: {failures}")
    print("\nOK  all language-filter checks passed")
