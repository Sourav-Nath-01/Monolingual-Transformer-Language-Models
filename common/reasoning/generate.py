"""Generate the synthetic comparative-reasoning datasets for Phase 3.

Spec section 3.1 requires a dataset built *programmatically* -- not downloaded --
so the ground-truth label is known by construction, in the target language's own
script and natural phrasing, with train/val/test splits and a defensible story
about train-test leakage.

Three design decisions worth stating, because each is a trade-off:

**Latin digits, not Devanagari or Bengali-Assamese numerals.** Sampling 400
documents from each cleaned corpus gives 51,196 Latin digits and *zero* native
numerals in Hindi, and 24,128 against zero in Assamese. The pretrained models
therefore never saw ``२५`` or ``২৫``, and the tokenizers never learned them as
units. Writing ``25`` keeps the finetuning distribution inside the one the model
was actually pretrained on.

**Comparatives that do not inflect for gender.** Hindi adjectives agree with
their subject (``लंबा`` / ``लंबी``), so a template that fills a name into an
adjectival slot produces grammatical errors half the time. Phrasing every
comparison through an invariant quantity word -- ``ज़्यादा`` / ``कम`` in Hindi,
``বেছি`` / ``কম`` in Assamese -- attached to the *attribute* rather than the
person sidesteps agreement entirely and stays natural.

**Leakage blocked by held-out relation patterns, not held-out entities.** The
spec offers either mechanism. An earlier version of this file used both, giving
each split a disjoint name pool -- and that turned out to make the task
unlearnable at this model scale rather than merely harder. Because the answer is
always one of the entities named in the prompt, disjoint pools trained the model
to emit one set of ~33 tokens and then scored it on ~25 tokens it had never been
trained to emit. Training loss fell to 0.75 while validation loss rose from 4.2
to 15.4: it had learned the answer *vocabulary*, not the comparison. Entities are
therefore shared across train/val/test, and leakage is blocked by reserving one
template per task type for the test split, so test probes phrasings never
trained on.

**A separate generalisation probe.** ``test_unseen`` uses entities held out of
training entirely. It is not the headline metric -- it measures something
different and harder: whether the model learned to *select from the prompt* or
merely which tokens are plausible answers. Reporting it beside the main test
accuracy makes that distinction visible instead of hiding it.

Output, one JSONL file per language per split -- ``train``, ``val``, ``test``
and ``test_unseen``::

    <lang>/train/reasoning_<split>.jsonl

Each line is ``{"id", "task", "template", "prompt", "answer", "meta"}``. The
model is trained on ``prompt + " " + answer``; evaluation generates from
``prompt`` and exact-matches against ``answer``.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["ReasoningConfig", "generate_split", "generate_language", "main"]


# ===========================================================================
# LANGUAGE CONTENT -- the part a native speaker should read
# ===========================================================================
#
# Each language provides:
#   names / objects          shared entity pools, used by train/val/test
#   names_unseen / objects_*  held out of training, for the test_unseen probe
#   attrs                     noun, unit, and every agreement form it governs
#   more / less               invariant comparative words
#   templates per task        the one suffixed _test is reserved for test

HINDI = {
    # ONE shared entity pool across train/val/test. An earlier version gave each
    # split its own disjoint names, which looked like a stronger leakage control
    # and was in fact fatal: the answer is always an entity, so the model was
    # trained to emit one of ~33 tokens and then scored on ~25 it had never been
    # trained to emit. Validation loss went 4.2 -> 15.4 while training loss fell
    # to 0.75 -- it had learned the answer vocabulary, not the comparison.
    # Leakage is instead blocked by held-out relation patterns, which is the
    # other mechanism the spec names. `names_unseen` / `objects_unseen` are kept
    # aside for the separate generalisation probe in the test_unseen split.
    "names": ["राम", "अमित", "सुरेश", "विकास", "राहुल", "मोहन", "संजय", "दीपक",
              "अनिल", "प्रकाश", "विजय", "अजय", "रवि", "मनोज", "सुनील", "आशीष",
              "नितिन", "गौरव", "पंकज", "हरीश", "कमल", "नरेश", "जितेन्द्र", "भरत",
              "शेखर", "अरुण", "सुधीर", "विनोद", "राजेश", "महेश", "दिनेश", "अशोक",
              "माधव", "तरुण", "यशवंत", "ओमकार", "धीरज", "श्रीकांत", "नागेश",
              "वसंत", "हेमंत", "उदय", "चंदन", "मुकेश", "जगदीश", "रमेश", "सतीश",
              "नरेन्द्र", "किशोर", "मुरली"],
    "names_unseen": ["बालकृष्ण", "त्रिलोक", "शंकर", "गोपाल", "वासुदेव", "इंद्रजीत",
                     "पुरुषोत्तम", "देवदत्त"],
    # Feminine or consonant-final only: Hindi masculine -आ nouns would need an
    # oblique form before a postposition (जूता -> जूते की).
    "objects": ["किताब", "कलम", "कुर्सी", "मेज़", "बैग", "घड़ी", "साइकिल",
                "बोतल", "कमीज़", "पेंसिल", "टोपी", "छतरी", "चाबी", "बाल्टी",
                "सुई", "रस्सी", "कैंची", "प्लेट", "चादर", "कंघी", "थाली",
                "तस्वीर", "अलमारी", "बत्ती"],
    "objects_unseen": ["डोरी", "सीटी", "छलनी", "तराजू", "बाँसुरी", "पतंग"],
    # noun, unit, possessive particle (agrees with the NOUN, not the owner),
    # which entity pool it applies to, and a plausible value range.
    "attrs": [
        {"noun": "उम्र", "obl": "की", "part": "पाई गई", "fut": "होगी",   "unit": "साल",        "poss": "की", "q": "किसकी", "kind": "person", "range": (6, 85)},
        {"noun": "लंबाई", "obl": "की", "part": "पाई गई", "fut": "होगी",  "unit": "सेंटीमीटर",  "poss": "की", "q": "किसकी", "kind": "person", "range": (140, 195)},
        {"noun": "वज़न", "obl": "के", "part": "पाया गया", "fut": "होगा",   "unit": "किलो",       "poss": "का", "q": "किसका", "kind": "person", "range": (35, 110)},
        {"noun": "कीमत", "obl": "की", "part": "पाई गई", "fut": "होगी",   "unit": "रुपये",      "poss": "की", "q": "किसकी", "kind": "object", "range": (20, 4000)},
    ],
    "group": {"person": "तीन लोगों", "object": "तीन चीज़ों"},
    "more": "ज़्यादा",
    "less": "कम",
    "therefore": "इसलिए",
    "order_word": "क्रम",
}

ASSAMESE = {
    # See the Hindi note above: one shared pool, held-out templates as the
    # leakage control, and a reserved pool for the generalisation probe.
    "names": ["ৰাম", "অমিত", "সুৰেশ", "বিকাশ", "ৰাহুল", "মোহন", "সঞ্জয়", "দীপক",
              "অনিল", "প্ৰকাশ", "ৰাজীৱ", "ভাস্কৰ", "জিতেন", "ৰঞ্জিত", "হীৰেন",
              "বিপুল", "পংকজ", "ৰূপম", "অনুপম", "ধ্ৰুৱ", "কমল", "নৰেশ", "ভৰত",
              "শেখৰ", "অৰুণ", "নৱজ্যোতি", "সুধীৰ", "বিনোদ", "ৰাজেশ", "মহেশ",
              "দিনেশ", "অশোক", "মাধৱ", "তৰুণ", "যশৱন্ত", "ওঁকাৰ", "ধীৰজ",
              "শ্ৰীকান্ত", "নাগেন", "বসন্ত", "হেমন্ত", "উদয়", "চন্দন", "মুকেশ",
              "জগদীশ", "ৰমেশ", "সতীশ", "নৰেন", "কিশোৰ", "মুৰলী"],
    "names_unseen": ["বলৰাম", "ত্ৰিলোক", "শংকৰ", "গোপাল", "বাসুদেৱ", "ইন্দ্ৰজিৎ",
                     "পুৰুষোত্তম", "দেৱদত্ত"],
    "objects": ["কিতাপ", "কলম", "চকী", "মেজ", "বেগ", "ঘড়ী", "চাইকেল", "জোতা",
                "বটল", "চছমা", "টুপী", "ছাতি", "মোনা", "বালটি", "বেজী", "ৰচী",
                "কেঁচী", "প্লেট", "বালিচ", "চাদৰ", "কাঁইট", "ফটো", "আলমাৰি",
                "বাটি"],
    "objects_unseen": ["চাবি", "কাপ", "চালনী", "দোন", "বাঁহী", "ঘুৰণী"],
    # Assamese genitive -ৰ does not inflect, so "poss" is empty; kept for a
    # single code path across both languages.
    "attrs": [
        {"noun": "বয়স", "obl": "", "part": "", "fut": "",   "unit": "বছৰ",          "poss": "", "q": "কাৰ", "kind": "person", "range": (6, 85)},
        {"noun": "উচ্চতা", "obl": "", "part": "", "fut": "", "unit": "ছেণ্টিমিটাৰ",  "poss": "", "q": "কাৰ", "kind": "person", "range": (140, 195)},
        {"noun": "ওজন", "obl": "", "part": "", "fut": "",    "unit": "কিলো",         "poss": "", "q": "কাৰ", "kind": "person", "range": (35, 110)},
        # কাৰ is the animate "whose"; things take কোনটোৰ, "which one's".
        {"noun": "দাম", "obl": "", "part": "", "fut": "",    "unit": "টকা",          "poss": "", "q": "কোনটোৰ", "kind": "object", "range": (20, 4000)},
    ],
    # Assamese classifiers are animacy-sensitive: জন for people, টা for things.
    "group": {"person": "তিনিজন মানুহৰ", "object": "তিনিটা বস্তুৰ"},
    "more": "বেছি",
    "less": "কম",
    "therefore": "গতিকে",
    "order_word": "ক্ৰম",
}

# ---- Hindi templates -------------------------------------------------------
# {a},{b},{c} names · {attr} attribute noun · {u} unit · {x},{y},{z} values
# {cmp} the invariant comparative word

HI_COMPARE2 = [
    ("hi_c2_a", "{a} {poss} {attr} {x} {u} है। {b} {poss} {attr} {y} {u} है। "
                "दोनों में से {q} {attr} {cmp} है?"),
    ("hi_c2_b", "{a} {poss} {attr} {x} {u} है और {b} {poss} {attr} {y} {u} है। "
                "बताइए, {cmp} {attr} {q} है?"),
    ("hi_c2_test", "एक सूची के अनुसार {a} {poss} {attr} {x} {u} और {b} {poss} {attr} "
                   "{y} {u} दर्ज है। इनमें {cmp} {attr} {q} है?"),
]

HI_ORDER3 = [
    ("hi_o3_a", "{a} {poss} {attr} {x} {u} है। {b} {poss} {attr} {y} {u} है। "
                "{c} {poss} {attr} {z} {u} है। तीनों में सबसे {cmp} {attr} {q} है?"),
    ("hi_o3_b", "{group} {poss} {attr} इस प्रकार है: {a} {poss} {x} {u}, "
                "{b} {poss} {y} {u}, और {c} {poss} {z} {u}। "
                "इनमें सबसे {cmp} {attr} {q} है?"),
    ("hi_o3_test", "{a}, {b} और {c} {poss} {attr} क्रमशः {x} {u}, {y} {u} और {z} {u} है। "
                   "सबसे {cmp} {attr} {q} है?"),
]

HI_TRANSITIVE = [
    ("hi_tr_a", "{a} {poss} {attr} {b} {obl} {attr} से {cmp} है। "
                "{b} {poss} {attr} {c} {obl} {attr} से {cmp} है। "
                "तो {a} और {c} में से {q} {attr} {qcmp} है?"),
    ("hi_tr_b", "अगर {a} {poss} {attr} {b} से {cmp} है, और {b} {poss} {attr} {c} से "
                "{cmp} है, तो {a} और {c} में {q} {attr} {qcmp} {fut}?"),
    ("hi_tr_test", "{a} {poss} {attr} {b} से {cmp} {part} और {b} {poss} {attr} {c} से "
                   "{cmp} {part}। इस आधार पर {a} और {c} में {qcmp} {attr} {q} है?"),
]

# ---- Assamese templates ----------------------------------------------------

AS_COMPARE2 = [
    ("as_c2_a", "{a}ৰ {attr} {x} {u}। {b}ৰ {attr} {y} {u}। "
                "দুয়োৰে ভিতৰত {q} {attr} {cmp}?"),
    ("as_c2_b", "{a}ৰ {attr} {x} {u} আৰু {b}ৰ {attr} {y} {u}। "
                "কোৱাচোন, {q} {attr} {cmp}?"),
    ("as_c2_test", "এখন তালিকা অনুসৰি {a}ৰ {attr} {x} {u} আৰু {b}ৰ {attr} {y} {u} "
                   "লিপিবদ্ধ আছে। ইয়াৰ ভিতৰত {q} {attr} {cmp}?"),
]

AS_ORDER3 = [
    ("as_o3_a", "{a}ৰ {attr} {x} {u}। {b}ৰ {attr} {y} {u}। {c}ৰ {attr} {z} {u}। "
                "তিনিওৰে ভিতৰত {q} {attr} আটাইতকৈ {cmp}?"),
    ("as_o3_b", "{group} {attr} এনেধৰণৰ: {a}ৰ {x} {u}, {b}ৰ {y} {u}, "
                "আৰু {c}ৰ {z} {u}। ইয়াৰ ভিতৰত {q} {attr} আটাইতকৈ {cmp}?"),
    ("as_o3_test", "{a}, {b} আৰু {c}ৰ {attr} ক্ৰমে {x} {u}, {y} {u} আৰু {z} {u}। "
                   "{q} {attr} আটাইতকৈ {cmp}?"),
]

AS_TRANSITIVE = [
    ("as_tr_a", "{a}ৰ {attr} {b}ৰ {attr}তকৈ {cmp}। "
                "{b}ৰ {attr} {c}ৰ {attr}তকৈ {cmp}। "
                "তেন্তে {a} আৰু {c}ৰ ভিতৰত {q} {attr} {qcmp}?"),
    ("as_tr_b", "যদি {a}ৰ {attr} {b}তকৈ {cmp} হয়, আৰু {b}ৰ {attr} {c}তকৈ {cmp} হয়, "
                "তেন্তে {a} আৰু {c}ৰ ভিতৰত {q} {attr} {qcmp} হ'ব?"),
    ("as_tr_test", "{a}ৰ {attr} {b}তকৈ {cmp} পোৱা গ'ল আৰু {b}ৰ {attr} {c}তকৈ {cmp} "
                   "পোৱা গ'ল। ইয়াৰ ভিত্তিত {a} আৰু {c}ৰ ভিতৰত {q} {attr} {qcmp}?"),
]

ANSWER_PREFIX = {"hindi": "उत्तर:", "assamese": "উত্তৰ:"}

LANGS = {
    "hindi": dict(HINDI, compare2=HI_COMPARE2, order3=HI_ORDER3,
                  transitive=HI_TRANSITIVE),
    "assamese": dict(ASSAMESE, compare2=AS_COMPARE2, order3=AS_ORDER3,
                     transitive=AS_TRANSITIVE),
}

# ===========================================================================


@dataclass
class ReasoningConfig:
    """How many examples to build, and with what mix.

    Attributes:
        language: ``"hindi"`` or ``"assamese"``.
        n_train / n_val / n_test: Examples per split.
        task_mix: Share of each task type; must sum to 1.
        value_range: Inclusive range the numeric quantities are drawn from.
        min_gap: Smallest allowed difference between two values, so a question
            is never ambiguous at the surface level.
        seed: Makes generation reproducible.
        out_dir: Defaults to ``<language>/train``.
    """

    language: str
    n_train: int = 40000
    n_val: int = 2000
    n_test: int = 2000
    n_test_unseen: int = 1000
    task_mix: dict = field(default_factory=lambda: {
        "compare2": 0.4, "order3": 0.3, "transitive": 0.3})
    value_range: tuple[int, int] = (5, 200)
    min_gap: int = 2
    seed: int = 20252
    out_dir: str = ""

    def __post_init__(self) -> None:
        self.out_dir = self.out_dir or f"{self.language}/train"
        total = sum(self.task_mix.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"task_mix must sum to 1, got {total}")


def _distinct_values(rng: random.Random, n: int, lo: int, hi: int,
                     min_gap: int) -> list[int]:
    """Draw n values that are pairwise at least ``min_gap`` apart.

    Without the gap, a question like "whose age is greater, 40 or 41?" is
    technically well-posed but tests reading a digit rather than comparing, and
    a tokenizer that splits numbers unevenly can make it accidentally hard.
    """
    while True:
        vals = [rng.randint(lo, hi) for _ in range(n)]
        s = sorted(vals)
        if all(b - a >= min_gap for a, b in zip(s, s[1:])):
            return vals


def _make_example(rng: random.Random, lang: str, task: str, split: str,
                  templates: list[tuple[str, str]], cfg: ReasoningConfig) -> dict:
    """Build one example of the given task type, with its known-correct answer.

    The entity pool follows the attribute: ages and heights belong to people,
    prices belong to objects. Asking for "the price of three people" would be
    grammatical and meaningless, which is worse than an obvious error because
    the model would learn the pattern anyway.
    """
    L = LANGS[lang]
    attr = rng.choice(L["attrs"])
    base = "names" if attr["kind"] == "person" else "objects"
    # test_unseen draws from entities never used in training -- a probe for
    # whether the model learned to select from the prompt or to memorise which
    # tokens are plausible answers.
    pool = L[base + "_unseen"] if split == "test_unseen" else L[base]
    want_more = rng.random() < 0.5
    cmp_word = L["more"] if want_more else L["less"]
    tid, tpl = rng.choice(templates)
    lo, hi = attr["range"]
    fixed = dict(attr=attr["noun"], u=attr["unit"], poss=attr["poss"],
                 q=attr["q"], obl=attr["obl"], part=attr["part"], fut=attr["fut"],
                 group=L["group"][attr["kind"]], cmp=cmp_word)

    if task == "compare2":
        a, b = rng.sample(pool, 2)
        x, y = _distinct_values(rng, 2, lo, hi, cfg.min_gap)
        answer = a if (x > y) == want_more else b
        prompt = tpl.format(a=a, b=b, x=x, y=y, **fixed)
        # Naming the decisive value first gives the model an intermediate step:
        # copying the relevant number is easier than selecting the entity in one
        # jump, and the entity then follows from the number it chose.
        chosen = max(x, y) if want_more else min(x, y)
        rationale = f"{chosen} {attr['unit']}"
        meta = {"values": {a: x, b: y}}

    elif task == "order3":
        a, b, c = rng.sample(pool, 3)
        x, y, z = _distinct_values(rng, 3, lo, hi, cfg.min_gap)
        vals = {a: x, b: y, c: z}
        answer = max(vals, key=vals.get) if want_more else min(vals, key=vals.get)
        prompt = tpl.format(a=a, b=b, c=c, x=x, y=y, z=z, **fixed)
        chosen = max(vals.values()) if want_more else min(vals.values())
        rationale = f"{chosen} {attr['unit']}"
        meta = {"values": vals}

    else:  # transitive -- no numbers at all, the chain is the whole task
        a, b, c = rng.sample(pool, 3)
        # The premises fix the chain a > b > c. The QUESTION's direction is drawn
        # independently: asking the same way gives the head, the opposite way
        # gives the tail. Without this the answer is always the first entity
        # named, and a model scores 100% by echoing it without reasoning at all.
        ask_more = rng.random() < 0.5
        answer = a if ask_more == want_more else c
        fixed["qcmp"] = L["more"] if ask_more else L["less"]
        prompt = tpl.format(a=a, b=b, c=c, **fixed)
        # No numbers here, so the intermediate step is the chain itself, written
        # out in premise order. Having stated it, picking an end is a local
        # decision rather than a two-hop inference.
        rationale = f"{L['order_word']} {a}, {b}, {c}"
        meta = {"chain": [a, b, c],
                "premise_direction": "more" if want_more else "less",
                "question_direction": "more" if ask_more else "less"}

    # The supervised target is rationale, marker, answer -- so the model must
    # produce the working before the conclusion, and evaluation scores only what
    # follows the marker.
    target = f"{rationale}, {L['therefore']} {answer}"
    return {"task": task, "template": tid, "attr": attr["noun"],
            "prompt": prompt, "rationale": rationale, "answer": answer,
            "target": target, "marker": L["therefore"], "meta": meta}


def generate_split(cfg: ReasoningConfig, split: str, log=print) -> dict:
    """Generate one split and write it as JSONL. Returns its statistics."""
    L = LANGS[cfg.language]
    n = {"train": cfg.n_train, "val": cfg.n_val, "test": cfg.n_test,
         "test_unseen": cfg.n_test_unseen}[split]
    rng = random.Random(f"{cfg.seed}-{cfg.language}-{split}")

    counts = {t: 0 for t in cfg.task_mix}
    per_template: dict[str, int] = {}
    seen: set[str] = set()
    rows, collisions = [], 0

    tasks = list(cfg.task_mix)
    weights = [cfg.task_mix[t] for t in tasks]

    stalled = 0
    while len(rows) < n:
        if stalled > 20000:
            raise RuntimeError(
                f"{cfg.language}/{split}: {len(rows):,} of {n:,} examples before the "
                f"generator stopped finding new prompts. The transitive task uses no "
                f"numbers, so its space is bounded by entities x attributes x "
                f"templates -- enlarge the {split} entity pools or lower n_{split}.")
        task = rng.choices(tasks, weights=weights)[0]
        pool = L[task]
        # Held-out template: reserved for test, excluded everywhere else.
        pool = [t for t in pool if (t[0].endswith("_test") == (split == "test"))] or \
               [t for t in pool if not t[0].endswith("_test")]
        ex = _make_example(rng, cfg.language, task, split, pool, cfg)
        if ex["prompt"] in seen:          # exact-duplicate prompts add nothing
            collisions += 1
            stalled += 1
            continue
        stalled = 0
        seen.add(ex["prompt"])
        ex["id"] = f"{cfg.language}-{split}-{len(rows):06d}"
        rows.append(ex)
        counts[task] += 1
        per_template[ex["template"]] = per_template.get(ex["template"], 0) + 1

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"reasoning_{split}.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = {
        "language": cfg.language, "split": split, "examples": len(rows),
        "tasks": counts, "templates": per_template,
        "distinct_entities": {
            "names": len(L["names_unseen" if split == "test_unseen" else "names"]),
            "objects": len(L["objects_unseen" if split == "test_unseen" else "objects"])}, "duplicate_prompts_skipped": collisions,
        "mean_prompt_chars": round(sum(len(r["prompt"]) for r in rows) / len(rows), 1),
    }
    log(f"  {cfg.language}/{split}: {len(rows):,} examples, "
        f"{len(per_template)} templates -> {path}")
    return stats


def generate_language(cfg: ReasoningConfig, log=print) -> dict:
    """Generate all four splits for one language and write a statistics file."""
    L = LANGS[cfg.language]
    for kind in ("names", "objects"):
        overlap = set(L[kind]) & set(L[f"{kind}_unseen"])
        if overlap:
            raise ValueError(
                f"{kind}_unseen must share nothing with {kind}; "
                f"shared: {sorted(overlap)}")

    splits = ("train", "val", "test", "test_unseen")
    stats = {s: generate_split(cfg, s, log=log) for s in splits}
    stats["leakage_controls"] = {
        "mechanism": "held-out relation patterns",
        "note": ("Entity names are shared across train/val/test deliberately. "
                 "An earlier design gave each split disjoint names; because the "
                 "answer is always an entity, that trained the model to emit one "
                 "set of tokens and scored it on another, and validation loss "
                 "rose 4.2 -> 15.4 while training loss fell to 0.75. Leakage is "
                 "blocked by held-out templates instead, the alternative the "
                 "spec names."),
        "held_out_templates": sorted(
            t[0] for task in ("compare2", "order3", "transitive")
            for t in L[task] if t[0].endswith("_test")),
        "shared_entity_pool": {"names": len(L["names"]),
                               "objects": len(L["objects"])},
        "generalisation_probe": {
            "split": "test_unseen",
            "purpose": ("entities never seen in training, to measure whether the "
                        "model selects from the prompt or memorises which tokens "
                        "are plausible answers"),
            "names": len(L["names_unseen"]),
            "objects": len(L["objects_unseen"])},
    }
    out = Path(f"{cfg.language}/stats/reasoning_dataset.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"  wrote {out}")
    return stats


def main(argv: list[str] | None = None) -> int:
    """CLI. Example::

        python -m common.reasoning.generate --lang hindi
        python -m common.reasoning.generate --lang assamese --n-train 40000
        python -m common.reasoning.generate --lang hindi --preview 6
    """
    import argparse
    from common.io import enable_utf8_console

    enable_utf8_console()
    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("--lang", required=True, choices=("hindi", "assamese"))
    p.add_argument("--n-train", type=int, default=40000)
    p.add_argument("--n-val", type=int, default=2000)
    p.add_argument("--n-test", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20252)
    p.add_argument("--preview", type=int, default=0,
                   help="print N samples of each task and exit, writing nothing")
    args = p.parse_args(argv)

    cfg = ReasoningConfig(language=args.lang, n_train=args.n_train,
                          n_val=args.n_val, n_test=args.n_test, seed=args.seed)

    if args.preview:
        rng = random.Random(args.seed)
        L = LANGS[args.lang]
        for task in ("compare2", "order3", "transitive"):
            print(f"\n=== {args.lang} · {task} ===")
            for _ in range(args.preview):
                ex = _make_example(rng, args.lang, task, "train",
                                   [t for t in L[task] if not t[0].endswith("_test")],
                                   cfg)
                print(f"  [{ex['template']}] {ex['prompt']}")
                print(f"      {ANSWER_PREFIX[args.lang]} {ex['target']}")
        print(f"\n=== {args.lang} · held-out test-only templates ===")
        for task in ("compare2", "order3", "transitive"):
            ex = _make_example(rng, args.lang, task, "test",
                               [t for t in L[task] if t[0].endswith("_test")], cfg)
            print(f"  [{ex['template']}] {ex['prompt']}")
            print(f"      {ANSWER_PREFIX[args.lang]} {ex['target']}")
        return 0

    generate_language(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
