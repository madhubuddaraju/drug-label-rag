"""Labeled questions for retrieval evaluation.

Three sources:
- templated: generated from label structure, so the gold section is known exactly (large, systematic)
- natural: hand-written questions in everyday wording, with the gold drug and section(s) (evals/natural.jsonl)
- unanswerable: questions the labels cannot answer, used to test abstention (evals/unanswerable.jsonl)
"""

import json
import random

from .config import EVALS, SEED

TEMPLATES = {
    "indications": ["What is {d} used for?", "What conditions does {d} treat?"],
    "dosage": ["What is the usual dose of {d}?", "How should {d} be taken?"],
    "contraindications": ["Who should not take {d}?", "When is {d} contraindicated?"],
    "warnings": ["What are the main safety warnings for {d}?", "What precautions should I know before starting {d}?"],
    "adverse_reactions": ["What are the common side effects of {d}?", "What adverse reactions were reported with {d}?"],
    "interactions": ["Which medicines interact with {d}?", "Can {d} be taken with other drugs?"],
    "pregnancy": ["Is {d} safe during pregnancy?", "Can I take {d} while pregnant?"],
    "lactation": ["Can I breastfeed while taking {d}?"],
    "pediatric": ["Can children take {d}?"],
    "geriatric": ["Are there special considerations for elderly patients taking {d}?"],
    "overdosage": ["What happens if someone takes too much {d}?", "How is an overdose of {d} treated?"],
    "boxed_warning": ["Does {d} have a boxed warning?"],
}
# sections that also answer a question about another section
ALSO = {"warnings": ["boxed_warning"], "boxed_warning": ["warnings"], "pregnancy": ["lactation"],
        "lactation": ["pregnancy"]}


def templated(labels: list[dict], n: int = 600, seed: int = SEED) -> list[dict]:
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        lab = rng.choice(labels)
        sec = rng.choice([s for s in lab["sections"] if s in TEMPLATES])
        brand = bool(lab["aliases"]) and rng.random() < 0.5
        name = lab["aliases"][0] if brand else lab["drug"]
        q = rng.choice(TEMPLATES[sec]).format(d=name.title() if brand else name)
        out.append({"qid": f"t{len(out)}", "question": q, "drug": lab["drug"], "sections": [sec] + ALSO.get(sec, []),
                    "source": "templated", "name_type": "brand" if brand else "generic", "section": sec})
    return out


def load_jsonl(name: str) -> list[dict]:
    p = EVALS / name
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def natural() -> list[dict]:
    rows = load_jsonl("natural.jsonl")
    for i, r in enumerate(rows):
        r.setdefault("qid", f"n{i}")
        r["source"] = "natural"
        r.setdefault("section", r["sections"][0])
        r.setdefault("name_type", "mixed")
    return rows


def unanswerable() -> list[dict]:
    rows = load_jsonl("unanswerable.jsonl")
    for i, r in enumerate(rows):
        r.setdefault("qid", f"u{i}")
        r["source"] = "unanswerable"
    return rows
