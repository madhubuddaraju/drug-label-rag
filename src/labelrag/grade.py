"""Grade generated answers and apply the faithfulness guardrail.

Each answer sentence is a claim. A claim is supported when an NLI cross-encoder says it is entailed by a sentence,
or a pair of adjacent sentences, from the retrieved passages. NLI models are trained on short premises, so scoring
against whole 160-word passages under-counts support; windows fix that.

The guardrail keeps only supported claims. If none survive, the answer becomes a refusal.

python -m labelrag.grade --tag qwen2.5-1.5b     # re-grade saved answers without regenerating them
"""

import argparse
import json
import re

import numpy as np

from . import corpus
from .config import CHUNKS, NLI_MODEL, REPORTS

REFUSAL = "Not found in the FDA labels provided."
THRESHOLD = 0.5
_CITE = re.compile(r"\s*\[(\d+)\]")


def citations(answer: str) -> list[int]:
    return sorted({int(n) for n in _CITE.findall(answer)})


def is_refusal(answer: str) -> bool:
    a = answer.lower().strip()
    return "not found in the fda labels" in a or a.startswith(("i can't find", "i cannot find", "the passages do not")) or not a


def split_answer(answer: str) -> list[tuple[str, str]]:
    """(sentence as written, with its citation markers) and (claim text without markers), for claims of 4+ words."""
    a = re.sub(r"([.!?])\s*((?:\[\d+\]\s*)+)", lambda m: " " + m.group(2).strip() + m.group(1) + " ", answer)
    out = []
    for raw in re.split(r"(?<=[.!?])\s+(?=[^\[\s])", a.strip()):
        claim = _CITE.sub("", raw).strip()
        if len(claim.split()) >= 4:
            out.append((raw.strip(), claim))
    return out


def sentences(answer: str) -> list[str]:
    return [c for _, c in split_answer(answer)]


def windows(text: str) -> list[str]:
    s = corpus.split_sentences(text)
    return s + [f"{a} {b}" for a, b in zip(s, s[1:], strict=False)]


def key_facts(answer: str, must: list) -> float | None:
    """`must` is a list of groups; each group is a list of acceptable phrasings (any one counts)."""
    if not must:
        return None
    a = answer.lower()
    return float(np.mean([any(m.lower() in a for m in group) for group in must]))


class Judge:
    def __init__(self, model: str = NLI_MODEL):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model, device="cpu")
        labels = {v.lower(): int(k) for k, v in self.model.model.config.id2label.items()}
        self.ent = labels["entailment"]
        self.name = model

    def support(self, claims: list[str], passages: list[str]) -> np.ndarray:
        """Highest entailment probability of each claim over all passage windows."""
        prem = [w for p in passages for w in windows(p)]
        if not claims or not prem:
            return np.zeros(len(claims))
        pairs = [(w, c) for c in claims for w in prem]
        logits = np.asarray(self.model.predict(pairs, batch_size=64))
        p = np.exp(logits - logits.max(axis=1, keepdims=True))
        p = (p / p.sum(axis=1, keepdims=True))[:, self.ent]
        return p.reshape(len(claims), len(prem)).max(axis=1)


def grade_row(row: dict, passages: list[str], judge) -> dict:
    """Adds support scores, the guarded answer and key-fact recall for both versions."""
    row = dict(row)
    row["refused"] = is_refusal(row["answer"])
    parts = [] if row["refused"] else split_answer(row["answer"])
    probs = judge.support([c for _, c in parts], passages) if parts else np.zeros(0)
    keep = [raw for (raw, _), p in zip(parts, probs, strict=True) if p >= THRESHOLD]
    row["claims"] = len(parts)
    row["supported"] = len(keep)
    row["support"] = [round(float(p), 3) for p in probs]
    row["guarded_answer"] = " ".join(keep) if keep else REFUSAL
    row["guarded_refused"] = not keep
    row["key_facts"] = key_facts(row["answer"], row.get("must"))
    row["guarded_key_facts"] = key_facts(row["guarded_answer"], row.get("must"))
    return row


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(float(np.mean(xs)), 4) if xs else None


def summarise(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["source"] != "unanswerable"]
    un = [r for r in rows if r["source"] == "unanswerable"]
    answered = [r for r in ans if not r["refused"]]
    claims = sum(r.get("claims", 0) for r in answered)
    cites = [x for r in answered for x in r.get("cited_right_section", [])]
    drug_cites = [x for r in answered for x in r.get("cited_right_drug", [])]
    raw = {
        "answer_rate": _mean([not r["refused"] for r in ans]),
        "correct_refusal_rate": _mean([r["refused"] for r in un]),
        "false_refusal_when_gold_retrieved": _mean([r["refused"] for r in ans if r.get("retrieved_gold")]),
        "faithfulness": round(sum(r.get("supported", 0) for r in answered) / claims, 4) if claims else None,
        "fully_supported_answers": _mean([r.get("claims", 0) > 0 and r["supported"] == r["claims"] for r in answered]),
        "answers_with_citation": _mean([bool(r.get("citations")) for r in answered]),
        "citation_right_drug": _mean(drug_cites),
        "citation_right_section": _mean(cites),
        "key_fact_recall": _mean([r.get("key_facts") for r in ans]),
    }
    guarded = {
        "answer_rate": _mean([not r["guarded_refused"] for r in ans]),
        "correct_refusal_rate": _mean([r["guarded_refused"] for r in un]),
        "false_refusal_when_gold_retrieved": _mean([r["guarded_refused"] for r in ans if r.get("retrieved_gold")]),
        "claims_removed": round(1 - sum(r.get("supported", 0) for r in answered) / claims, 4) if claims else None,
        "key_fact_recall": _mean([r.get("guarded_key_facts") for r in ans]),
    }
    out = {"answerable": len(ans), "unanswerable": len(un), "raw": raw, "guarded": guarded,
           "key_fact_recall_closed_book": _mean([r.get("closed_book_key_facts") for r in ans]),
           "median_seconds": float(np.median([r["seconds"] for r in rows])) if rows else None, "by_source": {}}
    for s in sorted({r["source"] for r in rows}):
        rs = [r for r in rows if r["source"] == s]
        out["by_source"][s] = {"n": len(rs), "refusal_rate": _mean([r["refused"] for r in rs]),
                               "guarded_refusal_rate": _mean([r["guarded_refused"] for r in rs]),
                               "key_fact_recall": _mean([r.get("key_facts") for r in rs]),
                               "guarded_key_fact_recall": _mean([r.get("guarded_key_facts") for r in rs])}
    return out


def regrade(tag: str) -> dict:
    path = REPORTS / f"answers_{tag}.json"
    rows = json.loads(path.read_text())
    from . import evalset
    must = {q["qid"]: q.get("must") for q in evalset.natural() + evalset.natural("holdout")}
    for r in rows:
        r.setdefault("must", must.get(r["qid"]))
    text = {c["id"]: c["text"] for c in corpus.load_chunks(CHUNKS)}
    judge = Judge()
    rows = [grade_row(r, [text[p["id"]] for p in r["passages"]], judge) for r in rows]
    path.write_text(json.dumps(rows, indent=1))
    mpath = REPORTS / f"generation_metrics_{tag}.json"
    meta = json.loads(mpath.read_text()) if mpath.exists() else {}
    summary = {k: meta[k] for k in ("model", "top_k", "minutes") if k in meta} | summarise(rows) | {"nli_model": judge.name}
    mpath.write_text(json.dumps(summary, indent=1))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    print(json.dumps(regrade(ap.parse_args().tag), indent=1))


if __name__ == "__main__":
    main()
