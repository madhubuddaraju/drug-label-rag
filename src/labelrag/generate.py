"""Answer questions with an open-source LLM grounded in retrieved label passages, then grade the answers.

python -m labelrag.generate --templated 60

Runs on CPU (GitHub Actions): Qwen2.5-1.5B-Instruct, 4-bit GGUF, through llama.cpp.
"""

import argparse
import json
import logging
import random
import re
import time

import numpy as np

from . import corpus, evalset
from .config import CHUNKS, EMBEDDINGS, NLI_MODEL, REPORTS, RERANK_MODEL, SEED
from .evaluate import same_drug
from .search import Retriever

log = logging.getLogger("labelrag")

LLM_REPO = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
LLM_FILE = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
REFUSAL = "Not found in the FDA labels provided."
TOP_K = 4

SYSTEM = (
    "You answer questions about prescription drugs using ONLY the numbered FDA label passages given. "
    "Cite the passages you use in square brackets, like [1] or [2][3], at the end of each sentence. "
    "Answer in 2 to 4 short sentences. Do not add facts that are not in the passages. "
    f"If the passages do not answer the question, or are about a different drug, reply exactly: {REFUSAL}"
)
SYSTEM_CLOSED = "You answer questions about prescription drugs. Answer in 2 to 4 short sentences."


def prompt(question: str, passages: list[dict]) -> str:
    ctx = "\n\n".join(f"[{i + 1}] {p['header']}: {p['text']}" for i, p in enumerate(passages))
    return f"Passages:\n{ctx}\n\nQuestion: {question}"


def citations(answer: str) -> list[int]:
    return sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)})


def is_refusal(answer: str) -> bool:
    a = answer.lower()
    return "not found in the fda labels" in a or a.startswith(("i can't find", "i cannot find", "the passages do not"))


def sentences(answer: str) -> list[str]:
    a = re.sub(r"\s*\[\d+\]", "", answer)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", a) if len(s.split()) >= 4]


class Faithfulness:
    """Share of answer sentences entailed by the passages they cite (NLI cross-encoder)."""

    def __init__(self):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(NLI_MODEL, device="cpu")
        labels = {v.lower(): int(k) for k, v in self.model.model.config.id2label.items()}
        self.ent = labels["entailment"]

    def score(self, answer: str, passages: list[str]) -> dict:
        sents = sentences(answer)
        if not sents or not passages:
            return {"sentences": len(sents), "supported": 0}
        pairs = [(p, s) for s in sents for p in passages]
        logits = np.asarray(self.model.predict(pairs, batch_size=32))
        prob = np.exp(logits - logits.max(axis=1, keepdims=True))
        prob = (prob / prob.sum(axis=1, keepdims=True))[:, self.ent].reshape(len(sents), len(passages))
        return {"sentences": len(sents), "supported": int((prob.max(axis=1) > 0.5).sum())}


def key_facts(answer: str, must: list) -> float | None:
    """`must` is a list of groups; each group is a list of acceptable phrasings (any one counts)."""
    if not must:
        return None
    a = answer.lower()
    return float(np.mean([any(m.lower() in a for m in group) for group in must]))


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--templated", type=int, default=60)
    ap.add_argument("--limit", type=int, default=0, help="quick test: only this many questions")
    a = ap.parse_args()

    from llama_cpp import Llama
    from sentence_transformers import CrossEncoder

    from .build import encoder

    labels = corpus.load_labels()
    chunks = corpus.load_chunks(CHUNKS)
    emb = np.load(EMBEDDINGS).astype(np.float32)
    enc = encoder()
    retr = Retriever(chunks, corpus.alias_index(labels), emb, encode_query=lambda t: enc(t, query=True),
                     reranker=CrossEncoder(RERANK_MODEL, device="cpu"))
    llm = Llama.from_pretrained(LLM_REPO, LLM_FILE, n_ctx=4096, n_threads=4, verbose=False, seed=SEED)
    faith = Faithfulness()

    known = {lab["drug"] for lab in labels}
    nat = [q for q in evalset.natural() if q["drug"] in known]
    tmpl = random.Random(SEED).sample(evalset.templated(labels), a.templated)
    unans = evalset.unanswerable()
    questions = nat + tmpl + unans
    if a.limit:
        questions = questions[: a.limit]

    def chat(system, user):
        out = llm.create_chat_completion(messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                                         temperature=0.0, max_tokens=220)
        return out["choices"][0]["message"]["content"].strip()

    rows, t0 = [], time.time()
    for n, q in enumerate(questions):
        ids = retr.search(q["question"], method="hybrid", k=TOP_K, drug_filter=True, rerank=True)
        passages = [chunks[i] for i in ids]
        t = time.time()
        ans = chat(SYSTEM, prompt(q["question"], passages))
        secs = time.time() - t
        cites = [c for c in citations(ans) if 1 <= c <= len(passages)]
        cited = [passages[c - 1] for c in cites]
        row = {"qid": q["qid"], "source": q["source"], "question": q["question"], "answer": ans, "seconds": round(secs, 1),
               "refused": is_refusal(ans), "citations": cites,
               "passages": [{"id": p["id"], "drug": p["drug"], "section": p["section"], "header": p["header"]} for p in passages]}
        if q["source"] != "unanswerable":
            gold_secs = set(q["sections"])
            row["retrieved_gold"] = any(same_drug(p["drug"], q["drug"]) and p["section"] in gold_secs for p in passages)
            row["cited_right_drug"] = [same_drug(p["drug"], q["drug"]) for p in cited]
            row["cited_right_section"] = [same_drug(p["drug"], q["drug"]) and p["section"] in gold_secs for p in cited]
            if not row["refused"]:
                row |= faith.score(ans, [p["text"] for p in cited] or [p["text"] for p in passages])
            row["key_facts"] = key_facts(ans, q.get("must"))
            if q["source"] == "natural":
                closed = chat(SYSTEM_CLOSED, q["question"])
                row["closed_book_answer"] = closed
                row["closed_book_key_facts"] = key_facts(closed, q.get("must"))
        rows.append(row)
        if n % 10 == 0:
            log.info("%d/%d answered (%.0f min)", n + 1, len(questions), (time.time() - t0) / 60)

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "answers.json").write_text(json.dumps(rows, indent=1))
    summary = summarise(rows) | {"model": f"{LLM_REPO}/{LLM_FILE}", "top_k": TOP_K, "nli_model": NLI_MODEL,
                                 "minutes": round((time.time() - t0) / 60, 1)}
    (REPORTS / "generation_metrics.json").write_text(json.dumps(summary, indent=1))
    log.info(json.dumps(summary, indent=1))


def summarise(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["source"] != "unanswerable"]
    un = [r for r in rows if r["source"] == "unanswerable"]
    answered = [r for r in ans if not r["refused"]]
    sent = sum(r.get("sentences", 0) for r in answered)
    cites = [x for r in answered for x in r["cited_right_section"]]
    drug_cites = [x for r in answered for x in r["cited_right_drug"]]

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return round(float(np.mean(xs)), 4) if xs else None

    out = {
        "answerable": len(ans), "unanswerable": len(un),
        "answer_rate": mean([not r["refused"] for r in ans]),
        "false_refusal_when_gold_retrieved": mean([r["refused"] for r in ans if r["retrieved_gold"]]),
        "correct_refusal_rate": mean([r["refused"] for r in un]),
        "faithfulness": round(sum(r.get("supported", 0) for r in answered) / sent, 4) if sent else None,
        "answers_with_citation": mean([bool(r["citations"]) for r in answered]),
        "citation_right_drug": mean(drug_cites),
        "citation_right_section": mean(cites),
        "key_fact_recall_rag": mean([r["key_facts"] for r in ans]),
        "key_fact_recall_closed_book": mean([r.get("closed_book_key_facts") for r in ans]),
        "median_seconds": float(np.median([r["seconds"] for r in rows])) if rows else None,
        "by_source": {},
    }
    for s in sorted({r["source"] for r in rows}):
        rs = [r for r in rows if r["source"] == s]
        out["by_source"][s] = {"n": len(rs), "refusal_rate": mean([r["refused"] for r in rs]),
                               "key_fact_recall": mean([r.get("key_facts") for r in rs])}
    return out


if __name__ == "__main__":
    main()
