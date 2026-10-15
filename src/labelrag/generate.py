"""Answer questions with an open-source LLM grounded in retrieved label passages, then grade the answers.

python -m labelrag.generate --model qwen2.5-3b --templated 60

Runs on CPU (GitHub Actions): Qwen2.5-Instruct, 4-bit GGUF, through llama.cpp. Grading and the faithfulness
guardrail are in grade.py.
"""

import argparse
import json
import logging
import random
import time

import numpy as np

from . import corpus, evalset
from .config import CHUNKS, EMBEDDINGS, REPORTS, SEED
from .evaluate import same_drug
from .grade import REFUSAL, Judge, citations, grade_row, key_facts, summarise
from .search import Retriever

log = logging.getLogger("labelrag")

MODELS = {
    "qwen2.5-1.5b": ("Qwen/Qwen2.5-1.5B-Instruct-GGUF", "qwen2.5-1.5b-instruct-q4_k_m.gguf"),
    "qwen2.5-3b": ("Qwen/Qwen2.5-3B-Instruct-GGUF", "qwen2.5-3b-instruct-q4_k_m.gguf"),
}
TOP_K = 4

SYSTEM = (
    "You answer questions about prescription drugs using ONLY the numbered FDA label passages given.\n"
    "Rules:\n"
    "1. Write 1 to 3 sentences in your own words. Do not copy passage headers or whole passages.\n"
    "2. Every sentence must state something a passage says, and end with that passage's number, like [2].\n"
    "3. Do not start with 'Yes' or 'No'. State what the label says instead.\n"
    "4. Never add facts, advice or reassurance that is not in the passages.\n"
    f"5. If the passages do not answer the question, or are about a different drug, reply exactly: {REFUSAL}"
)
SYSTEM_CLOSED = "You answer questions about prescription drugs. Answer in 1 to 3 short sentences."

# one worked example so small models see the expected format (made-up drug, not in the corpus)
EXAMPLE_Q = "Can teenagers take examplamine?"
EXAMPLE_P = [
    {"header": "Examplamine | Pediatric use",
     "text": "Safety and effectiveness of examplamine in pediatric patients below 18 years of age have not been established."},
    {"header": "Examplamine | Dosage and administration",
     "text": "The recommended dose for adults is 10 mg once daily. Do not exceed 20 mg per day."},
]
EXAMPLE_A = ("The label says safety and effectiveness of examplamine have not been established in patients under 18 "
             "years old [1]. The adult dose is 10 mg once daily, up to 20 mg per day [2].")


def prompt(question: str, passages: list[dict]) -> str:
    ctx = "\n\n".join(f"[{i + 1}] {p['header']}: {p['text']}" for i, p in enumerate(passages))
    return f"Passages:\n{ctx}\n\nQuestion: {question}"


def messages(question: str, passages: list[dict]) -> list[dict]:
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt(EXAMPLE_Q, EXAMPLE_P)},
            {"role": "assistant", "content": EXAMPLE_A},
            {"role": "user", "content": prompt(question, passages)}]


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), default="qwen2.5-1.5b")
    ap.add_argument("--templated", type=int, default=60)
    ap.add_argument("--limit", type=int, default=0, help="quick test: only this many questions")
    a = ap.parse_args()

    from llama_cpp import Llama

    from .build import encoder

    labels = corpus.load_labels()
    chunks = corpus.load_chunks(CHUNKS)
    emb = np.load(EMBEDDINGS).astype(np.float32)
    enc = encoder()
    retr = Retriever(chunks, corpus.alias_index(labels), emb, encode_query=lambda t: enc(t, query=True))
    repo, file = MODELS[a.model]
    llm = Llama.from_pretrained(repo, file, n_ctx=4096, n_threads=4, verbose=False, seed=SEED)
    judge = Judge()

    known = {lab["drug"] for lab in labels}
    nat = [q for q in evalset.natural() + evalset.natural("holdout") if q["drug"] in known]
    tmpl = random.Random(SEED).sample(evalset.templated(labels), a.templated)
    unans = evalset.unanswerable()
    questions = nat + tmpl + unans
    if a.limit:
        questions = questions[: a.limit]

    def chat(msgs):
        out = llm.create_chat_completion(messages=msgs, temperature=0.0, max_tokens=220)
        return out["choices"][0]["message"]["content"].strip()

    rows, t0 = [], time.time()
    for n, q in enumerate(questions):
        # best configuration in reports/retrieval_metrics.json (the cross-encoder re-ranker lowered hit@1 there)
        ids = retr.search(q["question"], method="hybrid", k=TOP_K, drug_filter=True, rerank=False, hints=True)
        drug_found = bool(retr.allowed(q["question"])[1])
        passages = [chunks[i] for i in ids]
        t = time.time()
        ans = chat(messages(q["question"], passages))
        secs = time.time() - t
        cites = [c for c in citations(ans) if 1 <= c <= len(passages)]
        cited = [passages[c - 1] for c in cites]
        row = {"qid": q["qid"], "source": q["source"], "question": q["question"], "answer": ans, "seconds": round(secs, 1),
               "citations": cites, "must": q.get("must"),
               "passages": [{"id": p["id"], "drug": p["drug"], "section": p["section"], "header": p["header"]} for p in passages]}
        if q["source"] != "unanswerable":
            gold_secs = set(q["sections"])
            row["retrieved_gold"] = any(same_drug(p["drug"], q["drug"]) and p["section"] in gold_secs for p in passages)
            row["cited_right_drug"] = [same_drug(p["drug"], q["drug"]) for p in cited]
            row["cited_right_section"] = [same_drug(p["drug"], q["drug"]) and p["section"] in gold_secs for p in cited]
            if q["source"] in ("natural", "holdout"):
                closed = chat([{"role": "system", "content": SYSTEM_CLOSED}, {"role": "user", "content": q["question"]}])
                row["closed_book_answer"] = closed
                row["closed_book_key_facts"] = key_facts(closed, q.get("must"))
        rows.append(grade_row(row, [(p["header"], p["text"]) for p in passages], judge, drug_found=drug_found))
        if n % 10 == 0:
            log.info("%d/%d answered (%.0f min)", n + 1, len(questions), (time.time() - t0) / 60)

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"answers_{a.model}.json").write_text(json.dumps(rows, indent=1))
    summary = {"model": f"{repo}/{file}", "top_k": TOP_K, "minutes": round((time.time() - t0) / 60, 1)} | summarise(rows) | \
        {"nli_model": judge.name}
    (REPORTS / f"generation_metrics_{a.model}.json").write_text(json.dumps(summary, indent=1))
    log.info(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
