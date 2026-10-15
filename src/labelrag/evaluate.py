"""Retrieval metrics."""

import math
import time

import numpy as np

from .corpus import base_name


def same_drug(a: str, b: str) -> bool:
    """Salt forms and formulations of one ingredient (fluoxetine / fluoxetine hydrochloride) count as the same drug."""
    return a == b or base_name(a) == base_name(b)


def relevant(chunks: list[dict], q: dict) -> set[int]:
    secs = set(q["sections"])
    return {c["id"] for c in chunks if same_drug(c["drug"], q["drug"]) and c["section"] in secs}


def score_ranking(ranked: list[int], gold: set[int], chunks: list[dict], drug: str) -> dict:
    hits = [i in gold for i in ranked]
    first = next((p for p, h in enumerate(hits) if h), None)
    dcg = sum(1 / math.log2(p + 2) for p, h in enumerate(hits[:10]) if h)
    idcg = sum(1 / math.log2(p + 2) for p in range(min(len(gold), 10)))
    return {"hit@1": float(bool(hits[:1] and hits[0])), "hit@5": float(any(hits[:5])), "hit@10": float(any(hits[:10])),
            "mrr@10": 1 / (first + 1) if first is not None and first < 10 else 0.0,
            "ndcg@10": dcg / idcg if idcg else 0.0,
            "drug@1": float(bool(ranked) and same_drug(chunks[ranked[0]]["drug"], drug))}


def run_config(retriever, questions, qvecs, chunks, gold, **kw) -> tuple[dict, list[dict]]:
    rows, t0 = [], time.perf_counter()
    for q in questions:
        ranked = retriever.search(q["question"], qvec=qvecs.get(q["qid"]) if qvecs else None, **kw)
        rows.append({"qid": q["qid"], "ranked": ranked[:10], **score_ranking(ranked, gold[q["qid"]], chunks, q["drug"])})
    ms = (time.perf_counter() - t0) / max(len(questions), 1) * 1000
    return summarise(rows) | {"ms_per_query": round(ms, 1)}, rows


def summarise(rows: list[dict]) -> dict:
    keys = ["hit@1", "hit@5", "hit@10", "mrr@10", "ndcg@10", "drug@1"]
    return {k: round(float(np.mean([r[k] for r in rows])), 4) for k in keys} | {"n": len(rows)}


def breakdown(rows: list[dict], questions: list[dict], field: str) -> dict:
    by = {}
    meta = {q["qid"]: q for q in questions}
    for r in rows:
        by.setdefault(meta[r["qid"]].get(field, "?"), []).append(r)
    return {k: summarise(v) for k, v in sorted(by.items())}
