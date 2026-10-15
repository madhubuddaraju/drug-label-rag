"""Build the chunk index, embed it, and evaluate every retrieval configuration.

python -m labelrag.build            # full run (needs the embedding and re-ranking models)
python -m labelrag.build --no-dense # BM25 only, works offline
"""

import argparse
import json
import logging
import time

import numpy as np

from . import corpus, evalset
from .config import CHUNKS, DATA, EMBED_MODEL, EMBEDDINGS, QUERY_PREFIX, REPORTS, RERANK_MODEL
from .evaluate import breakdown, relevant, run_config
from .search import Retriever, doc_text

log = logging.getLogger("labelrag")

CONFIGS = [
    # name, method, header, drug_filter, rerank
    ("BM25", "bm25", False, False, False),
    ("BM25 + chunk headers", "bm25", True, False, False),
    ("BM25 + headers + drug-name filter", "bm25", True, True, False),
    ("Dense (bge-small)", "dense", False, False, False),
    ("Dense + chunk headers", "dense", True, False, False),
    ("Dense, binary-quantised", "binary", True, False, False),
    ("Hybrid (RRF)", "hybrid", True, False, False),
    ("Hybrid + drug-name filter", "hybrid", True, True, False),
    ("Hybrid, binary + drug-name filter", "hybrid_binary", True, True, False),
    ("Hybrid + drug filter + re-ranker", "hybrid", True, True, True),
]


def encoder():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL, device="cpu")

    def enc(texts, query=False):
        texts = [QUERY_PREFIX + t for t in texts] if query else texts
        return model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    return enc


def embed_cached(enc, chunks, with_header, path):
    if path.exists():
        e = np.load(path).astype(np.float32)
        if len(e) == len(chunks):
            return e
    t0 = time.time()
    e = enc([doc_text(c, with_header) for c in chunks])
    log.info("embedded %d chunks (headers=%s) in %.0fs", len(chunks), with_header, time.time() - t0)
    np.save(path, e.astype(np.float16))
    return e


def run(dense: bool = True) -> dict:
    labels = corpus.load_labels()
    chunks = corpus.build_chunks(labels)
    corpus.save_chunks(chunks, CHUNKS)
    aliases = corpus.alias_index(labels)
    log.info("%d drugs -> %d chunks, %d drug-name aliases", len(labels), len(chunks), len(aliases))

    questions = evalset.templated(labels) + evalset.natural()
    known = {lab["drug"] for lab in labels}
    questions = [q for q in questions if q["drug"] in known]
    gold = {q["qid"]: relevant(chunks, q) for q in questions}

    enc = encoder() if dense else None
    emb_h = emb_p = None
    qvecs = {}
    reranker = None
    if dense:
        emb_h = embed_cached(enc, chunks, True, EMBEDDINGS)
        emb_p = embed_cached(enc, chunks, False, DATA / "embeddings_plain.npy")
        qv = enc([q["question"] for q in questions], query=True)
        qvecs = {q["qid"]: v for q, v in zip(questions, qv, strict=True)}
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder(RERANK_MODEL, device="cpu")

    retrievers = {True: Retriever(chunks, aliases, emb_h, reranker=reranker, with_header=True),
                  False: Retriever(chunks, aliases, emb_p, reranker=reranker, with_header=False)}
    results, per_q = [], {}
    for name, method, hdr, filt, rr in CONFIGS:
        if method != "bm25" and not dense:
            continue
        summary, rows = run_config(retrievers[hdr], questions, qvecs, chunks, gold, method=method, k=10,
                                   drug_filter=filt, rerank=rr)
        by_source = breakdown(rows, questions, "source")
        results.append({"name": name, "method": method, "headers": hdr, "drug_filter": filt, "rerank": rr,
                        **summary, "by_source": by_source})
        per_q[name] = rows
        log.info("%-36s hit@5 %.3f  mrr %.3f  natural hit@5 %.3f  %.1f ms", name, summary["hit@5"], summary["mrr@10"],
                 by_source.get("natural", {}).get("hit@5", float("nan")), summary["ms_per_query"])

    best = max(results, key=lambda r: (r["by_source"].get("natural", r)["mrr@10"], r["mrr@10"]))
    rows = per_q[best["name"]]
    qmeta = {q["qid"]: q for q in questions}
    failures = []
    for r in rows:
        if not r["hit@5"]:
            q = qmeta[r["qid"]]
            t = chunks[r["ranked"][0]] if r["ranked"] else None
            failures.append({"question": q["question"], "gold_drug": q["drug"], "gold_section": q["section"],
                             "source": q["source"], "got_drug": t and t["drug"], "got_section": t and t["section"]})
    report = {
        "corpus": {"drugs": len(labels), "chunks": len(chunks), "aliases": len(aliases),
                   "avg_chunk_words": round(float(np.mean([len(c["text"].split()) for c in chunks])), 1),
                   "sections": {s: sum(c["section"] == s for c in chunks) for s in sorted({c["section"] for c in chunks})}},
        "questions": {"templated": sum(q["source"] == "templated" for q in questions),
                      "natural": sum(q["source"] == "natural" for q in questions)},
        "configs": results,
        "best": best["name"],
        "best_by_section": breakdown(rows, questions, "section"),
        "best_by_name_type": breakdown(rows, questions, "name_type"),
        "failures": failures[:40],
        "failure_count": len(failures),
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / ("retrieval_metrics.json" if dense else "retrieval_metrics_bm25.json")).write_text(json.dumps(report, indent=1))
    return report


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-dense", action="store_true")
    run(dense=not ap.parse_args().no_dense)


if __name__ == "__main__":
    main()
