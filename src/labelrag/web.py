"""Build the static site: docs/index.html (results + search demo) and docs/corpus.json (index for the browser).

python -m labelrag.web
"""

import base64
import json

import numpy as np

from . import corpus
from .config import CHUNKS, DOCS, EMBED_MODEL_WEB, EMBEDDINGS, QUERY_PREFIX, REPORTS, ROOT, SECTION_TITLES
from .grade import THRESHOLD, split_answer
from .search import to_binary


def browser_corpus(labels: list[dict], chunks: list[dict], emb: np.ndarray | None) -> dict:
    return {
        "stop": sorted(corpus.STOP),
        "aliases": corpus.alias_index(labels),
        "dim": int(emb.shape[1]) if emb is not None else 0,
        "bits": base64.b64encode(to_binary(emb).tobytes()).decode() if emb is not None else None,
        "model": EMBED_MODEL_WEB,
        "query_prefix": QUERY_PREFIX,
        "sections": SECTION_TITLES,
        "hints": corpus.SECTION_HINTS,
        "drugs": {lab["drug"]: {"set_id": lab["set_id"], "effective_time": lab["effective_time"],
                                "manufacturer": lab["manufacturer"]} for lab in labels},
        "chunks": [{"drug": c["drug"], "section": c["section"], "header": c["header"], "text": c["text"]} for c in chunks],
    }


def _load(name):
    p = REPORTS / name
    return json.loads(p.read_text()) if p.exists() else None


PROMPT1 = "qwen2.5-1.5b-prompt1"     # first run: shorter prompt without the worked example


def generation_runs() -> dict:
    """tag -> metrics for every graded run (only runs graded with the current, sentence-level judge)."""
    out = {}
    for p in sorted(REPORTS.glob("generation_metrics_*.json")):
        m = json.loads(p.read_text())
        if "guarded" in m:
            out[p.stem.removeprefix("generation_metrics_")] = m
    return out


def answers_for_web(tag: str) -> list[dict]:
    rows = json.loads((REPORTS / f"answers_{tag}.json").read_text())
    out = []
    for r in rows:
        if r["source"] not in ("natural", "holdout", "unanswerable"):
            continue
        parts = [] if r["refused"] else [{"text": raw, "ok": p >= THRESHOLD}
                                         for (raw, _), p in zip(split_answer(r["answer"]), r["support"], strict=True)]
        out.append({k: r.get(k) for k in ("source", "question", "answer", "refused", "guarded_refused", "guarded_answer",
                                          "key_facts", "guarded_key_facts", "citations", "closed_book_answer", "guarded_by")}
                   | {"parts": parts, "passages": [{"header": p["header"]} for p in r["passages"]]})
    return out


def build():
    labels = corpus.load_labels()
    chunks = corpus.load_chunks(CHUNKS)
    emb = np.load(EMBEDDINGS).astype(np.float32) if EMBEDDINGS.exists() else None
    DOCS.mkdir(exist_ok=True)
    (DOCS / "corpus.json").write_text(json.dumps(browser_corpus(labels, chunks, emb), separators=(",", ":")))

    runs = generation_runs()
    best = max((t for t in runs if t != PROMPT1), key=lambda t: runs[t]["guarded"]["key_fact_recall"] or 0, default=None)
    data = {
        "retrieval": _load("retrieval_metrics.json") or _load("retrieval_metrics_bm25.json"),
        "generation": runs,
        "best_model": best,
        "answers": answers_for_web(best) if best else [],
        "meta": json.loads((ROOT / "data" / "labels_meta.json").read_text()),
    }
    page = (ROOT / "dashboard" / "template.html").read_text()
    page = page.replace("__RAG_JS__", (ROOT / "dashboard" / "rag.js").read_text())
    page = page.replace("__DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    (DOCS / "index.html").write_text(page)
    for f in ("index.html", "corpus.json"):
        print(f"docs/{f}: {(DOCS / f).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    build()
