"""Build the static site: docs/index.html (results + search demo) and docs/corpus.json (index for the browser).

python -m labelrag.web
"""

import base64
import json

import numpy as np

from . import corpus
from .config import CHUNKS, DOCS, EMBED_MODEL_WEB, EMBEDDINGS, QUERY_PREFIX, REPORTS, ROOT, SECTION_TITLES
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


def build():
    labels = corpus.load_labels()
    chunks = corpus.load_chunks(CHUNKS)
    emb = np.load(EMBEDDINGS).astype(np.float32) if EMBEDDINGS.exists() else None
    DOCS.mkdir(exist_ok=True)
    (DOCS / "corpus.json").write_text(json.dumps(browser_corpus(labels, chunks, emb), separators=(",", ":")))

    answers = _load("answers.json") or []
    data = {
        "retrieval": _load("retrieval_metrics.json") or _load("retrieval_metrics_bm25.json"),
        "generation": _load("generation_metrics.json"),
        "answers": [a for a in answers if a["source"] in ("natural", "holdout", "unanswerable")],
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
