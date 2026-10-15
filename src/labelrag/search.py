"""Retrievers: BM25, dense (float and binary-quantised), hybrid fusion, drug filtering and cross-encoder re-ranking."""

import math
from collections import Counter

import numpy as np
from scipy import sparse

from .corpus import find_drugs, tokenize


def doc_text(c: dict, with_header: bool = True) -> str:
    return f"{c['header']}. {c['text']}" if with_header else c["text"]


class BM25:
    """Okapi BM25 (k1=1.2, b=0.75) with Lucene-style idf, as a sparse term-document matrix."""

    def __init__(self, docs: list[str], k1: float = 1.2, b: float = 0.75):
        toks = [tokenize(d) for d in docs]
        self.vocab = {t: i for i, t in enumerate(sorted({t for d in toks for t in d}))}
        n = len(docs)
        lens = np.array([len(d) for d in toks], dtype=np.float64)
        avg = lens.mean() if n else 1.0
        rows, cols, vals = [], [], []
        df = Counter()
        for i, d in enumerate(toks):
            tf = Counter(d)
            df.update(tf.keys())
            norm = k1 * (1 - b + b * lens[i] / avg)
            for t, c in tf.items():
                rows.append(i)
                cols.append(self.vocab[t])
                vals.append(c * (k1 + 1) / (c + norm))
        self.idf = np.zeros(len(self.vocab))
        for t, f in df.items():
            self.idf[self.vocab[t]] = math.log(1 + (n - f + 0.5) / (f + 0.5))
        self.m = sparse.csr_matrix((vals, (rows, cols)), shape=(n, len(self.vocab))).tocsc()

    def scores(self, query: str) -> np.ndarray:
        q = Counter(t for t in tokenize(query) if t in self.vocab)
        out = np.zeros(self.m.shape[0])
        for t in q:
            j = self.vocab[t]
            out += self.idf[j] * self.m[:, j].toarray().ravel()
        return out


def to_binary(emb: np.ndarray) -> np.ndarray:
    """Sign-quantise embeddings to 1 bit per dimension (384 floats -> 48 bytes)."""
    return np.packbits(emb > 0, axis=1)


_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.int32)


def hamming_scores(qbits: np.ndarray, bits: np.ndarray) -> np.ndarray:
    """Similarity = number of matching bits (higher is better)."""
    return bits.shape[1] * 8 - _POP[np.bitwise_xor(bits, qbits)].sum(axis=1)


def rrf(rankings: list[np.ndarray], k: int = 60, depth: int = 100) -> dict[int, float]:
    """Reciprocal rank fusion of several ranked id lists."""
    out: dict[int, float] = {}
    for r in rankings:
        for pos, i in enumerate(r[:depth]):
            out[int(i)] = out.get(int(i), 0.0) + 1.0 / (k + pos + 1)
    return out


def top(scores: np.ndarray, k: int, allowed: np.ndarray | None = None) -> np.ndarray:
    s = scores if allowed is None else np.where(allowed, scores, -np.inf)
    k = min(k, int(np.isfinite(s).sum()))
    if k <= 0:
        return np.array([], dtype=int)
    idx = np.argpartition(-s, k - 1)[:k]
    return idx[np.argsort(-s[idx], kind="stable")]


class Retriever:
    """Holds every index so evaluation can compare configurations on identical inputs."""

    def __init__(self, chunks, aliases, embeddings=None, encode_query=None, reranker=None, with_header=True):
        self.chunks = chunks
        self.aliases = aliases
        self.drug_of = np.array([c["drug"] for c in chunks])
        self.bm25 = BM25([doc_text(c, with_header) for c in chunks])
        self.emb = embeddings
        self.bits = to_binary(embeddings) if embeddings is not None else None
        self.encode_query = encode_query
        self.reranker = reranker

    def allowed(self, question: str):
        drugs = find_drugs(question, self.aliases)
        return (np.isin(self.drug_of, drugs) if drugs else None), drugs

    def search(self, question: str, method: str = "hybrid", k: int = 10, drug_filter: bool = True,
               rerank: bool = False, qvec: np.ndarray | None = None) -> list[int]:
        allowed, _ = self.allowed(question) if drug_filter else (None, [])
        depth = 50 if rerank else k
        if method in ("dense", "binary", "hybrid", "hybrid_binary") and qvec is None:
            qvec = self.encode_query([question])[0]
        if method == "bm25":
            ids = top(self.bm25.scores(question), depth, allowed)
        elif method == "dense":
            ids = top(self.emb @ qvec, depth, allowed)
        elif method == "binary":
            ids = top(hamming_scores(to_binary(qvec[None]), self.bits).astype(float), depth, allowed)
        elif method in ("hybrid", "hybrid_binary"):
            dense = self.emb @ qvec if method == "hybrid" else hamming_scores(to_binary(qvec[None]), self.bits).astype(float)
            fused = rrf([top(self.bm25.scores(question), 100, allowed), top(dense, 100, allowed)])
            ids = np.array(sorted(fused, key=lambda i: -fused[i])[:depth], dtype=int)
        else:
            raise ValueError(method)
        if rerank and len(ids):
            pairs = [(question, doc_text(self.chunks[i])) for i in ids]
            s = np.asarray(self.reranker.predict(pairs, batch_size=64))
            ids = ids[np.argsort(-s, kind="stable")]
        return [int(i) for i in ids[:k]]
