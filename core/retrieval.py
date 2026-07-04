"""Hybrid retrieval: BM25 + dense embeddings, fused with Reciprocal Rank Fusion.

BM25 is implemented from scratch (Okapi variant) — no search dependency.
Dense scores join the fusion only when embeddings exist; the retriever is
lexical-only otherwise, so demo mode needs zero keys.

Why hybrid: loan files are full of exact tokens (₹41,000, NACH, employer
names) where lexical wins, and paraphrased questions ("what does she earn?")
where dense wins. RRF takes the best of both without score calibration.
"""

import math
import re
from collections import Counter

import numpy as np

_STOP = {
    "the", "a", "an", "and", "or", "of", "in", "on", "to", "is", "was", "for",
    "with", "at", "by", "from", "as", "it", "this", "that", "be", "are", "has",
    "have", "had", "i", "my", "he", "she", "his", "her", "we", "they", "you",
}

_TOKEN_RE = re.compile(r"[a-z0-9₹]+")

_SUFFIXES = ("ing", "ed", "es", "ly", "s")


def _stem(t: str) -> str:
    """Light suffix stripping so 'employs'/'employed'/'employment' meet halfway."""
    if len(t) <= 4 or t.isdigit():
        return t
    for suf in _SUFFIXES:
        if t.endswith(suf) and len(t) - len(suf) >= 3:
            return t[: -len(suf)]
    return t


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP]


class BM25:
    """Okapi BM25 (k1=1.5, b=0.75)."""

    def __init__(self, docs: list[str], k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.doc_tokens = [tokenize(d) for d in docs]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.avg_len = sum(self.doc_len) / len(self.doc_len) if docs else 1.0
        self.tf = [Counter(t) for t in self.doc_tokens]
        df = Counter()
        for toks in self.doc_tokens:
            df.update(set(toks))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def scores(self, query: str) -> list[float]:
        q = tokenize(query)
        out = []
        for i in range(len(self.doc_tokens)):
            s = 0.0
            for t in q:
                if t not in self.tf[i]:
                    continue
                f = self.tf[i][t]
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avg_len)
                s += self.idf.get(t, 0.0) * f * (self.k1 + 1) / denom
            out.append(s)
        return out


def _rank(scores: list[float]) -> dict[int, int]:
    """index → rank (1-based), best first. Zero scores get no rank."""
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return {i: r + 1 for r, i in enumerate(order) if scores[i] > 0}


class HybridRetriever:
    """Retrieval over a segment list with optional dense fusion."""

    RRF_K = 60

    def __init__(self, segments: list[dict]):
        self.segments = segments
        self.bm25 = BM25([s["text"] for s in segments]) if segments else None
        vecs = [s.get("embedding") for s in segments]
        self.dense = None
        if segments and all(v is not None for v in vecs):
            m = np.array(vecs, dtype=np.float32)
            self.dense = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)

    def retrieve(self, query: str, k: int = 5, query_vec=None) -> list[dict]:
        if not self.segments:
            return []
        lex_ranks = _rank(self.bm25.scores(query))

        dense_ranks = {}
        if self.dense is not None and query_vec is not None:
            qv = np.array(query_vec, dtype=np.float32)
            qv = qv / (np.linalg.norm(qv) + 1e-9)
            sims = self.dense @ qv
            dense_ranks = _rank(sims.tolist())

        fused: dict[int, float] = {}
        for i, r in lex_ranks.items():
            fused[i] = fused.get(i, 0.0) + 1.0 / (self.RRF_K + r)
        for i, r in dense_ranks.items():
            fused[i] = fused.get(i, 0.0) + 1.0 / (self.RRF_K + r)

        top = sorted(fused, key=fused.get, reverse=True)[:k]
        results = []
        for i in top:
            seg = dict(self.segments[i])
            seg["score"] = round(fused[i], 5)
            seg["matched_by"] = [
                name for name, ranks in (("bm25", lex_ranks), ("dense", dense_ranks)) if i in ranks
            ]
            results.append(seg)
        return results
