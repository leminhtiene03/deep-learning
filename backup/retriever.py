"""Hybrid BM25 + FAISS retriever for chunks_new.json.

Usage:
    from retriever import HybridRetriever
    r = HybridRetriever()
    hits = r.search("Học viện có bao nhiêu cơ sở đào tạo?", top_k=5)
    for h in hits:
        print(h["score"], h["chunk"]["content"][:120])
"""

from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from pyvi import ViTokenizer
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).parent
INDEX_DIR = ROOT / "index"


def _segment(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return ViTokenizer.tokenize(text.lower())


def _minmax(x: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]. Returns zeros if all values are equal."""
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


@dataclass
class Hit:
    chunk: dict
    score: float
    bm25_score: float
    faiss_score: float


class HybridRetriever:
    def __init__(self, index_dir: Path = INDEX_DIR, alpha: float = 0.5) -> None:
        """alpha = BM25 weight in the fused score; (1 - alpha) goes to FAISS."""
        self.index_dir = Path(index_dir)
        self.alpha = alpha

        with (self.index_dir / "meta.json").open("r", encoding="utf-8") as f:
            meta = json.load(f)
        self.model_name: str = meta["model"]

        with (self.index_dir / "chunks.json").open("r", encoding="utf-8") as f:
            self.chunks: list[dict] = json.load(f)

        with (self.index_dir / "bm25.pkl").open("rb") as f:
            bm25_blob = pickle.load(f)
        self.bm25 = bm25_blob["bm25"]

        self.faiss_index = faiss.read_index(str(self.index_dir / "faiss.index"))
        self.model = SentenceTransformer(self.model_name)

    def _bm25_scores(self, query: str) -> np.ndarray:
        tokens = _segment(query).split()
        return np.asarray(self.bm25.get_scores(tokens), dtype=np.float32)

    def _faiss_scores(self, query: str) -> np.ndarray:
        """Cosine similarity to every chunk (IndexFlatIP + normalized vectors)."""
        q = self.model.encode(
            [_segment(query)],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        n = self.faiss_index.ntotal
        sims, idx = self.faiss_index.search(q, n)
        scores = np.zeros(n, dtype=np.float32)
        scores[idx[0]] = sims[0]
        return scores

    def search(self, query: str, top_k: int = 5) -> list[Hit]:
        bm25_raw = self._bm25_scores(query)
        faiss_raw = self._faiss_scores(query)

        bm25_norm = _minmax(bm25_raw)
        faiss_norm = _minmax(faiss_raw)

        fused = self.alpha * bm25_norm + (1.0 - self.alpha) * faiss_norm
        top_idx = np.argsort(-fused)[:top_k]

        return [
            Hit(
                chunk=self.chunks[i],
                score=float(fused[i]),
                bm25_score=float(bm25_raw[i]),
                faiss_score=float(faiss_raw[i]),
            )
            for i in top_idx
        ]
