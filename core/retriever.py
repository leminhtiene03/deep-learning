# core/retriever.py
# Hybrid Retriever using RRF instead of raw weighted-sum BM25 + FAISS

import os
import sys
import json
import pickle
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from pathlib import Path

import numpy as np
import faiss
from pyvi import ViTokenizer
from sentence_transformers import SentenceTransformer

# Add project root to path for config import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import INDEX_DIR

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


DEFAULT_INDEX_DIR = str(INDEX_DIR)
DEFAULT_MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder"


DOMAIN_STOPWORDS = {
    "học", "viện", "sinh", "viên", "quy", "định", "điều", "chương",
    "các", "của", "và", "là", "có", "được", "không", "trong", "theo",
    "với", "cho", "về", "khi", "này", "đó", "em", "anh", "chị", "bạn",
    "mình", "ạ", "gì", "nào", "những", "nào", "phải"
}


@dataclass
class Hit:
    chunk: Dict[str, Any]
    score: float
    rrf_score: float
    bm25_rank: Optional[int]
    faiss_rank: Optional[int]
    bm25_score: float
    faiss_score: float


def segment_text(text: str, remove_stopwords: bool = False) -> List[str]:
    text = str(text).lower().strip()
    segmented = ViTokenizer.tokenize(text)
    tokens = segmented.split()

    if remove_stopwords:
        tokens = [
            t for t in tokens
            if t not in DOMAIN_STOPWORDS and len(t) >= 2
        ]

    return tokens


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class RRFHybridRetriever:
    def __init__(
        self,
        index_dir: str = DEFAULT_INDEX_DIR,
        model_name: str = DEFAULT_MODEL_NAME,
        bm25_weight: float = 0.30,
        faiss_weight: float = 0.70,
        rrf_k: int = 60,
        remove_bm25_stopwords: bool = True
    ):
        self.index_dir = index_dir
        self.model_name = model_name
        self.bm25_weight = bm25_weight
        self.faiss_weight = faiss_weight
        self.rrf_k = rrf_k
        self.remove_bm25_stopwords = remove_bm25_stopwords

        self.chunks_path = os.path.join(index_dir, "chunks.json")
        self.bm25_path = os.path.join(index_dir, "bm25.pkl")
        self.faiss_path = os.path.join(index_dir, "faiss.index")
        self.meta_path = os.path.join(index_dir, "meta.json")

        self._load()

    def _load(self):
        if not os.path.exists(self.chunks_path):
            raise FileNotFoundError(f"Missing {self.chunks_path}")

        if not os.path.exists(self.bm25_path):
            raise FileNotFoundError(f"Missing {self.bm25_path}")

        if not os.path.exists(self.faiss_path):
            raise FileNotFoundError(f"Missing {self.faiss_path}")

        self.chunks = load_json(self.chunks_path)

        with open(self.bm25_path, "rb") as f:
            bm25_obj = pickle.load(f)

        if isinstance(bm25_obj, dict):
            print("BM25 pickle is dict. Keys:", bm25_obj.keys())

            if "bm25" in bm25_obj:
                self.bm25 = bm25_obj["bm25"]
            else:
                raise ValueError(
                    f"Không tìm thấy key 'bm25' trong bm25.pkl. Keys hiện có: {list(bm25_obj.keys())}"
                )
        else:
            self.bm25 = bm25_obj

        if not hasattr(self.bm25, "get_scores"):
            raise TypeError(
                f"BM25 object không có get_scores(). Type hiện tại: {type(self.bm25)}"
            )

        self.faiss_index = faiss.read_index(self.faiss_path)

        if os.path.exists(self.meta_path):
            try:
                self.meta = load_json(self.meta_path)
            except Exception:
                self.meta = {}
        else:
            self.meta = {}

        print("Loading embedding model:", self.model_name)
        self.encoder = SentenceTransformer(self.model_name)

        print("Chunks:", len(self.chunks))
        print("FAISS ntotal:", self.faiss_index.ntotal)
        print("BM25 weight:", self.bm25_weight)
        print("FAISS weight:", self.faiss_weight)
        print("RRF k:", self.rrf_k)

    def _search_bm25(self, query: str, candidate_k: int):
        tokens = segment_text(
            query,
            remove_stopwords=self.remove_bm25_stopwords
        )

        if len(tokens) == 0:
            return [], {}

        scores = self.bm25.get_scores(tokens)
        scores = np.asarray(scores, dtype=np.float32)

        top_idx = np.argsort(scores)[::-1][:candidate_k]

        results = []
        raw_scores = {}

        for idx in top_idx:
            idx = int(idx)
            score = float(scores[idx])

            if score <= 0:
                continue

            results.append(idx)
            raw_scores[idx] = score

        return results, raw_scores

    def _search_faiss(self, query: str, candidate_k: int):
        query_emb = self.encoder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        scores, indices = self.faiss_index.search(query_emb, candidate_k)

        results = []
        raw_scores = {}

        for idx, score in zip(indices[0], scores[0]):
            idx = int(idx)

            if idx < 0:
                continue

            results.append(idx)
            raw_scores[idx] = float(score)

        return results, raw_scores

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 30
    ) -> List[Hit]:
        candidate_k = max(candidate_k, top_k * 5)

        bm25_ids, bm25_raw = self._search_bm25(query, candidate_k)
        faiss_ids, faiss_raw = self._search_faiss(query, candidate_k)

        rrf_scores = {}
        bm25_rank = {}
        faiss_rank = {}

        # BM25 RRF
        for rank, idx in enumerate(bm25_ids, start=1):
            bm25_rank[idx] = rank
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (
                self.bm25_weight / (self.rrf_k + rank)
            )

        # FAISS RRF
        for rank, idx in enumerate(faiss_ids, start=1):
            faiss_rank[idx] = rank
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (
                self.faiss_weight / (self.rrf_k + rank)
            )

        ranked = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        hits = []

        for idx, score in ranked:
            chunk = self.chunks[idx]

            hits.append(
                Hit(
                    chunk=chunk,
                    score=float(score),
                    rrf_score=float(score),
                    bm25_rank=bm25_rank.get(idx),
                    faiss_rank=faiss_rank.get(idx),
                    bm25_score=float(bm25_raw.get(idx, 0.0)),
                    faiss_score=float(faiss_raw.get(idx, 0.0))
                )
            )

        return hits


def print_hits(hits: List[Hit]):
    for i, h in enumerate(hits, start=1):
        chunk = h.chunk
        metadata = chunk.get("metadata", {}) or {}

        title = metadata.get("title", "")
        page = chunk.get("page", "")
        chunk_id = chunk.get("chunk_id", "")

        print("\n" + "=" * 80)
        print(f"[{i}] chunk_id={chunk_id} page={page}")
        print(f"RRF={h.rrf_score:.6f}")
        print(f"BM25 rank={h.bm25_rank} score={h.bm25_score:.4f}")
        print(f"FAISS rank={h.faiss_rank} score={h.faiss_score:.4f}")

        if title:
            print("Title:", title)

        print("-" * 80)
        print(chunk.get("content", "")[:1200])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str)
    parser.add_argument("-k", "--top_k", type=int, default=5)
    parser.add_argument("-c", "--candidate_k", type=int, default=30)
    parser.add_argument("--bm25_weight", type=float, default=0.30)
    parser.add_argument("--faiss_weight", type=float, default=0.70)
    parser.add_argument("--rrf_k", type=int, default=60)
    parser.add_argument("--no_stopwords", action="store_true")

    args = parser.parse_args()

    retriever = RRFHybridRetriever(
        bm25_weight=args.bm25_weight,
        faiss_weight=args.faiss_weight,
        rrf_k=args.rrf_k,
        remove_bm25_stopwords=not args.no_stopwords
    )

    hits = retriever.search(
        args.query,
        top_k=args.top_k,
        candidate_k=args.candidate_k
    )

    print_hits(hits)