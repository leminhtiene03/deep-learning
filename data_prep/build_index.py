"""Build BM25 + FAISS hybrid index over chunks_new.json.

Outputs (all under ./index/):
    bm25.pkl          - rank_bm25.BM25Okapi over pyvi-segmented chunks
    faiss.index       - FAISS IndexFlatIP over L2-normalized embeddings
    embeddings.npy    - chunk embeddings (float32, [N, dim])
    chunks.json       - the original chunk records, kept alongside the index
    meta.json         - model name, dim, N, etc.
"""

from __future__ import annotations

import json
import pickle
import re
import sys
from pathlib import Path

import faiss
import numpy as np
from pyvi import ViTokenizer
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# Add project root to path for config import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CHUNKS_TABLE_AWARE_PATH, INDEX_DIR as CONFIG_INDEX_DIR

CHUNKS_PATH = Path(CHUNKS_TABLE_AWARE_PATH)
INDEX_DIR = Path(CONFIG_INDEX_DIR)
MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder"


def segment(text: str) -> str:
    """Vietnamese word segmentation -> single string with underscores in compounds."""
    text = re.sub(r"\s+", " ", text).strip()
    return ViTokenizer.tokenize(text.lower())


def tokenize_for_bm25(text: str) -> list[str]:
    return segment(text).split()


def main() -> None:
    INDEX_DIR.mkdir(exist_ok=True)

    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"[load] {len(chunks)} chunks")

    # BM25 over segmented tokens
    print("[bm25] segmenting + tokenizing ...")
    corpus_tokens = [tokenize_for_bm25(c["content"]) for c in chunks]
    bm25 = BM25Okapi(corpus_tokens)
    with (INDEX_DIR / "bm25.pkl").open("wb") as f:
        pickle.dump({"bm25": bm25, "corpus_tokens": corpus_tokens}, f)
    print(f"[bm25] saved -> {INDEX_DIR / 'bm25.pkl'}")

    # Dense embeddings. The bkai model was trained on pyvi-segmented input.
    print(f"[embed] loading {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)
    segmented_texts = [segment(c["content"]) for c in chunks]
    print("[embed] encoding chunks ...")
    embeddings = model.encode(
        segmented_texts,
        batch_size=16,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")
    np.save(INDEX_DIR / "embeddings.npy", embeddings)
    print(f"[embed] embeddings shape={embeddings.shape}")

    # FAISS: inner product on L2-normalized vectors = cosine similarity
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))
    print(f"[faiss] saved -> {INDEX_DIR / 'faiss.index'} (ntotal={index.ntotal})")

    # Stash the chunks alongside the index so the retriever is self-contained
    with (INDEX_DIR / "chunks.json").open("w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)

    with (INDEX_DIR / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(
            {"model": MODEL_NAME, "dim": dim, "num_chunks": len(chunks)},
            f,
            ensure_ascii=False,
            indent=2,
        )
    print("[done]")


if __name__ == "__main__":
    main()
