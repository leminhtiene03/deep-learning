"""Interactive CLI for the hybrid retriever.

    python search.py                          # interactive prompt
    python search.py "câu hỏi của bạn" -k 5   # one-shot
"""

from __future__ import annotations

import argparse
import sys

# Force UTF-8 stdout so Vietnamese prints on the Windows console.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from retriever import HybridRetriever


def _print_hits(query: str, hits) -> None:
    print(f"\n=== Query: {query} ===")
    for rank, h in enumerate(hits, 1):
        c = h.chunk
        title = (c.get("metadata") or {}).get("title", "")
        preview = c["content"].replace("\n", " ")
        print(
            f"\n[{rank}] chunk_id={c['chunk_id']}  "
            f"score={h.score:.4f}  bm25={h.bm25_score:.3f}  faiss={h.faiss_score:.3f}"
        )
        if title:
            print(f"     title: {title}")
        print(f"     {preview}{'...' if len(c['content']) > 220 else ''}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("query", nargs="?", help="Câu hỏi cần truy vấn")
    p.add_argument("-k", "--top-k", type=int, default=5)
    p.add_argument(
        "-a",
        "--alpha",
        type=float,
        default=0.5,
        help="BM25 weight in fused score; FAISS gets (1 - alpha). Default 0.5",
    )
    args = p.parse_args()

    print("Loading hybrid retriever ...")
    r = HybridRetriever(alpha=args.alpha)
    print(f"Ready. Index has {len(r.chunks)} chunks. alpha={args.alpha}")

    if args.query:
        _print_hits(args.query, r.search(args.query, top_k=args.top_k))
        return 0

    while True:
        try:
            q = input("\n? ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not q:
            continue
        if q.lower() in {"exit", "quit", ":q"}:
            return 0
        _print_hits(q, r.search(q, top_k=args.top_k))


if __name__ == "__main__":
    sys.exit(main())
