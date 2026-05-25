"""Prepend structural metadata (Chương, Điều, Khoản) to chunk text for retrieval."""

from __future__ import annotations

import copy
import re
from typing import Any, Dict, List, Optional


def _normalize_label(value: str, kind: str) -> str:
    value = str(value or "").strip()
    if not value:
        return ""

    low = value.lower()
    if kind == "chapter":
        if low.startswith("chương"):
            return value
        return f"Chương {value}"
    if kind == "article":
        if "điều" in low:
            return value
        return f"Điều {value}"
    if kind == "section":
        if "khoản" in low:
            return value
        return f"Khoản {value}"
    return value


def build_context_prefix(metadata: Optional[Dict[str, Any]]) -> str:
    """Build a short header from chapter / article / section / title metadata."""
    meta = metadata or {}

    parts: List[str] = []

    chapter = _normalize_label(meta.get("chapter", ""), "chapter")
    article = _normalize_label(meta.get("article", ""), "article")
    section = _normalize_label(meta.get("section", ""), "section")
    title = str(meta.get("title", "") or "").strip()

    for label in (chapter, article, section):
        if label and label not in parts:
            parts.append(label)

    if title and title not in parts:
        # Skip title if it duplicates article heading text
        title_low = title.lower()
        if not any(title_low in p.lower() or p.lower() in title_low for p in parts):
            parts.append(title)

    return " | ".join(parts)


def _content_already_has_prefix(content: str, prefix: str) -> bool:
    if not prefix:
        return True

    head = content.strip()[: max(len(prefix) + 40, 80)].lower()
    prefix_low = prefix.lower()

    if head.startswith(prefix_low):
        return True

    # Chunk may start with "Điều N." while prefix is "Điều N | Title"
    first_segment = prefix_low.split("|")[0].strip()
    return bool(first_segment) and head.startswith(first_segment)


def prepend_context_to_chunk(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a copy of chunk with metadata context prepended to content.
    Original content is preserved in metadata for traceability.
    """
    meta = chunk.get("metadata", {}) or {}
    prefix = build_context_prefix(meta)
    content = str(chunk.get("content", "") or "")

    if not prefix or _content_already_has_prefix(content, prefix):
        return chunk

    updated = copy.deepcopy(chunk)
    updated_meta = updated.get("metadata", {}) or {}
    if "raw_content" not in updated_meta:
        updated_meta["raw_content"] = content
    updated["metadata"] = updated_meta
    updated["content"] = f"{prefix}\n{content}".strip()

    updated["char_count"] = len(updated["content"])
    updated["word_count"] = len(updated["content"].split())

    return updated


def apply_context_to_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [prepend_context_to_chunk(c) for c in chunks]


def extract_article_number(article_label: str) -> Optional[str]:
    m = re.search(r"(\d+)", str(article_label or ""))
    return m.group(1) if m else None
