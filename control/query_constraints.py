"""Parse legal citations in user queries (quy chế / sổ tay sinh viên)."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, Optional


# e.g. "điểm a khoản 3 điều 33", "điểm b khoản 1 điều 5"
LEGAL_CITATION_RE = re.compile(
    r"điểm\s+([a-zđ])\s+khoản\s+(\d+)\s+điều\s+(\d+)",
    re.IGNORECASE | re.UNICODE,
)

# Shorter form: "khoản 3 điều 33"
LEGAL_KHOAN_DIEU_RE = re.compile(
    r"khoản\s+(\d+)\s+điều\s+(\d+)",
    re.IGNORECASE | re.UNICODE,
)


@dataclass(frozen=True)
class LegalCitation:
    diem: Optional[str]
    khoan: str
    dieu: str

    @property
    def exact_phrase(self) -> str:
        if self.diem:
            return f"điểm {self.diem.lower()} khoản {self.khoan} điều {self.dieu}"
        return f"khoản {self.khoan} điều {self.dieu}"


def normalize_vietnamese(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text or ""))
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def parse_legal_citation(query: str) -> Optional[LegalCitation]:
    """Detect structured legal references in a user question."""
    q = normalize_vietnamese(query)

    m = LEGAL_CITATION_RE.search(q)
    if m:
        return LegalCitation(
            diem=m.group(1).lower(),
            khoan=m.group(2),
            dieu=m.group(3),
        )

    m = LEGAL_KHOAN_DIEU_RE.search(q)
    if m:
        return LegalCitation(
            diem=None,
            khoan=m.group(1),
            dieu=m.group(2),
        )

    return None


def _article_number_from_metadata(metadata: Dict[str, Any]) -> Optional[str]:
    article = str(metadata.get("article", "") or "")
    m = re.search(r"(\d+)", article)
    return m.group(1) if m else None


def chunk_matches_legal_citation(
    chunk: Dict[str, Any],
    citation: LegalCitation,
) -> bool:
    """
    True if chunk belongs to the cited Điều and contains the cited điểm/khoản.
    Uses metadata filter first, then normalized exact phrase in content.
    """
    metadata = chunk.get("metadata", {}) or {}
    content = normalize_vietnamese(chunk.get("content", ""))

    article_num = _article_number_from_metadata(metadata)
    if article_num and article_num != citation.dieu:
        return False

    if citation.exact_phrase in content:
        return True

    has_dieu = (
        f"điều {citation.dieu}" in content
        or f"điều {citation.dieu}." in content
    )
    has_khoan = (
        f"khoản {citation.khoan}" in content
        or re.search(rf"(?<!\d){citation.khoan}\.\s", content)
    )

    if not (has_dieu and has_khoan):
        return False

    if citation.diem:
        diem_patterns = [
            f"điểm {citation.diem})",
            f"điểm {citation.diem}.",
            f"{citation.diem})",
        ]
        if not any(p in content for p in diem_patterns):
            return False

    return True
