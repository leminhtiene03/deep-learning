# control/extractive_answer.py
# Trả lời extractive (giữ nguyên nội dung chunk) khi cần liệt kê đầy đủ.

from __future__ import annotations

import re
from typing import Optional

from control.query_constraints import LegalCitation, parse_legal_citation, normalize_vietnamese


# Câu hỏi cần trả lời đủ các ý/khoản/điểm, không tóm tắt
ENUMERATION_QUESTION_PATTERNS = [
    r"quy định gì",
    r"nội dung",
    r"các ý",
    r"những ý",
    r"bao nhiêu",
    r"liệt kê",
    r"gồm những",
    r"gồm các",
    r"tất cả",
    r"đầy đủ",
    r"toàn bộ",
    r"chi tiết",
    r"có những gì",
    r"là gì",
    r"như thế nào",
    r"ra sao",
    r"thế nào",
    r"nêu",
    r"trình bày",
]

DIEU_ONLY_RE = re.compile(
    r"điều\s+(\d+)",
    re.IGNORECASE | re.UNICODE,
)

KHOAN_SECTION_RE = re.compile(
    r"(khoản\s+\d+[^\n]*(?:\n(?!\s*khoản\s+\d+).*)*)",
    re.IGNORECASE | re.UNICODE,
)

DIEM_LINE_RE = re.compile(
    r"điểm\s+[a-zđ]\)",
    re.IGNORECASE | re.UNICODE,
)

NUMBERED_LINE_RE = re.compile(
    r"(?:^|\n)\s*\d+\.\s+",
    re.MULTILINE,
)


def count_structured_items(text: str) -> int:
    """Đếm số mục có cấu trúc (điểm/khoản/dòng đánh số) trong văn bản."""
    t = normalize_vietnamese(text)
    diem = len(DIEM_LINE_RE.findall(t))
    khoan = len(re.findall(r"khoản\s+\d+", t))
    numbered = len(NUMBERED_LINE_RE.findall(text))
    return max(diem, khoan, numbered)


def is_enumeration_question(question: str) -> bool:
    q = normalize_vietnamese(question)
    if any(re.search(p, q) for p in ENUMERATION_QUESTION_PATTERNS):
        return True
    # Hỏi cả điều mà không chỉ một điểm/khoản cụ thể
    if DIEU_ONLY_RE.search(q) and not re.search(
        r"điểm\s+[a-zđ]\s+khoản|khoản\s+\d+", q
    ):
        return True
    return False


def should_use_extractive_answer(question: str, context: str) -> bool:
    """
    Dùng trích xuất trực tiếp từ chunk khi câu hỏi cần liệt kê đầy đủ
    và context có nhiều mục có cấu trúc.
    """
    if not context or len(context.strip()) < 80:
        return False

    if not is_enumeration_question(question):
        return False

    return count_structured_items(context) >= 2


def _extract_khoan_block(context: str, khoan: str) -> Optional[str]:
    """Lấy đoạn bắt đầu từ 'khoản N' đến trước 'khoản N+1' (nếu có)."""
    pattern = re.compile(
        rf"(khoản\s+{re.escape(khoan)}\b.*?)(?=\nkhoản\s+\d+\b|$)",
        re.IGNORECASE | re.DOTALL | re.UNICODE,
    )
    m = pattern.search(context)
    if m:
        return m.group(1).strip()
    return None


def _extract_by_citation(context: str, citation: LegalCitation) -> Optional[str]:
    block = _extract_khoan_block(context, citation.khoan)
    if not block:
        return None

    if citation.diem:
        diem = citation.diem.lower()
        lines = block.split("\n")
        picked = []
        in_diem = False
        for line in lines:
            ln = normalize_vietnamese(line)
            if re.search(rf"điểm\s+{re.escape(diem)}\)", ln) or re.search(
                rf"^{re.escape(diem)}\)", ln
            ):
                in_diem = True
                picked.append(line)
                continue
            if in_diem and re.search(r"điểm\s+[a-zđ]\)", ln):
                break
            if in_diem:
                picked.append(line)
        if picked:
            return "\n".join(picked).strip()

    return block


def extract_passage_for_question(question: str, context: str) -> str:
    """
    Trả về đoạn context phù hợp câu hỏi.
    Mặc định: toàn bộ context (giữ đủ các ý).
    """
    context = (context or "").strip()
    if not context:
        return ""

    citation = parse_legal_citation(question)
    if citation:
        section = _extract_by_citation(context, citation)
        if section:
            return section

    return context


def format_extractive_answer(question: str, context: str) -> str:
    """Định dạng câu trả lời extractive cho user."""
    passage = extract_passage_for_question(question, context)
    if not passage:
        return ""

    citation = parse_legal_citation(question)
    if citation:
        header = f"Theo {citation.exact_phrase} trong quy định:\n\n"
    elif DIEU_ONLY_RE.search(normalize_vietnamese(question)):
        m = DIEU_ONLY_RE.search(normalize_vietnamese(question))
        header = f"Theo Điều {m.group(1)} trong quy định:\n\n"
    else:
        header = "Theo nội dung quy định trong tài liệu:\n\n"

    return header + passage
