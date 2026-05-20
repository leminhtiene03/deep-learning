# control/text_utils.py

import re
from typing import Set


STOPWORDS = {
    "là", "và", "của", "có", "cho", "các", "với", "trong", "theo",
    "được", "không", "phải", "này", "đó", "thì", "về", "để", "khi",
    "em", "anh", "chị", "bạn", "mình", "ạ", "nhé", "nào", "gì",
    "sinh", "viên", "học", "viện", "quy", "định", "điều", "chương",
    "người", "theo", "trường", "hợp", "nếu", "thì", "sẽ", "cần"
}


def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = text.replace("học kì", "học kỳ")
    text = text.replace("cntt", "công nghệ thông tin")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_keywords(text: str, min_len: int = 3) -> Set[str]:
    text = normalize_text(text)

    words = re.findall(r"[a-zA-ZÀ-ỹ0-9]+", text)

    keywords = set()
    for w in words:
        if len(w) < min_len:
            continue
        if w in STOPWORDS:
            continue
        keywords.add(w)

    return keywords


def keyword_overlap_score(a: str, b: str) -> float:
    a_keywords = extract_keywords(a)
    b_text = normalize_text(b)

    if not a_keywords:
        return 0.0

    hit = 0
    for kw in a_keywords:
        if kw in b_text:
            hit += 1

    return hit / max(len(a_keywords), 1)

def is_permission_question(question: str) -> bool:
    q = normalize_text(question)

    patterns = [
        "có được",
        "được phép",
        "có thể",
        "được không",
        "có bắt buộc",
        "bắt buộc không",
        "có cần",
        "phải không",
        "được học ít hơn",
        "học ít hơn",
        "đăng ký ít hơn",
        "ít hơn số tín chỉ",
        "khác với kế hoạch",
        "giảm số tín chỉ"
    ]

    return any(p in q for p in patterns)


def is_curriculum_table_context(context: str) -> bool:
    c = normalize_text(context)

    signals = [
        "kế hoạch và tiến trình học tập chuẩn",
        "danh sách môn học",
        "học kỳ:",
        "tổng số tín chỉ",
        "tc"
    ]

    hit = sum(1 for s in signals if s in c)

    return hit >= 3


def has_policy_evidence(context: str) -> bool:
    c = normalize_text(context)

    evidence_patterns = [
        "đăng ký tối thiểu",
        "đăng ký tối đa",
        "số tín chỉ tối thiểu",
        "số tín chỉ tối đa",
        "ngoại lệ",
        "cố vấn học tập",
        "xem xét",
    ]

    return any(p in c for p in evidence_patterns)
def infer_topic(question: str) -> str:
    q = normalize_text(question)

    if any(x in q for x in ["học phí", "nộp phí", "thu phí", "công nợ"]):
        return "hoc_phi"

    if any(x in q for x in ["tín chỉ", "đăng ký học phần", "môn học", "học kỳ", "học kì", "hk"]):
        return "dang_ky_hoc_phan"

    if any(x in q for x in ["thực tập", "thực tập cuối khóa"]):
        return "thuc_tap"

    if any(x in q for x in ["tốt nghiệp", "bằng tốt nghiệp", "xét tốt nghiệp"]):
        return "tot_nghiep"

    if any(x in q for x in ["nghiên cứu khoa học", "đạo văn", "bản quyền"]):
        return "nghien_cuu_khoa_hoc"

    if any(x in q for x in ["kỷ luật", "đình chỉ", "cảnh cáo"]):
        return "ky_luat"

    return "general"