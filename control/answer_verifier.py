# control/answer_verifier.py

from typing import Dict, Any

from control.text_utils import (
    keyword_overlap_score,
    extract_keywords,
    normalize_text,
    is_permission_question,
    is_curriculum_table_context,
    has_policy_evidence
)

RISKY_PHRASES = [
    "liên hệ phòng đào tạo",
    "liên hệ phòng tài chính",
    "liên hệ cố vấn",
    "được phép",
    "không được phép",
    "bắt buộc",
    "chắc chắn",
    "luôn luôn",
    "tất cả trường hợp"
]


def verify_answer(
    question: str,
    context: str,
    answer: str
) -> Dict[str, Any]:
    """
    Kiểm tra answer có bám context và trả lời đúng trọng tâm câu hỏi không.
    Đây là bản heuristic nhẹ, chưa dùng model reranker.
    """

    reasons = []

    if not answer or len(answer.strip()) < 10:
        return {
            "answer_status": "bad_answer",
            "answer_score": 0.0,
            "reasons": ["Answer rỗng hoặc quá ngắn."]
        }

    q_answer_overlap = keyword_overlap_score(question, answer)
    answer_context_overlap = keyword_overlap_score(answer, context)

    q_norm = normalize_text(question)
    c_norm = normalize_text(context)
    a_norm = normalize_text(answer)
    # Nếu câu hỏi hỏi "có được / được phép / có thể không",
    # nhưng context không có căn cứ quy định, thì answer không được kết luận chắc.
    if is_permission_question(question):
        if is_curriculum_table_context(context) and not has_policy_evidence(context):
            if any(x in a_norm for x in ["không", "được phép", "không được", "có thể", "bắt buộc"]):
                reasons.append(
                    "Answer đưa ra kết luận về việc được/không được phép, nhưng context chỉ là bảng kế hoạch học tập."
                )
                answer_score = 0.20
    # Nếu answer có cụm mang tính kết luận mạnh nhưng context không có căn cứ tương ứng
    unsupported_risky = []

    for phrase in RISKY_PHRASES:
        if phrase in a_norm and phrase not in c_norm:
            unsupported_risky.append(phrase)

    if unsupported_risky:
        reasons.append(
            "Answer có cụm kết luận/tư vấn không thấy rõ trong context: "
            + ", ".join(unsupported_risky)
        )

    # Kiểm tra nhầm khái niệm đơn giản
    if "tín chỉ" in q_norm and "tập hợp các hoạt động giảng dạy" in a_norm:
        reasons.append("Có dấu hiệu nhầm định nghĩa tín chỉ với định nghĩa học phần.")

    if "mã số" in q_norm and "điều kiện tiên quyết" in a_norm:
        reasons.append("Có dấu hiệu nhầm mã số học phần với điều kiện tiên quyết.")

    # Nếu answer dài nhưng overlap context quá thấp thì có nguy cơ hallucination
    if answer_context_overlap < 0.25:
        reasons.append("Answer có ít từ khóa được hỗ trợ bởi context.")

    # Nếu answer không liên quan nhiều tới question
    if q_answer_overlap < 0.15:
        reasons.append("Answer có ít từ khóa khớp với câu hỏi.")

    # Score tổng hợp
    answer_score = 0.65 * answer_context_overlap + 0.35 * q_answer_overlap

    if unsupported_risky:
        answer_score -= 0.15

    if reasons:
        answer_score -= 0.05

    answer_score = max(0.0, min(1.0, answer_score))

    if answer_score >= 0.45 and not unsupported_risky:
        status = "supported_answer"
    elif answer_score >= 0.30:
        status = "partial_answer"
    else:
        status = "weak_answer"

    return {
        "answer_status": status,
        "answer_score": round(answer_score, 4),
        "answer_context_overlap": round(answer_context_overlap, 4),
        "question_answer_overlap": round(q_answer_overlap, 4),
        "unsupported_risky_phrases": unsupported_risky,
        "reasons": reasons
    }