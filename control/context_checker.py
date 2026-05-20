# control/context_checker.py

from typing import Dict, Any, Optional

from control.text_utils import (
    keyword_overlap_score,
    normalize_text,
    is_permission_question,
    is_curriculum_table_context,
    has_policy_evidence
)

def check_context(
    question: str,
    chunk: Optional[Dict[str, Any]],
    retrieval_score: Optional[float] = None,
    bm25_rank: Optional[int] = None,
    faiss_rank: Optional[int] = None
) -> Dict[str, Any]:
    """
    Kiểm tra context top 1 có đủ liên quan để đưa cho model sinh answer không.
    """

    if chunk is None:
        return {
            "context_status": "no_context",
            "context_score": 0.0,
            "reasons": ["Không có chunk phù hợp."]
        }

    content = chunk.get("content", "") or ""
    metadata = chunk.get("metadata", {}) or {}
    title = metadata.get("title", "") or ""

    reasons = []

    if len(content.strip()) < 50:
        reasons.append("Context quá ngắn.")

    question_context_overlap = keyword_overlap_score(question, title + "\n" + content)

    if question_context_overlap < 0.2:
        reasons.append("Ít từ khóa quan trọng của câu hỏi xuất hiện trong context.")

    if bm25_rank is not None and faiss_rank is not None:
        if bm25_rank <= 5 and faiss_rank <= 5:
            rank_agreement = 1.0
        elif bm25_rank <= 10 or faiss_rank <= 10:
            rank_agreement = 0.6
        else:
            rank_agreement = 0.3
    else:
        rank_agreement = 0.5

    # retrieval_score RRF thường nhỏ, không nên dùng tuyệt đối quá mạnh.
    # Tạm ưu tiên keyword overlap + rank agreement.
    context_score = 0.7 * question_context_overlap + 0.3 * rank_agreement

    q_norm = normalize_text(question)
    c_norm = normalize_text(title + "\n" + content)
    # Nếu câu hỏi hỏi quyền/được phép/ngoại lệ,
    # nhưng context chỉ là bảng kế hoạch học tập và không có căn cứ quy định,
    # thì không được xem là strong_context.
    if is_permission_question(question):
        full_context = title + "\n" + content

        if is_curriculum_table_context(full_context) and not has_policy_evidence(full_context):
            reasons.append(
                "Câu hỏi hỏi về quyền/được phép/ngoại lệ, nhưng context chỉ là bảng kế hoạch học tập, không có căn cứ quy định."
            )
            context_score = min(context_score, 0.30)

    # Một số kiểm tra logic đơn giản
    if "công nghệ thông tin" in q_norm and "công nghệ thông tin" not in c_norm:
        reasons.append("Câu hỏi hỏi ngành Công nghệ thông tin nhưng context không chứa ngành này.")
        context_score -= 0.2

    if ("hk1" in q_norm or "học kỳ 1" in q_norm) and ("hk1" not in c_norm and "học kỳ: hk1" not in c_norm):
        reasons.append("Câu hỏi hỏi HK1 nhưng context không thể hiện rõ HK1.")
        context_score -= 0.2

    context_score = max(0.0, min(1.0, context_score))

    if context_score >= 0.55:
        status = "strong_context"
    elif context_score >= 0.35:
        status = "partial_context"
    else:
        status = "weak_context"

    return {
        "context_status": status,
        "context_score": round(context_score, 4),
        "question_context_overlap": round(question_context_overlap, 4),
        "rank_agreement": rank_agreement,
        "reasons": reasons,
        "chunk_id": chunk.get("chunk_id"),
        "page": chunk.get("page"),
        "title": title
    }