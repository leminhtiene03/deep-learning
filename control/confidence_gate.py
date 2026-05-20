# control/confidence_gate.py

from typing import Dict, Any, List

from control.text_utils import infer_topic


def generate_teacher_questions(question: str, topic: str) -> List[str]:
    """
    Sinh câu hỏi phụ để hỏi thầy cô/admin.
    Bản đầu dùng template đơn giản.
    """

    questions = [
        f"Câu hỏi gốc của sinh viên: {question}",
        "Quy định chính xác áp dụng cho trường hợp này là gì?",
        "Có điều kiện hoặc ngoại lệ nào cần lưu ý không?",
        "Đơn vị nào có thẩm quyền xác nhận hoặc xử lý trường hợp này?"
    ]

    if topic == "dang_ky_hoc_phan":
        questions.extend([
            "Sinh viên có được đăng ký khác với kế hoạch học tập chuẩn không?",
            "Có giới hạn số tín chỉ tối thiểu hoặc tối đa trong học kỳ không?",
            "Có cần cố vấn học tập hoặc phòng đào tạo phê duyệt không?"
        ])

    elif topic == "hoc_phi":
        questions.extend([
            "Sinh viên có được hoãn hoặc chậm nộp học phí trong trường hợp đặc biệt không?",
            "Nếu được thì cần làm đơn hoặc liên hệ đơn vị nào?",
            "Thời hạn xử lý hoặc phản hồi là bao lâu?"
        ])

    elif topic == "tot_nghiep":
        questions.extend([
            "Điều kiện chính xác để được xét tốt nghiệp là gì?",
            "Thời hạn cấp bằng hoặc giấy chứng nhận tạm thời là bao lâu?"
        ])

    return questions


def decide_action(
    question: str,
    context_check: Dict[str, Any],
    answer_check: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Quyết định cuối cùng:
    - answer_directly
    - answer_with_caution
    - ask_teacher
    """

    topic = infer_topic(question)

    context_status = context_check.get("context_status")
    answer_status = answer_check.get("answer_status")

    context_score = float(context_check.get("context_score", 0.0))
    answer_score = float(answer_check.get("answer_score", 0.0))

    reasons = []
    reasons.extend(context_check.get("reasons", []))
    reasons.extend(answer_check.get("reasons", []))

    final_score = 0.55 * context_score + 0.45 * answer_score

    if context_status in {"no_context", "weak_context"}:
        decision = "ask_teacher"
        confidence = "low"

    elif answer_status == "weak_answer":
        decision = "ask_teacher"
        confidence = "low"

    elif context_status == "partial_context" or answer_status == "partial_answer":
        decision = "answer_with_caution"
        confidence = "medium"

    elif final_score >= 0.50:
        decision = "answer_directly"
        confidence = "high"

    else:
        decision = "answer_with_caution"
        confidence = "medium"

    teacher_questions = []

    if decision == "ask_teacher":
        teacher_questions = generate_teacher_questions(question, topic)

    return {
        "decision": decision,
        "confidence": confidence,
        "final_score": round(final_score, 4),
        "topic": topic,
        "reasons": reasons,
        "teacher_questions": teacher_questions
    }


def build_user_response(answer: str, gate_result: Dict[str, Any]) -> str:
    """
    Tạo câu trả lời cuối cho user dựa trên decision.
    """

    decision = gate_result.get("decision")

    if decision == "answer_directly":
        return answer

    if decision == "answer_with_caution":
        return (
            "Theo ngữ cảnh hiện có, mình có thể trả lời như sau:\n\n"
            + answer
            + "\n\nLưu ý: câu trả lời này nên được kiểm tra thêm nếu trường hợp của bạn có chi tiết đặc biệt."
        )

    if decision == "ask_teacher":
        return (
            "Mình chưa có đủ thông tin chắc chắn từ ngữ cảnh hiện tại để trả lời chính xác. "
            "Câu hỏi này nên được chuyển để thầy cô hoặc đơn vị phụ trách xác minh thêm."
        )

    return answer