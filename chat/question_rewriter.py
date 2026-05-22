"""
Module viết lại câu hỏi (Query Rewriting) cho hệ thống RAG chatbot.
Sử dụng LLM để chuyển đổi câu hỏi multi-turn thành câu truy vấn độc lập.
"""

from typing import List, Dict, Optional
import logging
import torch

logger = logging.getLogger(__name__)


# System prompt tối ưu cho tiếng Việt
REWRITE_SYSTEM_PROMPT = """Bạn là trợ lý AI chuyên viết lại câu hỏi cho hệ thống tìm kiếm.

NHIỆM VỤ: Đọc lịch sử hội thoại và câu hỏi mới nhất, sau đó viết lại thành một câu truy vấn hoàn chỉnh, độc lập.

QUY TẮC BẮT BUỘC:
1. Thay thế TẤT CẢ đại từ phiếm chỉ (nó, đó, vậy, thế, họ, ông ấy, cô ấy...) bằng danh từ cụ thể từ ngữ cảnh
2. Bổ sung đầy đủ chủ ngữ, vị ngữ và từ khóa quan trọng
3. Giữ nguyên ý định tìm kiếm của người dùng
4. CHỈ trả về câu hỏi đã viết lại, KHÔNG giải thích, KHÔNG thêm lời mở đầu
5. Nếu câu hỏi đã rõ ràng và độc lập, giữ nguyên

VÍ DỤ:
Lịch sử: "Ngành Công nghệ thông tin học những gì?"
Câu hỏi mới: "Vậy còn ngành khác thì sao?"
→ Viết lại: "Các ngành khác ngoài Công nghệ thông tin học những gì?"

Lịch sử: "Điểm chuẩn ngành Kỹ thuật phần mềm là bao nhiêu?"
Câu hỏi mới: "Giải thích chi tiết hơn"
→ Viết lại: "Giải thích chi tiết về điểm chuẩn ngành Kỹ thuật phần mềm"
"""


def format_history_for_prompt(history: List[Dict[str, str]], max_turns: int = 3) -> str:
    """
    Định dạng lịch sử chat thành chuỗi văn bản cho prompt.

    Args:
        history: Danh sách tin nhắn [{"role": "user/assistant", "content": "..."}]
        max_turns: Số lượng lượt hội thoại tối đa để đưa vào context

    Returns:
        Chuỗi lịch sử đã định dạng
    """
    if not history:
        return "Không có lịch sử hội thoại."

    # Lấy max_turns*2 tin nhắn gần nhất (mỗi turn có user + assistant)
    recent_history = history[-(max_turns * 2):]

    formatted_lines = []
    for msg in recent_history:
        role_label = "Người dùng" if msg["role"] == "user" else "Trợ lý"
        formatted_lines.append(f"{role_label}: {msg['content']}")

    return "\n".join(formatted_lines)


def rewrite_question(
    history: List[Dict[str, str]],
    current_query: str,
    llm_model,
    tokenizer,
    max_length: int = 256,
    temperature: float = 0.3
) -> str:
    """
    Viết lại câu hỏi dựa trên lịch sử hội thoại.

    Args:
        history: Lịch sử chat từ history_store
        current_query: Câu hỏi hiện tại của người dùng
        llm_model: Mô hình LLM (ví dụ: BARTpho)
        tokenizer: Tokenizer tương ứng
        max_length: Độ dài tối đa của câu trả lời
        temperature: Nhiệt độ sampling (thấp = ổn định hơn)

    Returns:
        Câu hỏi đã được viết lại
    """
    try:
        # Nếu không có lịch sử, trả về câu hỏi gốc
        if not history or len(history) == 0:
            logger.info("No history available, returning original query")
            return current_query

        # Định dạng lịch sử
        history_text = format_history_for_prompt(history, max_turns=3)

        # Tạo prompt
        user_prompt = f"""LỊCH SỬ HỘI THOẠI:
{history_text}

CÂU HỎI MỚI: {current_query}

VIẾT LẠI:"""

        # Kết hợp system prompt và user prompt
        full_prompt = f"{REWRITE_SYSTEM_PROMPT}\n\n{user_prompt}"

        # Tokenize
        inputs = tokenizer(
            full_prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )

        # Chuyển sang GPU nếu có
        device = next(llm_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = llm_model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                num_beams=1,
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # Decode
        rewritten = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Làm sạch output (loại bỏ prompt nếu model echo lại)
        rewritten = rewritten.strip()

        # Nếu model trả về cả prompt, chỉ lấy phần sau "VIẾT LẠI:"
        if "VIẾT LẠI:" in rewritten:
            rewritten = rewritten.split("VIẾT LẠI:")[-1].strip()

        # Kiểm tra output hợp lệ
        if not rewritten or len(rewritten) < 5:
            logger.warning("Rewritten query too short, using original")
            return current_query

        logger.info(f"Original: {current_query}")
        logger.info(f"Rewritten: {rewritten}")

        return rewritten

    except Exception as e:
        logger.error(f"Error in query rewriting: {e}", exc_info=True)
        # Fallback: trả về câu hỏi gốc nếu có lỗi
        return current_query


def rewrite_question_simple(
    history: List[Dict[str, str]],
    current_query: str
) -> str:
    """
    Phiên bản đơn giản của query rewriting không cần LLM.
    Sử dụng rule-based approach để xử lý các trường hợp cơ bản.

    Args:
        history: Lịch sử chat
        current_query: Câu hỏi hiện tại

    Returns:
        Câu hỏi đã được viết lại (hoặc gốc nếu không cần)
    """
    if not history:
        return current_query

    query_lower = current_query.lower().strip()

    # Các từ khóa chỉ sự tham chiếu
    reference_keywords = [
        "vậy", "thế", "còn", "nó", "đó", "ấy", "họ",
        "chi tiết hơn", "giải thích thêm", "cụ thể hơn"
    ]

    # Kiểm tra xem có chứa từ tham chiếu không
    has_reference = any(keyword in query_lower for keyword in reference_keywords)

    if not has_reference:
        return current_query

    # Lấy chủ đề từ câu hỏi user gần nhất
    last_user_msg = None
    for msg in reversed(history):
        if msg["role"] == "user":
            last_user_msg = msg["content"]
            break

    if not last_user_msg:
        return current_query

    # Kết hợp context đơn giản
    if query_lower.startswith(("vậy", "thế", "còn")):
        rewritten = f"{last_user_msg} {current_query}"
    elif "chi tiết" in query_lower or "giải thích" in query_lower:
        rewritten = f"Giải thích chi tiết về {last_user_msg}"
    else:
        rewritten = current_query

    logger.info(f"Simple rewrite: {current_query} -> {rewritten}")
    return rewritten
