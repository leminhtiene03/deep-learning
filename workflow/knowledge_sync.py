# workflow/knowledge_sync.py

from typing import List, Dict, Any, Optional

from control.text_utils import keyword_overlap_score
from knowledge.store import (
    get_confirmed_answer_by_id,
    find_logs_that_may_need_correction,
    create_correction
)


def should_create_correction(
    log: Dict[str, Any],
    confirmed: Dict[str, Any],
    min_overlap: float = 0.20
) -> bool:
    """
    Quyết định answer log cũ có nên tạo correction hay không.

    Bản đầu dùng heuristic:
    - cùng topic
    - không phải câu trả lời lấy từ confirmed cache
    - chưa correction
    - câu hỏi cũ có overlap với câu hỏi confirmed
    - hoặc decision trước đó là ask_teacher / answer_with_caution
    """

    if not log:
        return False

    if not confirmed:
        return False

    if log.get("status") in {"needs_correction", "corrected"}:
        return False

    if log.get("decision") == "use_confirmed_cache":
        return False

    old_answer = (log.get("answer") or "").strip()
    new_answer = (confirmed.get("canonical_answer") or "").strip()

    if not old_answer or not new_answer:
        return False

    if old_answer == new_answer:
        return False

    log_question = log.get("question") or ""
    confirmed_question = confirmed.get("canonical_question") or ""

    overlap = keyword_overlap_score(log_question, confirmed_question)

    decision = log.get("decision")
    confidence = log.get("confidence")

    if decision in {"ask_teacher", "answer_with_caution"}:
        return True

    if confidence in {"low", "medium"} and overlap >= min_overlap:
        return True

    if overlap >= 0.45:
        return True

    return False


def build_correction_message(
    old_log: Dict[str, Any],
    confirmed: Dict[str, Any]
) -> str:
    """
    Tạo nội dung correction gửi lại cho user hoặc lưu để admin duyệt.
    """

    new_answer = confirmed.get("canonical_answer", "")

    return (
        "Mình cập nhật lại câu trả lời trước đó dựa trên thông tin đã được xác nhận:\n\n"
        + new_answer
    )


def sync_corrections_from_confirmed_answer(
    confirmed_answer_id: int,
    limit: int = 100,
    dry_run: bool = False
) -> List[Dict[str, Any]]:
    """
    Sau khi có confirmed answer mới:
    - lấy các answer_logs cùng topic
    - tìm log có khả năng cần correction
    - tạo correction nếu dry_run=False
    """

    confirmed = get_confirmed_answer_by_id(confirmed_answer_id)

    if confirmed is None:
        raise ValueError(f"Không tìm thấy confirmed_answers id={confirmed_answer_id}")

    topic = confirmed.get("topic")

    if not topic:
        return []

    logs = find_logs_that_may_need_correction(
        topic=topic,
        limit=limit
    )

    results = []

    for log in logs:
        if not should_create_correction(log, confirmed):
            continue

        new_message = build_correction_message(log, confirmed)

        item = {
            "answer_log_id": log["id"],
            "user_id": log.get("user_id"),
            "question": log.get("question"),
            "old_answer": log.get("answer"),
            "new_answer": new_message,
            "created": False,
            "correction_id": None
        }

        if not dry_run:
            correction_id = create_correction(
                answer_log_id=log["id"],
                new_answer=new_message,
                old_answer=log.get("answer"),
                reason="Có câu trả lời đã xác nhận mới từ thầy cô/admin.",
                source_confirmed_answer_id=confirmed_answer_id,
                status="pending"
            )

            item["created"] = True
            item["correction_id"] = correction_id

        results.append(item)

    return results