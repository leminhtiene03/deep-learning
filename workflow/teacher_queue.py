# workflow/teacher_queue.py

import json
from typing import Optional, Dict, Any, List

from knowledge.db import get_connection
from knowledge.store import (
    list_pending_questions,
    save_teacher_answer_for_pending
)
from workflow.knowledge_sync import sync_corrections_from_confirmed_answer


def get_pending_question_by_id(pending_id: int) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT *
        FROM pending_questions
        WHERE id = ?
    """, (pending_id,))

    row = cur.fetchone()
    conn.close()

    if row is None:
        return None

    item = dict(row)

    raw = item.get("teacher_questions_json")

    if raw:
        try:
            item["teacher_questions"] = json.loads(raw)
        except Exception:
            item["teacher_questions"] = []
    else:
        item["teacher_questions"] = []

    return item


def format_pending_question(item: Dict[str, Any]) -> str:
    teacher_questions = item.get("teacher_questions", [])

    lines = []
    lines.append("=" * 80)
    lines.append(f"Pending ID : {item.get('id')}")
    lines.append(f"User ID    : {item.get('user_id')}")
    lines.append(f"Topic      : {item.get('topic')}")
    lines.append(f"Status     : {item.get('status')}")
    lines.append(f"Created at : {item.get('created_at')}")
    lines.append("")
    lines.append("Question:")
    lines.append(item.get("question") or "")
    lines.append("")
    lines.append("Reason:")
    lines.append(item.get("reason") or "")

    if item.get("retrieved_chunk_id") is not None:
        lines.append("")
        lines.append(f"Retrieved chunk ID: {item.get('retrieved_chunk_id')}")

    if teacher_questions:
        lines.append("")
        lines.append("Suggested teacher questions:")
        for i, q in enumerate(teacher_questions, start=1):
            lines.append(f"{i}. {q}")

    context = item.get("context_snapshot")

    if context:
        lines.append("")
        lines.append("Context snapshot:")
        lines.append(context[:1200])

        if len(context) > 1200:
            lines.append("...")

    return "\n".join(lines)


def print_pending_questions(
    status: str = "pending",
    limit: int = 20
) -> None:
    pending = list_pending_questions(
        status=status,
        limit=limit
    )

    if not pending:
        print(f"Không có pending_questions với status='{status}'.")
        return

    for item in pending:
        print(format_pending_question(item))


def answer_pending_question(
    pending_id: int,
    answer: str,
    verified_by: Optional[str] = None,
    source: str = "teacher",
    create_corrections: bool = True,
    dry_run_corrections: bool = False
) -> Dict[str, Any]:
    """
    Admin nhập câu trả lời từ thầy cô cho 1 pending question.

    Hàm này sẽ:
    - lưu vào confirmed_answers
    - mark pending là answered
    - tạo corrections cho answer_logs cũ liên quan
    """

    pending = get_pending_question_by_id(pending_id)

    if pending is None:
        raise ValueError(f"Không tìm thấy pending question id={pending_id}")

    confirmed_id = save_teacher_answer_for_pending(
        pending_id=pending_id,
        canonical_answer=answer,
        verified_by=verified_by,
        source=source
    )

    corrections = []

    if create_corrections:
        corrections = sync_corrections_from_confirmed_answer(
            confirmed_answer_id=confirmed_id,
            dry_run=dry_run_corrections
        )

    return {
        "pending_id": pending_id,
        "confirmed_answer_id": confirmed_id,
        "corrections": corrections,
        "correction_count": len(corrections)
    }