# knowledge/store.py

import json
from typing import Optional, List, Dict, Any

from knowledge.db import get_connection


def row_to_dict(row):
    if row is None:
        return None
    return dict(row)


def rows_to_dicts(rows):
    return [dict(r) for r in rows]


# ============================================================
# 1. CONFIRMED ANSWERS
# ============================================================

def add_confirmed_answer(
    canonical_question: str,
    canonical_answer: str,
    topic: Optional[str] = None,
    source: Optional[str] = None,
    verified_by: Optional[str] = None,
    status: str = "active"
) -> int:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO confirmed_answers (
            topic,
            canonical_question,
            canonical_answer,
            source,
            verified_by,
            status
        )
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        topic,
        canonical_question,
        canonical_answer,
        source,
        verified_by,
        status
    ))

    conn.commit()
    new_id = cur.lastrowid
    conn.close()

    return new_id


def get_confirmed_answer_by_id(answer_id: int) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT *
        FROM confirmed_answers
        WHERE id = ?
    """, (answer_id,))

    row = cur.fetchone()
    conn.close()

    return row_to_dict(row)


def search_confirmed_answers(
    query: str,
    topic: Optional[str] = None,
    limit: int = 5
) -> List[Dict[str, Any]]:
    """
    Tìm confirmed answer bằng LIKE đơn giản.
    Sau này có thể thay bằng embedding similarity.
    """
    conn = get_connection()
    cur = conn.cursor()

    like_query = f"%{query}%"

    if topic:
        cur.execute("""
            SELECT *
            FROM confirmed_answers
            WHERE status = 'active'
              AND topic = ?
              AND (
                    canonical_question LIKE ?
                    OR canonical_answer LIKE ?
                  )
            ORDER BY updated_at DESC
            LIMIT ?
        """, (topic, like_query, like_query, limit))
    else:
        cur.execute("""
            SELECT *
            FROM confirmed_answers
            WHERE status = 'active'
              AND (
                    canonical_question LIKE ?
                    OR canonical_answer LIKE ?
                  )
            ORDER BY updated_at DESC
            LIMIT ?
        """, (like_query, like_query, limit))

    rows = cur.fetchall()
    conn.close()

    return rows_to_dicts(rows)


# ============================================================
# 2. PENDING QUESTIONS
# ============================================================

def add_pending_question(
    question: str,
    user_id: Optional[str] = None,
    topic: Optional[str] = None,
    context_snapshot: Optional[str] = None,
    retrieved_chunk_id: Optional[int] = None,
    reason: Optional[str] = None,
    teacher_questions: Optional[List[str]] = None,
    status: str = "pending"
) -> int:
    conn = get_connection()
    cur = conn.cursor()

    teacher_questions_json = None
    if teacher_questions is not None:
        teacher_questions_json = json.dumps(
            teacher_questions,
            ensure_ascii=False
        )

    cur.execute("""
        INSERT INTO pending_questions (
            user_id,
            question,
            topic,
            context_snapshot,
            retrieved_chunk_id,
            reason,
            teacher_questions_json,
            status
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id,
        question,
        topic,
        context_snapshot,
        retrieved_chunk_id,
        reason,
        teacher_questions_json,
        status
    ))

    conn.commit()
    new_id = cur.lastrowid
    conn.close()

    return new_id


def list_pending_questions(
    status: str = "pending",
    limit: int = 50
) -> List[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT *
        FROM pending_questions
        WHERE status = ?
        ORDER BY created_at ASC
        LIMIT ?
    """, (status, limit))

    rows = cur.fetchall()
    conn.close()

    results = rows_to_dicts(rows)

    for item in results:
        raw = item.get("teacher_questions_json")
        if raw:
            try:
                item["teacher_questions"] = json.loads(raw)
            except Exception:
                item["teacher_questions"] = []
        else:
            item["teacher_questions"] = []

    return results


def update_pending_status(
    pending_id: int,
    status: str
) -> None:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        UPDATE pending_questions
        SET status = ?,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
    """, (status, pending_id))

    conn.commit()
    conn.close()


def mark_pending_answered(pending_id: int) -> None:
    update_pending_status(pending_id, "answered")


# ============================================================
# 3. ANSWER LOGS
# ============================================================

def add_answer_log(
    question: str,
    answer: str,
    user_id: Optional[str] = None,
    topic: Optional[str] = None,
    chunk_id: Optional[int] = None,
    page: Optional[int] = None,
    context_snapshot: Optional[str] = None,
    retrieval_score: Optional[float] = None,
    bm25_rank: Optional[int] = None,
    faiss_rank: Optional[int] = None,
    confidence: Optional[str] = None,
    decision: Optional[str] = None,
    status: str = "active"
) -> int:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO answer_logs (
            user_id,
            question,
            answer,
            topic,
            chunk_id,
            page,
            context_snapshot,
            retrieval_score,
            bm25_rank,
            faiss_rank,
            confidence,
            decision,
            status
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id,
        question,
        answer,
        topic,
        chunk_id,
        page,
        context_snapshot,
        retrieval_score,
        bm25_rank,
        faiss_rank,
        confidence,
        decision,
        status
    ))

    conn.commit()
    new_id = cur.lastrowid
    conn.close()

    return new_id


def get_answer_log_by_id(log_id: int) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT *
        FROM answer_logs
        WHERE id = ?
    """, (log_id,))

    row = cur.fetchone()
    conn.close()

    return row_to_dict(row)


def list_answer_logs_by_user(
    user_id: str,
    limit: int = 20
) -> List[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT *
        FROM answer_logs
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT ?
    """, (user_id, limit))

    rows = cur.fetchall()
    conn.close()

    return rows_to_dicts(rows)


def list_answer_logs_by_topic(
    topic: str,
    limit: int = 50
) -> List[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT *
        FROM answer_logs
        WHERE topic = ?
        ORDER BY created_at DESC
        LIMIT ?
    """, (topic, limit))

    rows = cur.fetchall()
    conn.close()

    return rows_to_dicts(rows)


def mark_answer_log_needs_correction(log_id: int) -> None:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        UPDATE answer_logs
        SET status = 'needs_correction'
        WHERE id = ?
    """, (log_id,))

    conn.commit()
    conn.close()


def mark_answer_log_corrected(log_id: int) -> None:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        UPDATE answer_logs
        SET status = 'corrected'
        WHERE id = ?
    """, (log_id,))

    conn.commit()
    conn.close()


# ============================================================
# 4. CORRECTIONS
# ============================================================

def create_correction(
    answer_log_id: int,
    new_answer: str,
    old_answer: Optional[str] = None,
    reason: Optional[str] = None,
    source_confirmed_answer_id: Optional[int] = None,
    status: str = "pending"
) -> int:
    if old_answer is None:
        log = get_answer_log_by_id(answer_log_id)
        if log:
            old_answer = log.get("answer")

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO corrections (
            answer_log_id,
            old_answer,
            new_answer,
            reason,
            source_confirmed_answer_id,
            status
        )
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        answer_log_id,
        old_answer,
        new_answer,
        reason,
        source_confirmed_answer_id,
        status
    ))

    conn.commit()
    new_id = cur.lastrowid
    conn.close()

    mark_answer_log_needs_correction(answer_log_id)

    return new_id


def list_corrections(
    status: str = "pending",
    limit: int = 50
) -> List[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT *
        FROM corrections
        WHERE status = ?
        ORDER BY created_at ASC
        LIMIT ?
    """, (status, limit))

    rows = cur.fetchall()
    conn.close()

    return rows_to_dicts(rows)


def update_correction_status(
    correction_id: int,
    status: str
) -> None:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        UPDATE corrections
        SET status = ?,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
    """, (status, correction_id))

    conn.commit()
    conn.close()


def mark_correction_notified(correction_id: int) -> None:
    update_correction_status(correction_id, "notified")


# ============================================================
# 5. HIGH-LEVEL HELPERS
# ============================================================

def save_teacher_answer_for_pending(
    pending_id: int,
    canonical_answer: str,
    verified_by: Optional[str] = None,
    source: Optional[str] = "teacher"
) -> int:
    """
    Khi thầy cô trả lời 1 pending question:
    - lấy pending question
    - lưu thành confirmed answer
    - đánh dấu pending là answered
    """
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT *
        FROM pending_questions
        WHERE id = ?
    """, (pending_id,))

    pending = cur.fetchone()
    conn.close()

    if pending is None:
        raise ValueError(f"Không tìm thấy pending question id={pending_id}")

    pending = dict(pending)

    confirmed_id = add_confirmed_answer(
        topic=pending.get("topic"),
        canonical_question=pending.get("question"),
        canonical_answer=canonical_answer,
        source=source,
        verified_by=verified_by,
        status="active"
    )

    mark_pending_answered(pending_id)

    return confirmed_id


def find_logs_that_may_need_correction(
    topic: str,
    limit: int = 50
) -> List[Dict[str, Any]]:
    """
    Bản đơn giản: lấy các answer_logs cùng topic.
    Sau này có thể nâng cấp bằng similarity search.
    """
    return list_answer_logs_by_topic(topic=topic, limit=limit)