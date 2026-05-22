# knowledge/db.py

import sqlite3
from pathlib import Path
from contextlib import contextmanager
import sys

# Add project root to path for config import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RAG_DB_PATH

DB_PATH = RAG_DB_PATH


def get_connection():
    # Bổ sung check_same_thread=False để tương thích với cơ chế worker threads của FastAPI
    # Thêm timeout=15.0 để luồng sau kiên nhẫn chờ luồng trước ghi xong, tránh lỗi 'database is locked'
    conn = sqlite3.connect(
        DB_PATH, 
        check_same_thread=False, 
        timeout=15.0
    )
    conn.row_factory = sqlite3.Row

    # Bật foreign key
    conn.execute("PRAGMA foreign_keys = ON;")

    # WAL giúp SQLite đọc/ghi ổn hơn khi có nhiều request nhỏ
    conn.execute("PRAGMA journal_mode = WAL;")
    
    # Đồng bộ ở mức NORMAL kết hợp với WAL để tăng tốc độ ghi mà vẫn an toàn dữ liệu
    conn.execute("PRAGMA synchronous = NORMAL;")

    return conn

# Cung cấp hàm dependency để FastAPI tạo session tự động đóng cho mỗi request
def get_db_session():
    conn = get_connection()
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    conn = get_connection()
    cur = conn.cursor()

    # =========================================================
    # 1. Câu trả lời đã xác nhận bởi thầy cô / admin
    # =========================================================
    cur.execute("""
    CREATE TABLE IF NOT EXISTS confirmed_answers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,

        topic TEXT,
        canonical_question TEXT NOT NULL,
        canonical_answer TEXT NOT NULL,

        source TEXT,
        verified_by TEXT,
        status TEXT DEFAULT 'active',

        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # =========================================================
    # 2. Câu hỏi cần hỏi thầy cô
    # =========================================================
    cur.execute("""
    CREATE TABLE IF NOT EXISTS pending_questions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,

        user_id TEXT,
        question TEXT NOT NULL,
        topic TEXT,

        context_snapshot TEXT,
        retrieved_chunk_id INTEGER,
        reason TEXT,

        teacher_questions_json TEXT,

        status TEXT DEFAULT 'pending',
        -- pending / sent_to_teacher / answered / rejected

        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # =========================================================
    # 3. Log mọi câu trả lời chatbot đã đưa ra
    # =========================================================
    cur.execute("""
    CREATE TABLE IF NOT EXISTS answer_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,

        user_id TEXT,
        question TEXT NOT NULL,
        answer TEXT NOT NULL,

        topic TEXT,

        chunk_id INTEGER,
        page INTEGER,
        context_snapshot TEXT,

        retrieval_score REAL,
        bm25_rank INTEGER,
        faiss_rank INTEGER,

        confidence TEXT,
        decision TEXT,
        -- answer_directly / answer_with_caution / ask_teacher / use_confirmed_cache

        status TEXT DEFAULT 'active',
        -- active / needs_correction / corrected

        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # =========================================================
    # 4. Các correction cho câu trả lời cũ
    # =========================================================
    cur.execute("""
    CREATE TABLE IF NOT EXISTS corrections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,

        answer_log_id INTEGER,

        old_answer TEXT,
        new_answer TEXT NOT NULL,

        reason TEXT,
        source_confirmed_answer_id INTEGER,

        status TEXT DEFAULT 'pending',
        -- pending / notified / dismissed

        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,

        FOREIGN KEY(answer_log_id) REFERENCES answer_logs(id),
        FOREIGN KEY(source_confirmed_answer_id) REFERENCES confirmed_answers(id)
    );
    """)

    # =========================================================
    # Index để tìm nhanh hơn
    # =========================================================
    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_confirmed_topic
    ON confirmed_answers(topic);
    """)

    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_pending_status
    ON pending_questions(status);
    """)

    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_pending_topic
    ON pending_questions(topic);
    """)

    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_logs_user
    ON answer_logs(user_id);
    """)

    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_logs_topic
    ON answer_logs(topic);
    """)

    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_logs_status
    ON answer_logs(status);
    """)

    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_corrections_status
    ON corrections(status);
    """)

    conn.commit()
    conn.close()