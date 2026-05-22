"""
Quản lý lịch sử trò chuyện với SQLite cho hệ thống RAG chatbot.
Thread-safe và tối ưu cho truy vấn nhanh.
"""

import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import logging
import sys

# Add project root to path for config import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CHAT_HISTORY_DB_PATH

logger = logging.getLogger(__name__)


class ChatHistoryStore:
    """Quản lý lịch sử chat với SQLite."""

    def __init__(self, db_path: str = None):
        """
        Khởi tạo kho lưu trữ lịch sử chat.

        Args:
            db_path: Đường dẫn đến file SQLite database (mặc định từ config)
        """
        self.db_path = db_path or CHAT_HISTORY_DB_PATH
        self._init_database()

    def _init_database(self) -> None:
        """Khởi tạo bảng chat_history nếu chưa tồn tại."""
        try:
            with closing(sqlite3.connect(self.db_path, check_same_thread=False)) as conn:
                with conn:
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS chat_history (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            conversation_id TEXT NOT NULL,
                            role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                            content TEXT NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                    # Tạo index để tối ưu truy vấn theo conversation_id và thời gian
                    conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_conversation_created
                        ON chat_history(conversation_id, created_at DESC)
                    """)
            logger.info(f"Database initialized at {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str
    ) -> bool:
        """
        Thêm tin nhắn mới vào lịch sử chat.

        Args:
            conversation_id: ID của phiên trò chuyện
            role: Vai trò ('user' hoặc 'assistant')
            content: Nội dung tin nhắn

        Returns:
            True nếu thêm thành công, False nếu thất bại
        """
        if role not in ('user', 'assistant'):
            logger.error(f"Invalid role: {role}")
            return False

        try:
            with closing(sqlite3.connect(self.db_path, check_same_thread=False)) as conn:
                with conn:
                    conn.execute(
                        """
                        INSERT INTO chat_history (conversation_id, role, content, created_at)
                        VALUES (?, ?, ?, ?)
                        """,
                        (conversation_id, role, content, datetime.now())
                    )
            logger.debug(f"Added message for conversation {conversation_id}")
            return True
        except sqlite3.Error as e:
            logger.error(f"Failed to add message: {e}")
            return False

    def get_recent_history(
        self,
        conversation_id: str,
        k: int = 5
    ) -> List[Dict[str, str]]:
        """
        Truy xuất k cặp hội thoại gần nhất.

        Args:
            conversation_id: ID của phiên trò chuyện
            k: Số lượng cặp hội thoại (user-assistant) cần lấy

        Returns:
            Danh sách các tin nhắn dạng [{"role": "user", "content": "..."}, ...]
            Sắp xếp từ cũ đến mới
        """
        try:
            with closing(sqlite3.connect(self.db_path, check_same_thread=False)) as conn:
                conn.row_factory = sqlite3.Row
                with conn:
                    cursor = conn.execute(
                        """
                        SELECT role, content, created_at
                        FROM chat_history
                        WHERE conversation_id = ?
                        ORDER BY created_at DESC
                        LIMIT ?
                        """,
                        (conversation_id, k * 2)  # k cặp = 2k tin nhắn
                    )
                    rows = cursor.fetchall()

            # Đảo ngược để có thứ tự từ cũ đến mới
            messages = [
                {"role": row["role"], "content": row["content"]}
                for row in reversed(rows)
            ]

            logger.debug(f"Retrieved {len(messages)} messages for conversation {conversation_id}")
            return messages

        except sqlite3.Error as e:
            logger.error(f"Failed to retrieve history: {e}")
            return []

    def clear_conversation(self, conversation_id: str) -> bool:
        """
        Xóa toàn bộ lịch sử của một phiên trò chuyện.

        Args:
            conversation_id: ID của phiên trò chuyện cần xóa

        Returns:
            True nếu xóa thành công
        """
        try:
            with closing(sqlite3.connect(self.db_path, check_same_thread=False)) as conn:
                with conn:
                    conn.execute(
                        "DELETE FROM chat_history WHERE conversation_id = ?",
                        (conversation_id,)
                    )
            logger.info(f"Cleared conversation {conversation_id}")
            return True
        except sqlite3.Error as e:
            logger.error(f"Failed to clear conversation: {e}")
            return False

    def get_conversation_count(self, conversation_id: str) -> int:
        """
        Đếm số tin nhắn trong một phiên trò chuyện.

        Args:
            conversation_id: ID của phiên trò chuyện

        Returns:
            Số lượng tin nhắn
        """
        try:
            with closing(sqlite3.connect(self.db_path, check_same_thread=False)) as conn:
                with conn:
                    cursor = conn.execute(
                        "SELECT COUNT(*) FROM chat_history WHERE conversation_id = ?",
                        (conversation_id,)
                    )
                    count = cursor.fetchone()[0]
            return count
        except sqlite3.Error as e:
            logger.error(f"Failed to count messages: {e}")
            return 0
