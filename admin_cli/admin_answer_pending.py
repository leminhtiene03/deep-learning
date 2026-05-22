# admin_cli/admin_answer_pending.py

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from knowledge.db import init_db
from workflow.teacher_queue import (
    get_pending_question_by_id,
    format_pending_question,
    answer_pending_question
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pending_id",
        type=int,
        required=True
    )

    parser.add_argument(
        "--answer",
        type=str,
        required=True
    )

    parser.add_argument(
        "--verified_by",
        type=str,
        default="admin"
    )

    parser.add_argument(
        "--source",
        type=str,
        default="teacher"
    )

    parser.add_argument(
        "--dry_run_corrections",
        action="store_true",
        help="Chỉ xem correction nào sẽ được tạo, chưa lưu vào DB."
    )

    parser.add_argument(
        "--no_corrections",
        action="store_true",
        help="Không tạo correction cho answer_logs cũ."
    )

    args = parser.parse_args()

    init_db()

    pending = get_pending_question_by_id(args.pending_id)

    if pending is None:
        print(f"Không tìm thấy pending_id={args.pending_id}")
        return

    print("PENDING QUESTION:")
    print(format_pending_question(pending))

    print("\nĐang lưu câu trả lời đã xác nhận...")

    result = answer_pending_question(
        pending_id=args.pending_id,
        answer=args.answer,
        verified_by=args.verified_by,
        source=args.source,
        create_corrections=not args.no_corrections,
        dry_run_corrections=args.dry_run_corrections
    )

    print("\nDONE")
    print("confirmed_answer_id:", result["confirmed_answer_id"])
    print("correction_count    :", result["correction_count"])

    if result["corrections"]:
        print("\nCorrections:")

        for c in result["corrections"]:
            print("-" * 80)
            print("answer_log_id :", c["answer_log_id"])
            print("user_id       :", c["user_id"])
            print("question      :", c["question"])
            print("created       :", c["created"])
            print("correction_id :", c["correction_id"])
            print("new_answer    :", c["new_answer"][:500])


if __name__ == "__main__":
    main()