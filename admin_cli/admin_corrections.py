# admin_cli/admin_corrections.py

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from knowledge.db import init_db
from knowledge.store import list_corrections, mark_correction_notified


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--status",
        type=str,
        default="pending"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=20
    )

    parser.add_argument(
        "--mark_notified",
        type=int,
        default=None,
        help="Correction ID muốn đánh dấu là đã thông báo."
    )

    args = parser.parse_args()

    init_db()

    if args.mark_notified is not None:
        mark_correction_notified(args.mark_notified)
        print(f"Marked correction {args.mark_notified} as notified.")
        return

    corrections = list_corrections(
        status=args.status,
        limit=args.limit
    )

    if not corrections:
        print(f"Không có corrections với status='{args.status}'.")
        return

    for c in corrections:
        print("=" * 80)
        print("Correction ID:", c["id"])
        print("Answer Log ID:", c["answer_log_id"])
        print("Status       :", c["status"])
        print("Reason       :", c["reason"])
        print("")
        print("Old answer:")
        print((c["old_answer"] or "")[:800])
        print("")
        print("New answer:")
        print((c["new_answer"] or "")[:800])


if __name__ == "__main__":
    main()