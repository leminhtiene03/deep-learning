# admin_pending.py

import argparse

from knowledge.db import init_db
from workflow.teacher_queue import print_pending_questions


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--status",
        type=str,
        default="pending",
        help="pending / sent_to_teacher / answered / rejected"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=20
    )

    args = parser.parse_args()

    init_db()

    print_pending_questions(
        status=args.status,
        limit=args.limit
    )


if __name__ == "__main__":
    main()