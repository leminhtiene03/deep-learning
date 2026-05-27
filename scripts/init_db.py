# scripts/init_db.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from knowledge.db import init_db, DB_PATH

if __name__ == "__main__":
    init_db()
    print("SQLite database initialized.")
    print("DB path:", Path(DB_PATH).resolve())