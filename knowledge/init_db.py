# knowledge/init_db.py

from db import init_db, DB_PATH

if __name__ == "__main__":
    init_db()
    print("SQLite database initialized.")
    print("DB path:", DB_PATH.resolve())