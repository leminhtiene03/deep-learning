# test_db.py

from knowledge.db import get_connection

conn = get_connection()
cur = conn.cursor()

cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cur.fetchall()

print("Tables:")
for t in tables:
    print("-", t["name"])

conn.close()