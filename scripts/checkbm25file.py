import sys
import pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import BM25_INDEX_PATH

with open(BM25_INDEX_PATH, "rb") as f:
    obj = pickle.load(f)

print("Type:", type(obj))

if isinstance(obj, dict):
    print("Keys:", obj.keys())
    for k, v in obj.items():
        print(k, "=>", type(v))