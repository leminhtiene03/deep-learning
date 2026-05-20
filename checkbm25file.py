import pickle

with open("./index/bm25.pkl", "rb") as f:
    obj = pickle.load(f)

print("Type:", type(obj))

if isinstance(obj, dict):
    print("Keys:", obj.keys())
    for k, v in obj.items():
        print(k, "=>", type(v))