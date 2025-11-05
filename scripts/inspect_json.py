# scripts/inspect_json.py
import json
from collections import Counter
import sys

def main(path="data/train_meta.json", n=10):
    with open(path,'r') as f:
        data = json.load(f)
    print("Total records:", len(data))
    print("Example record (first):")
    import pprint; pprint.pprint(data[0])
    # count keys and value types
    key_counts = Counter()
    type_counts = Counter()
    shapes = Counter()
    for rec in data:
        key_counts.update(rec.keys())
        for k,v in rec.items():
            type_counts[type(v).__name__] += 1
            # if list or dict check length
            if isinstance(v, list):
                shapes.update([len(v)])
    print("\nKey frequencies:", key_counts)
    print("Value types:", type_counts)
    if shapes:
        print("Common list lengths found in targets:", shapes)

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv)>1 else "data/train_meta.json"
    main(path)
