import json, os

real_json = "data/real_cifake_preds.json"
fake_json = "data/fake_cifake_preds.json"

if not (os.path.exists(real_json) and os.path.exists(fake_json)):
    raise FileNotFoundError("❌ Missing JSON files. Please check data/ folder.")

with open(real_json, "r") as f:
    real = json.load(f)
with open(fake_json, "r") as f:
    fake = json.load(f)

# Add labels
for r in real:
    r["label"] = 1
for f_ in fake:
    f_["label"] = 0

combined = real + fake
print(f"✅ Merged {len(real)} real + {len(fake)} fake = {len(combined)} total")

out_path = "data/train_meta.json"
with open(out_path, "w") as f:
    json.dump(combined, f, indent=2)
print("💾 Saved:", out_path)
