import os
import json
import zipfile
from datetime import datetime

def validate_json(pred_path):
    """Validate that the JSON file is formatted correctly."""
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"❌ Prediction file not found: {pred_path}")

    with open(pred_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("❌ JSON must contain a list of prediction entries.")

    for i, entry in enumerate(data):
        if not all(k in entry for k in ("image_id", "prediction")):
            raise ValueError(f"❌ Missing keys in entry {i}: {entry}")

    print(f"✅ JSON format validated successfully — {len(data)} predictions found.")
    return True


def make_submission(team_name="shashank"):
    """Package final hackathon submission (JSON + README + PPT)."""
    output_dir = "outputs"
    json_files = [f for f in os.listdir(output_dir) if f.endswith(".json")]

    if not json_files:
        raise FileNotFoundError("❌ No JSON files found in outputs/ folder.")
    elif len(json_files) > 1:
        print(f"⚠️ Multiple JSON files found. Using the first one: {json_files[0]}")

    pred_path = os.path.join(output_dir, json_files[0])
    ppt_path = f"{team_name}_presentation.pptx"
    readme_path = "README.md"

    # Validate JSON format
    validate_json(pred_path)

    # Ensure all required files exist
    missing_files = []
    for f in [pred_path, readme_path, ppt_path]:
        if not os.path.exists(f):
            missing_files.append(f)
    if missing_files:
        print(f"⚠️ Warning: The following files are missing: {', '.join(missing_files)}")

    # Create submissions folder
    os.makedirs("submissions", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    zip_name = os.path.join("submissions", f"{team_name}_submission_{timestamp}.zip")

    # Add files to zip
    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path in [pred_path, readme_path, ppt_path]:
            if os.path.exists(file_path):
                zipf.write(file_path, os.path.basename(file_path))
                print(f"📄 Added: {os.path.basename(file_path)}")

    print(f"\n📦 Submission package created: {zip_name}")
    print("✅ Upload this ZIP file on Unstop during the submission round.")


if __name__ == "__main__":
    make_submission(team_name="shashank")
