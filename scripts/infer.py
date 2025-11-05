import torch, os, json
from PIL import Image
import torchvision.transforms as T
from torchvision import models
from torch import nn

def make_model():
    model = models.resnet18(pretrained=True)
    n_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(n_features, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 1),
        nn.Sigmoid()
    )
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model().to(device)
    model.load_state_dict(torch.load("outputs/model.pth", map_location=device))
    model.eval()

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dir = "data/test_images"
    results = []

    for fname in sorted(os.listdir(test_dir)):
        path = os.path.join(test_dir, fname)
        img = Image.open(path).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(img).item()
        results.append({
            "image_id": fname,
            "prediction": float(pred)
        })

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/teamname_prediction.json", "w") as f:
        json.dump(results, f, indent=2)
    print("✅ Saved: outputs/teamname_prediction.json")

if __name__ == "__main__":
    main()
