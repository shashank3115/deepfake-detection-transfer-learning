import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import models
from dataset import DeepfakeDataset

def make_model():
    from torchvision.models import resnet18, ResNet18_Weights
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    n_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(n_features, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 1),
        nn.Sigmoid()
    )
    return model

from tqdm import tqdm

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    loop = tqdm(loader, desc="Training", leave=False)
    for imgs, labels in loop:
        imgs, labels = imgs.to(device), labels.to(device).unsqueeze(1)
        preds = model(imgs)
        loss = loss_fn(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        loop.set_postfix(loss=loss.item())
    return total_loss / len(loader.dataset)


def eval_one_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        loop = tqdm(loader, desc="Validating", leave=False)
        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device).unsqueeze(1)
            preds = model(imgs)
            loss = loss_fn(preds, labels)
            total_loss += loss.item() * imgs.size(0)
            preds_binary = (preds > 0.5).float()
            correct += (preds_binary == labels).sum().item()
            loop.set_postfix(loss=loss.item())
    avg_loss = total_loss / len(loader.dataset)
    acc = correct / len(loader.dataset)
    return avg_loss, acc


def main():
    print("🚀 Starting training…")

    dataset = DeepfakeDataset("data/train_meta.json", "data/")
    val_size = int(0.1 * len(dataset))
    train_ds, val_ds = random_split(dataset, [len(dataset)-val_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)
    print("✅ Dataset loaded:", len(dataset), "samples")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.BCELoss()

    for epoch in range(1, 6):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_acc = eval_one_epoch(model, val_loader, loss_fn, device)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.3f}")

    torch.save(model.state_dict(), "outputs/model.pth")
    print("✅ Model saved at outputs/model.pth")

if __name__ == "__main__":
    main()
