"""
DINOv2 classification on TU-Berlin Sketch Dataset (250 categories).
Frozen ViT-S/14 backbone + linear classifier head.
Tests how well self-supervised ViT features handle pure shape (sketch) data.
"""
import json
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

NUM_EPOCHS = 100
BATCH_SIZE = 128
IMG_SIZE = 224
LR = 1e-3

DATA_DIR = DATA_ROOT / "sketch"
CLASSES = sorted([d.name for d in (DATA_DIR / "train").iterdir() if d.is_dir()])
NUM_CLASSES = len(CLASSES)
print(f"Classes ({NUM_CLASSES}): {CLASSES[:10]}... (showing first 10)")


class DINOClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.head = nn.Linear(384, num_classes)

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)
        return self.head(features)


def run_experiment():
    print(f"\n{'='*60}")
    print(f"DINO Experiment: TU-Berlin Sketch (250 classes)")
    print(f"{'='*60}")

    transform_train = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(DATA_DIR / "train", transform=transform_train)
    test_ds = datasets.ImageFolder(DATA_DIR / "test", transform=transform_test)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"Train: {len(train_ds)}, Test: {len(test_ds)}")

    model = DINOClassifier(NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.head.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "test_acc": []}

    print("Training...")
    t0 = time.time()
    for epoch in range(NUM_EPOCHS):
        model.train()
        losses = []
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out = model(imgs)
            loss = criterion(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()

        avg_loss = np.mean(losses)
        history["train_loss"].append(avg_loss)

        # Eval
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                preds = model(imgs).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        test_acc = correct / total
        history["test_acc"].append(test_acc)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{NUM_EPOCHS}: loss={avg_loss:.4f}, test_acc={test_acc:.4f}")

    train_time = time.time() - t0

    # Final detailed eval
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(DEVICE)
            preds = model(imgs).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    test_acc = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=CLASSES, output_dict=True)
    cm = confusion_matrix(all_labels, all_preds)

    result_data = {
        "model": "DINOv2-ViTS14",
        "dataset": "tu-berlin-sketch",
        "variant": "sketch",
        "num_classes": NUM_CLASSES,
        "classes": CLASSES,
        "final_test_accuracy": float(test_acc),
        "train_time_s": round(train_time, 1),
        "history": history,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }

    out_path = RESULTS_DIR / "dino_sketch.json"
    with open(out_path, "w") as f:
        json.dump(result_data, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print(f"Final Test Accuracy: {test_acc:.4f}")
    return result_data


if __name__ == "__main__":
    result = run_experiment()
    print(f"\n{'='*60}")
    print(f"DINO TU-Berlin Sketch: {result['final_test_accuracy']:.4f}")
    print(f"{'='*60}")
