"""
DINOv2 feature extraction + linear classifier for CIFAR-10.
Runs on both original and hollow contour images.
"""
import os
import sys
import json
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# DINOv2 ViT-S/14 expects 224x224 with ImageNet normalization
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

BATCH_SIZE = 256
NUM_EPOCHS = 30
LR = 1e-3


class LinearClassifier(nn.Module):
    def __init__(self, in_dim, num_classes=10):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


def extract_features(model, loader):
    """Extract DINOv2 CLS token features for all images."""
    all_feats, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Extracting features"):
            imgs = imgs.to(DEVICE)
            feats = model(imgs)  # DINOv2 forward returns CLS token
            all_feats.append(feats.cpu())
            all_labels.append(labels)
    return torch.cat(all_feats), torch.cat(all_labels)


def train_classifier(train_feats, train_labels, test_feats, test_labels, tag):
    """Train linear probe and evaluate."""
    in_dim = train_feats.shape[1]
    classifier = LinearClassifier(in_dim).to(DEVICE)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    train_ds = torch.utils.data.TensorDataset(train_feats, train_labels)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    history = {"train_loss": [], "train_acc": [], "test_acc": []}

    for epoch in range(NUM_EPOCHS):
        classifier.train()
        total_loss, correct, total = 0, 0, 0
        for feats, labels in train_loader:
            feats, labels = feats.to(DEVICE), labels.to(DEVICE)
            logits = classifier(feats)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * feats.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += feats.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        # Evaluate
        classifier.eval()
        with torch.no_grad():
            test_logits = classifier(test_feats.to(DEVICE))
            test_preds = test_logits.argmax(1).cpu()
            test_acc = accuracy_score(test_labels, test_preds)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{NUM_EPOCHS} | "
                  f"Loss: {train_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | "
                  f"Test Acc: {test_acc:.4f}")

    # Final evaluation
    classifier.eval()
    with torch.no_grad():
        test_logits = classifier(test_feats.to(DEVICE))
        test_preds = test_logits.argmax(1).cpu().numpy()
    test_labels_np = test_labels.numpy()

    report = classification_report(test_labels_np, test_preds,
                                   target_names=CLASSES, output_dict=True)
    cm = confusion_matrix(test_labels_np, test_preds)

    return history, report, cm, test_acc


def run_experiment(variant):
    """Run DINO experiment on a dataset variant (original or contour)."""
    print(f"\n{'='*60}")
    print(f"DINO Experiment: {variant}")
    print(f"{'='*60}")

    data_dir = DATA_ROOT / f"cifar10_{variant}"
    train_ds = datasets.ImageFolder(data_dir / "train", transform=TRANSFORM)
    test_ds = datasets.ImageFolder(data_dir / "test", transform=TRANSFORM)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=4, pin_memory=True)

    print(f"Loading DINOv2 ViT-S/14...")
    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    dino = dino.to(DEVICE)
    dino.eval()

    print("Extracting train features...")
    t0 = time.time()
    train_feats, train_labels = extract_features(dino, train_loader)
    print("Extracting test features...")
    test_feats, test_labels = extract_features(dino, test_loader)
    extract_time = time.time() - t0
    print(f"Feature extraction time: {extract_time:.1f}s")
    print(f"Feature dim: {train_feats.shape[1]}")

    print("Training linear classifier...")
    t0 = time.time()
    history, report, cm, final_acc = train_classifier(
        train_feats, train_labels, test_feats, test_labels, variant
    )
    train_time = time.time() - t0

    # Save results
    result = {
        "model": "DINOv2-ViT-S/14",
        "variant": variant,
        "final_test_accuracy": final_acc,
        "feature_dim": int(train_feats.shape[1]),
        "extract_time_s": round(extract_time, 1),
        "train_time_s": round(train_time, 1),
        "history": history,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }

    out_path = RESULTS_DIR / f"dino_{variant}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {out_path}")
    print(f"Final Test Accuracy: {final_acc:.4f}")

    return result


if __name__ == "__main__":
    results = {}
    for variant in ["original", "contour"]:
        results[variant] = run_experiment(variant)

    print("\n" + "="*60)
    print("DINO Summary:")
    print(f"  Original: {results['original']['final_test_accuracy']:.4f}")
    print(f"  Contour:  {results['contour']['final_test_accuracy']:.4f}")
    print("="*60)
