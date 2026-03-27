"""
YOLOv8 classification on Overwatch frames (original and hollow contour).
Fine-tunes YOLOv8n-cls from pretrained ImageNet weights.
"""
import os
import json
import time
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torchvision import datasets, transforms
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

NUM_EPOCHS = 30
BATCH_SIZE = 64
IMG_SIZE = 224

CLASSES = sorted([d.name for d in (DATA_ROOT / "overwatch_original" / "train").iterdir() if d.is_dir()])
NUM_CLASSES = len(CLASSES)
print(f"Classes ({NUM_CLASSES}): {CLASSES}")


def run_experiment(variant):
    print(f"\n{'='*60}")
    print(f"YOLO Experiment: Overwatch {variant}")
    print(f"{'='*60}")

    data_dir = DATA_ROOT / f"overwatch_{variant}"

    model = YOLO("yolov8n-cls.pt")

    print(f"Training YOLOv8n-cls on overwatch {variant} data...")
    t0 = time.time()
    results = model.train(
        data=str(data_dir),
        epochs=NUM_EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=4,
        project=str(RESULTS_DIR / "yolo_overwatch_runs"),
        name=f"yolo_{variant}",
        exist_ok=True,
        verbose=True,
        pretrained=True,
    )
    train_time = time.time() - t0

    # Evaluate on test set
    print("Evaluating on test set...")
    t0 = time.time()
    val_results = model.val(
        data=str(data_dir),
        split="test",
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
    )
    eval_time = time.time() - t0

    # Get predictions for detailed metrics
    print("Getting per-image predictions...")
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    test_ds = datasets.ImageFolder(data_dir / "test", transform=transform)

    all_preds, all_labels = [], []
    for img_path, label in tqdm(test_ds.samples, desc="Predicting"):
        result = model.predict(img_path, imgsz=IMG_SIZE, verbose=False, device=DEVICE)
        pred_class = result[0].probs.top1
        all_preds.append(pred_class)
        all_labels.append(label)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    test_acc = accuracy_score(all_labels, all_preds)

    report = classification_report(all_labels, all_preds, target_names=CLASSES, output_dict=True)
    cm = confusion_matrix(all_labels, all_preds)

    # Collect training history
    history = {"train_loss": [], "test_acc": []}
    csv_path = RESULTS_DIR / "yolo_overwatch_runs" / f"yolo_{variant}" / "results.csv"
    if csv_path.exists():
        import pandas as pd
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        if "train/loss" in df.columns:
            history["train_loss"] = df["train/loss"].tolist()
        if "metrics/accuracy_top1" in df.columns:
            history["test_acc"] = df["metrics/accuracy_top1"].tolist()

    result_data = {
        "model": "YOLOv8n-cls",
        "dataset": "overwatch",
        "variant": variant,
        "num_classes": NUM_CLASSES,
        "classes": CLASSES,
        "final_test_accuracy": float(test_acc),
        "train_time_s": round(train_time, 1),
        "eval_time_s": round(eval_time, 1),
        "history": history,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }

    out_path = RESULTS_DIR / f"yolo_overwatch_{variant}.json"
    with open(out_path, "w") as f:
        json.dump(result_data, f, indent=2)
    print(f"Results saved to {out_path}")
    print(f"Final Test Accuracy: {test_acc:.4f}")
    return result_data


if __name__ == "__main__":
    results = {}
    for variant in ["original", "contour"]:
        results[variant] = run_experiment(variant)

    print("\n" + "=" * 60)
    print("YOLO Overwatch Summary:")
    print(f"  Original: {results['original']['final_test_accuracy']:.4f}")
    print(f"  Contour:  {results['contour']['final_test_accuracy']:.4f}")
    print("=" * 60)
