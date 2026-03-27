"""
YOLOv8 classification on TU-Berlin Sketch Dataset (250 categories).
Fine-tunes YOLOv8n-cls from pretrained ImageNet weights.
Tests how well fine-tuned CNNs handle pure shape (sketch) data.
"""
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
BATCH_SIZE = 128
IMG_SIZE = 224

DATA_DIR = DATA_ROOT / "sketch"
CLASSES = sorted([d.name for d in (DATA_DIR / "train").iterdir() if d.is_dir()])
NUM_CLASSES = len(CLASSES)
print(f"Classes ({NUM_CLASSES}): {CLASSES[:10]}... (showing first 10)")


def run_experiment():
    print(f"\n{'='*60}")
    print(f"YOLO Experiment: TU-Berlin Sketch (250 classes)")
    print(f"{'='*60}")

    model = YOLO("yolov8n-cls.pt")

    print(f"Training YOLOv8n-cls on TU-Berlin sketch data...")
    t0 = time.time()
    results = model.train(
        data=str(DATA_DIR),
        epochs=NUM_EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=4,
        project=str(RESULTS_DIR / "yolo_sketch_runs"),
        name="yolo_sketch",
        exist_ok=True,
        verbose=True,
        pretrained=True,
    )
    train_time = time.time() - t0

    # Evaluate on test set
    print("Evaluating on test set...")
    t0 = time.time()
    val_results = model.val(
        data=str(DATA_DIR),
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
    test_ds = datasets.ImageFolder(DATA_DIR / "test", transform=transform)

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
    csv_path = RESULTS_DIR / "yolo_sketch_runs" / "yolo_sketch" / "results.csv"
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
        "dataset": "tu-berlin-sketch",
        "variant": "sketch",
        "num_classes": NUM_CLASSES,
        "classes": CLASSES,
        "final_test_accuracy": float(test_acc),
        "train_time_s": round(train_time, 1),
        "eval_time_s": round(eval_time, 1),
        "history": history,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }

    out_path = RESULTS_DIR / "yolo_sketch.json"
    with open(out_path, "w") as f:
        json.dump(result_data, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print(f"Final Test Accuracy: {test_acc:.4f}")
    return result_data


if __name__ == "__main__":
    result = run_experiment()
    print(f"\n{'='*60}")
    print(f"YOLO TU-Berlin Sketch: {result['final_test_accuracy']:.4f}")
    print(f"{'='*60}")
