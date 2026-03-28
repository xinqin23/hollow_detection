"""
Generate showcase figures for TU-Berlin Sketch classification results.
- Correct predictions: 2x5 grid with true label as title
- Incorrect predictions: 2x5 grid with "True: X -> Pred: Y" as title
Uses YOLOv8n-cls (saved weights available).
"""
import random
from pathlib import Path
from collections import defaultdict

from PIL import Image
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

PROJECT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT / "data" / "sketch" / "test"
MODEL_PATH = PROJECT / "results" / "yolo_sketch_runs" / "yolo_sketch" / "weights" / "best.pt"
RESULTS_DIR = PROJECT / "results"


def main():
    print(f"Loading YOLO model from {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))

    correct = []   # (img_path, true_label)
    incorrect = []  # (img_path, true_label, pred_label)

    for class_dir in sorted(DATA_DIR.iterdir()):
        if not class_dir.is_dir():
            continue
        true_label = class_dir.name
        images = sorted(class_dir.glob("*.png"))
        if not images:
            continue

        print(f"  Predicting {true_label}: {len(images)} images")
        for img_path in images:
            results = model(str(img_path), verbose=False)
            pred_idx = results[0].probs.top1
            pred_label = results[0].names[pred_idx]

            if pred_label == true_label:
                correct.append((img_path, true_label))
            else:
                incorrect.append((img_path, true_label, pred_label))

    print(f"\nTotal correct: {len(correct)}, incorrect: {len(incorrect)}")

    def pick_diverse(items, n=10, key_fn=lambda x: x[1]):
        """Pick n items spread across different classes."""
        by_class = defaultdict(list)
        for item in items:
            by_class[key_fn(item)].append(item)
        for k in by_class:
            random.shuffle(by_class[k])
        picked = []
        class_keys = sorted(by_class.keys())
        idx = 0
        while len(picked) < n and any(by_class[k] for k in class_keys):
            k = class_keys[idx % len(class_keys)]
            if by_class[k]:
                picked.append(by_class[k].pop())
            idx += 1
        return picked

    random.seed(42)
    correct_samples = pick_diverse(correct, n=10)
    incorrect_samples = pick_diverse(incorrect, n=10)

    # --- Figure 1: Correct predictions ---
    fig, axes = plt.subplots(2, 5, figsize=(20, 9))
    fig.suptitle("Correctly Classified Sketches (YOLOv8n)", fontsize=18, fontweight="bold", y=0.98)
    for i, (img_path, true_label) in enumerate(correct_samples):
        ax = axes[i // 5, i % 5]
        img = Image.open(img_path).convert("RGB")
        ax.imshow(img)
        ax.set_title(true_label, fontsize=14, fontweight="bold", color="green")
        ax.axis("off")
    for j in range(len(correct_samples), 10):
        axes[j // 5, j % 5].axis("off")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out1 = RESULTS_DIR / "sketch_correct_examples.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out1}")

    # --- Figure 2: Incorrect predictions ---
    fig, axes = plt.subplots(2, 5, figsize=(20, 9))
    fig.suptitle("Misclassified Sketches (YOLOv8n)", fontsize=18, fontweight="bold", y=0.98)
    for i, (img_path, true_label, pred_label) in enumerate(incorrect_samples):
        ax = axes[i // 5, i % 5]
        img = Image.open(img_path).convert("RGB")
        ax.imshow(img)
        ax.set_title(f"True: {true_label} \u2192 Pred: {pred_label}", fontsize=12, fontweight="bold", color="red")
        ax.axis("off")
    for j in range(len(incorrect_samples), 10):
        axes[j // 5, j % 5].axis("off")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out2 = RESULTS_DIR / "sketch_incorrect_examples.png"
    fig.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out2}")


if __name__ == "__main__":
    main()
