"""
Generate comparison plots for DINO vs YOLO on original vs contour CIFAR-10.
Outputs:
  - results/accuracy_comparison.png  (bar chart)
  - results/confusion_matrices.png   (4 confusion matrices)
  - results/per_class_accuracy.png   (grouped bar chart per class)
  - results/training_curves.png      (loss/acc over epochs)
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

def load_results():
    data = {}
    for name in ["dino_original", "dino_contour", "yolo_original", "yolo_contour"]:
        with open(RESULTS_DIR / f"{name}.json") as f:
            data[name] = json.load(f)
    return data

def plot_accuracy_comparison(data):
    """Bar chart: overall accuracy for each model/variant combination."""
    fig, ax = plt.subplots(figsize=(8, 5))

    models = ["DINOv2", "YOLOv8n-cls"]
    original = [data["dino_original"]["final_test_accuracy"],
                data["yolo_original"]["final_test_accuracy"]]
    contour = [data["dino_contour"]["final_test_accuracy"],
               data["yolo_contour"]["final_test_accuracy"]]

    x = np.arange(len(models))
    w = 0.3
    bars1 = ax.bar(x - w/2, [v*100 for v in original], w, label="Original", color="#4C72B0")
    bars2 = ax.bar(x + w/2, [v*100 for v in contour], w, label="Contour", color="#DD8452")

    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Overall Test Accuracy: Original vs Contour", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=11)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "accuracy_comparison.png", dpi=150, bbox_inches="tight")
    print("Saved accuracy_comparison.png")

def plot_confusion_matrices(data):
    """2x2 grid of confusion matrices."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    configs = [
        ("dino_original", "DINOv2 — Original"),
        ("dino_contour", "DINOv2 — Contour"),
        ("yolo_original", "YOLOv8n — Original"),
        ("yolo_contour", "YOLOv8n — Contour"),
    ]

    for ax, (key, title) in zip(axes.flat, configs):
        cm = np.array(data[key]["confusion_matrix"])
        # Normalize by row
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
        im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=100)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(CLASSES, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        # Add text
        for i in range(10):
            for j in range(10):
                val = cm_norm[i, j]
                color = "white" if val > 50 else "black"
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                        fontsize=7, color=color)

    fig.colorbar(im, ax=axes, shrink=0.6, label="Accuracy (%)")
    plt.suptitle("Normalized Confusion Matrices", fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrices.png", dpi=150, bbox_inches="tight")
    print("Saved confusion_matrices.png")

def plot_per_class_accuracy(data):
    """Grouped bar chart: per-class accuracy for all 4 configs."""
    fig, ax = plt.subplots(figsize=(14, 6))

    configs = [
        ("dino_original", "DINOv2 Original", "#4C72B0"),
        ("dino_contour", "DINOv2 Contour", "#8DA0CB"),
        ("yolo_original", "YOLOv8n Original", "#DD8452"),
        ("yolo_contour", "YOLOv8n Contour", "#E5BA73"),
    ]

    x = np.arange(len(CLASSES))
    w = 0.2
    for i, (key, label, color) in enumerate(configs):
        report = data[key]["classification_report"]
        accs = [report[cls]["recall"] * 100 for cls in CLASSES]
        ax.bar(x + i*w - 1.5*w, accs, w, label=label, color=color)

    ax.set_ylabel("Recall / Per-class Accuracy (%)", fontsize=11)
    ax.set_title("Per-Class Accuracy Comparison", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, fontsize=10)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "per_class_accuracy.png", dpi=150, bbox_inches="tight")
    print("Saved per_class_accuracy.png")

def plot_training_curves(data):
    """Training loss and validation accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    configs = [
        ("dino_original", "DINOv2 Original", "#4C72B0", "-"),
        ("dino_contour", "DINOv2 Contour", "#8DA0CB", "--"),
        ("yolo_original", "YOLOv8n Original", "#DD8452", "-"),
        ("yolo_contour", "YOLOv8n Contour", "#E5BA73", "--"),
    ]

    for key, label, color, ls in configs:
        h = data[key].get("history", {})
        if h.get("train_loss"):
            epochs = range(1, len(h["train_loss"]) + 1)
            ax1.plot(epochs, h["train_loss"], label=label, color=color, ls=ls, lw=2)
        if h.get("test_acc"):
            epochs = range(1, len(h["test_acc"]) + 1)
            vals = [v * 100 if max(h["test_acc"]) <= 1 else v for v in h["test_acc"]]
            ax2.plot(epochs, vals, label=label, color=color, ls=ls, lw=2)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss")
    ax1.set_title("Training Loss Over Epochs")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Validation Accuracy Over Epochs")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "training_curves.png", dpi=150, bbox_inches="tight")
    print("Saved training_curves.png")

if __name__ == "__main__":
    data = load_results()
    plot_accuracy_comparison(data)
    plot_confusion_matrices(data)
    plot_per_class_accuracy(data)
    plot_training_curves(data)
    print("\nAll plots saved to results/")
