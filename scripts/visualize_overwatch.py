"""
Generate plots for DINO vs YOLO on Overwatch frames (original only, no contour).
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DATA_ROOT = Path(__file__).resolve().parent.parent / "data"

CLASSES = sorted([d.name for d in (DATA_ROOT / "overwatch_original" / "train").iterdir() if d.is_dir()])


def load_results():
    data = {}
    for name in ["dino_overwatch_original", "yolo_overwatch_original"]:
        with open(RESULTS_DIR / f"{name}.json") as f:
            data[name] = json.load(f)
    return data


def plot_accuracy_comparison(data):
    fig, ax = plt.subplots(figsize=(6, 5))
    models = ["DINOv2", "YOLOv8n-cls"]
    accs = [data["dino_overwatch_original"]["final_test_accuracy"] * 100,
            data["yolo_overwatch_original"]["final_test_accuracy"] * 100]

    colors = ["#4C72B0", "#DD8452"]
    bars = ax.bar(models, accs, color=colors, width=0.5)

    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Overwatch Character Classification (14 classes)", fontsize=13)
    ax.set_ylim(0, 80)
    ax.axhline(y=100/len(CLASSES), color='gray', linestyle='--', alpha=0.5)
    ax.text(1.3, 100/len(CLASSES)+1, f'Random ({100/len(CLASSES):.1f}%)', fontsize=9, color='gray')

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "overwatch_accuracy_comparison.png", dpi=150, bbox_inches="tight")
    print("Saved overwatch_accuracy_comparison.png")


def plot_confusion_matrices(data):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    configs = [
        ("dino_overwatch_original", "DINOv2 (frozen + linear)"),
        ("yolo_overwatch_original", "YOLOv8n-cls (fine-tuned)"),
    ]

    for ax, (key, title) in zip(axes, configs):
        cm = np.array(data[key]["confusion_matrix"])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
        im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=100)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(range(len(CLASSES)))
        ax.set_yticks(range(len(CLASSES)))
        ax.set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(CLASSES, fontsize=7)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                val = cm_norm[i, j]
                color = "white" if val > 50 else "black"
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=6, color=color)

    fig.colorbar(im, ax=axes, shrink=0.6, label="Accuracy (%)")
    plt.suptitle("Overwatch: Normalized Confusion Matrices", fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "overwatch_confusion_matrices.png", dpi=150, bbox_inches="tight")
    print("Saved overwatch_confusion_matrices.png")


def plot_per_class_accuracy(data):
    fig, ax = plt.subplots(figsize=(14, 6))
    configs = [
        ("dino_overwatch_original", "DINOv2 (frozen + linear)", "#4C72B0"),
        ("yolo_overwatch_original", "YOLOv8n-cls (fine-tuned)", "#DD8452"),
    ]

    x = np.arange(len(CLASSES))
    w = 0.35
    for i, (key, label, color) in enumerate(configs):
        report = data[key]["classification_report"]
        accs = [report[cls]["recall"] * 100 for cls in CLASSES]
        ax.bar(x + i*w - w/2, accs, w, label=label, color=color)

    ax.set_ylabel("Recall / Per-class Accuracy (%)", fontsize=11)
    ax.set_title("Overwatch: Per-Class Accuracy Comparison", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, fontsize=9, rotation=30, ha="right")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "overwatch_per_class_accuracy.png", dpi=150, bbox_inches="tight")
    print("Saved overwatch_per_class_accuracy.png")


def plot_training_curves(data):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    configs = [
        ("dino_overwatch_original", "DINOv2", "#4C72B0", "-"),
        ("yolo_overwatch_original", "YOLOv8n-cls", "#DD8452", "-"),
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

    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Training Loss")
    ax1.set_title("Training Loss Over Epochs"); ax1.legend(fontsize=10); ax1.grid(alpha=0.3)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Validation Accuracy Over Epochs"); ax2.legend(fontsize=10); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "overwatch_training_curves.png", dpi=150, bbox_inches="tight")
    print("Saved overwatch_training_curves.png")


def plot_showcase():
    fig, axes = plt.subplots(1, len(CLASSES), figsize=(24, 3))
    fig.suptitle("Overwatch Character Samples", fontsize=16, y=1.05)

    for i, cls in enumerate(CLASSES):
        orig_dir = DATA_ROOT / "overwatch_original" / "test" / cls
        imgs = sorted(orig_dir.glob("*.png"))
        if imgs:
            axes[i].imshow(Image.open(imgs[0]))
        axes[i].set_title(cls, fontsize=8)
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "overwatch_showcase_grid.png", dpi=150, bbox_inches="tight")
    print("Saved overwatch_showcase_grid.png")


if __name__ == "__main__":
    data = load_results()
    plot_accuracy_comparison(data)
    plot_confusion_matrices(data)
    plot_per_class_accuracy(data)
    plot_training_curves(data)
    plot_showcase()
    print("\nAll Overwatch plots saved!")
