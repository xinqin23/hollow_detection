"""
Generate comparison plots for DINO vs YOLO on Overwatch frames (original vs contour).
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DATA_ROOT = Path(__file__).resolve().parent.parent / "data"

CLASSES = sorted([d.name for d in (DATA_ROOT / "overwatch_original" / "train").iterdir() if d.is_dir()])

def load_results():
    data = {}
    for name in ["dino_overwatch_original", "dino_overwatch_contour",
                  "yolo_overwatch_original", "yolo_overwatch_contour"]:
        with open(RESULTS_DIR / f"{name}.json") as f:
            data[name] = json.load(f)
    return data

def plot_accuracy_comparison(data):
    fig, ax = plt.subplots(figsize=(8, 5))
    models = ["DINOv2", "YOLOv8n-cls"]
    original = [data["dino_overwatch_original"]["final_test_accuracy"],
                data["yolo_overwatch_original"]["final_test_accuracy"]]
    contour = [data["dino_overwatch_contour"]["final_test_accuracy"],
               data["yolo_overwatch_contour"]["final_test_accuracy"]]

    x = np.arange(len(models))
    w = 0.3
    bars1 = ax.bar(x - w/2, [v*100 for v in original], w, label="Original", color="#4C72B0")
    bars2 = ax.bar(x + w/2, [v*100 for v in contour], w, label="Contour", color="#DD8452")

    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Overwatch Character Classification: Original vs Contour", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)
    ax.set_ylim(0, 80)
    ax.legend(fontsize=11)
    # Random baseline
    ax.axhline(y=100/len(CLASSES), color='gray', linestyle='--', alpha=0.5, label=f'Random ({100/len(CLASSES):.1f}%)')
    ax.legend(fontsize=10)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "overwatch_accuracy_comparison.png", dpi=150, bbox_inches="tight")
    print("Saved overwatch_accuracy_comparison.png")

def plot_confusion_matrices(data):
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    configs = [
        ("dino_overwatch_original", "DINOv2 — Original"),
        ("dino_overwatch_contour", "DINOv2 — Contour"),
        ("yolo_overwatch_original", "YOLOv8n — Original"),
        ("yolo_overwatch_contour", "YOLOv8n — Contour"),
    ]

    for ax, (key, title) in zip(axes.flat, configs):
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
    fig, ax = plt.subplots(figsize=(16, 6))
    configs = [
        ("dino_overwatch_original", "DINOv2 Original", "#4C72B0"),
        ("dino_overwatch_contour", "DINOv2 Contour", "#8DA0CB"),
        ("yolo_overwatch_original", "YOLOv8n Original", "#DD8452"),
        ("yolo_overwatch_contour", "YOLOv8n Contour", "#E5BA73"),
    ]

    x = np.arange(len(CLASSES))
    w = 0.2
    for i, (key, label, color) in enumerate(configs):
        report = data[key]["classification_report"]
        accs = [report[cls]["recall"] * 100 for cls in CLASSES]
        ax.bar(x + i*w - 1.5*w, accs, w, label=label, color=color)

    ax.set_ylabel("Recall / Per-class Accuracy (%)", fontsize=11)
    ax.set_title("Overwatch: Per-Class Accuracy Comparison", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, fontsize=9, rotation=30, ha="right")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "overwatch_per_class_accuracy.png", dpi=150, bbox_inches="tight")
    print("Saved overwatch_per_class_accuracy.png")

def plot_training_curves(data):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    configs = [
        ("dino_overwatch_original", "DINOv2 Original", "#4C72B0", "-"),
        ("dino_overwatch_contour", "DINOv2 Contour", "#8DA0CB", "--"),
        ("yolo_overwatch_original", "YOLOv8n Original", "#DD8452", "-"),
        ("yolo_overwatch_contour", "YOLOv8n Contour", "#E5BA73", "--"),
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
    ax1.set_title("Training Loss Over Epochs"); ax1.legend(fontsize=9); ax1.grid(alpha=0.3)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Validation Accuracy Over Epochs"); ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "overwatch_training_curves.png", dpi=150, bbox_inches="tight")
    print("Saved overwatch_training_curves.png")

def plot_showcase():
    """Show one original+contour sample per class."""
    from PIL import Image
    fig, axes = plt.subplots(2, len(CLASSES), figsize=(24, 5))
    fig.suptitle("Overwatch: Original (top) vs Canny Contour (bottom)", fontsize=16, y=1.02)

    for i, cls in enumerate(CLASSES):
        orig_dir = DATA_ROOT / "overwatch_original" / "test" / cls
        cont_dir = DATA_ROOT / "overwatch_contour" / "test" / cls
        imgs = sorted(orig_dir.glob("*.png"))
        if imgs:
            axes[0, i].imshow(Image.open(imgs[0]))
            axes[1, i].imshow(Image.open(cont_dir / imgs[0].name))
        axes[0, i].set_title(cls, fontsize=8)
        axes[0, i].axis("off")
        axes[1, i].axis("off")

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
