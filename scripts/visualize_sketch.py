"""
Generate plots for DINO vs YOLO on TU-Berlin Sketch Dataset (250 classes).
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

DATA_DIR = DATA_ROOT / "sketch"
CLASSES = sorted([d.name for d in (DATA_DIR / "train").iterdir() if d.is_dir()])


def load_results():
    data = {}
    for name in ["dino_sketch", "yolo_sketch"]:
        path = RESULTS_DIR / f"{name}.json"
        with open(path) as f:
            data[name] = json.load(f)
    return data


def plot_accuracy_comparison(data):
    fig, ax = plt.subplots(figsize=(6, 5))
    models = ["DINOv2", "YOLOv8n-cls"]
    accs = [data["dino_sketch"]["final_test_accuracy"] * 100,
            data["yolo_sketch"]["final_test_accuracy"] * 100]

    colors = ["#4C72B0", "#DD8452"]
    bars = ax.bar(models, accs, color=colors, width=0.5)

    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("TU-Berlin Sketch Classification (250 classes)", fontsize=13)
    ax.set_ylim(0, max(accs) * 1.3)
    chance = 100 / len(CLASSES)
    ax.axhline(y=chance, color='gray', linestyle='--', alpha=0.5)
    ax.text(1.3, chance + 0.5, f'Random ({chance:.1f}%)', fontsize=9, color='gray')

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "sketch_accuracy_comparison.png", dpi=150, bbox_inches="tight")
    print("Saved sketch_accuracy_comparison.png")


def plot_training_curves(data):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    configs = [
        ("dino_sketch", "DINOv2", "#4C72B0", "-"),
        ("yolo_sketch", "YOLOv8n-cls", "#DD8452", "-"),
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
    plt.savefig(RESULTS_DIR / "sketch_training_curves.png", dpi=150, bbox_inches="tight")
    print("Saved sketch_training_curves.png")


def plot_top_bottom_classes(data):
    """Plot top-20 and bottom-20 classes by accuracy for each model."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    configs = [
        ("dino_sketch", "DINOv2"),
        ("yolo_sketch", "YOLOv8n-cls"),
    ]

    for col, (key, model_name) in enumerate(configs):
        report = data[key]["classification_report"]
        class_accs = [(cls, report[cls]["recall"] * 100) for cls in CLASSES if cls in report]
        class_accs.sort(key=lambda x: x[1], reverse=True)

        # Top 20
        top20 = class_accs[:20]
        ax = axes[0][col]
        names, accs = zip(*top20)
        ax.barh(range(len(names)), accs, color="#4C72B0" if col == 0 else "#DD8452")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("Recall (%)")
        ax.set_title(f"{model_name}: Top 20 Classes", fontsize=12, fontweight="bold")
        ax.set_xlim(0, 105)
        ax.invert_yaxis()

        # Bottom 20
        bottom20 = class_accs[-20:]
        ax = axes[1][col]
        names, accs = zip(*bottom20)
        ax.barh(range(len(names)), accs, color="#4C72B0" if col == 0 else "#DD8452")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("Recall (%)")
        ax.set_title(f"{model_name}: Bottom 20 Classes", fontsize=12, fontweight="bold")
        ax.set_xlim(0, max(accs) * 1.5 if max(accs) > 0 else 10)
        ax.invert_yaxis()

    plt.suptitle("TU-Berlin Sketch: Per-Class Accuracy (Top & Bottom 20)", fontsize=15)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "sketch_per_class_accuracy.png", dpi=150, bbox_inches="tight")
    print("Saved sketch_per_class_accuracy.png")


def plot_showcase():
    """Show sample sketches from a selection of categories."""
    # Pick 20 evenly spaced classes for showcase
    step = max(1, len(CLASSES) // 20)
    sample_classes = CLASSES[::step][:20]

    fig, axes = plt.subplots(2, 10, figsize=(20, 5))
    fig.suptitle("TU-Berlin Sketch Dataset Samples (20 of 250 categories)", fontsize=14, y=1.02)

    for i, cls in enumerate(sample_classes):
        row, col = i // 10, i % 10
        test_dir = DATA_DIR / "test" / cls
        imgs = sorted(test_dir.glob("*.png"))
        if imgs:
            axes[row][col].imshow(Image.open(imgs[0]))
        axes[row][col].set_title(cls.replace("_", " "), fontsize=7)
        axes[row][col].axis("off")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "sketch_showcase_grid.png", dpi=150, bbox_inches="tight")
    print("Saved sketch_showcase_grid.png")


if __name__ == "__main__":
    data = load_results()
    plot_accuracy_comparison(data)
    plot_training_curves(data)
    plot_top_bottom_classes(data)
    plot_showcase()
    print("\nAll TU-Berlin Sketch plots saved!")
