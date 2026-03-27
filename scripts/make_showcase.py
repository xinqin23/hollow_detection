"""
Create a showcase grid: original vs contour for one sample per class.
Saves to results/showcase_grid.png
"""
import os
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent.parent / "results"

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

def get_first_image(variant, cls_name):
    d = DATA_ROOT / f"cifar10_{variant}" / "test" / cls_name
    imgs = sorted(os.listdir(d))
    return d / imgs[0]

fig, axes = plt.subplots(2, 10, figsize=(20, 5))
fig.suptitle("CIFAR-10: Original (top) vs Canny Contour (bottom)", fontsize=16, y=1.02)

for i, cls in enumerate(CLASSES):
    orig = Image.open(get_first_image("original", cls))
    cont = Image.open(get_first_image("contour", cls))

    axes[0, i].imshow(orig)
    axes[0, i].set_title(cls, fontsize=10)
    axes[0, i].axis("off")

    axes[1, i].imshow(cont, cmap="gray")
    axes[1, i].axis("off")

axes[0, 0].set_ylabel("Original", fontsize=12, rotation=90, labelpad=10)
axes[1, 0].set_ylabel("Contour", fontsize=12, rotation=90, labelpad=10)

plt.tight_layout()
plt.savefig(OUT_DIR / "showcase_grid.png", dpi=150, bbox_inches="tight")
print(f"Saved to {OUT_DIR / 'showcase_grid.png'}")
