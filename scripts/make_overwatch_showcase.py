"""
Generate a larger Overwatch showcase grid: 2 rows x 7 columns.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
CLASSES = sorted([d.name for d in (DATA_ROOT / "overwatch_original" / "train").iterdir() if d.is_dir()])

fig, axes = plt.subplots(2, 7, figsize=(21, 7))
fig.suptitle("Overwatch Character Samples (14 Classes)", fontsize=18, y=1.0)

for i, cls in enumerate(CLASSES):
    row, col = divmod(i, 7)
    orig_dir = DATA_ROOT / "overwatch_original" / "test" / cls
    imgs = sorted(orig_dir.glob("*.png"))
    if imgs:
        axes[row, col].imshow(Image.open(imgs[0]))
    axes[row, col].set_title(cls, fontsize=12, fontweight="bold")
    axes[row, col].axis("off")

plt.tight_layout()
plt.savefig(RESULTS_DIR / "overwatch_showcase_grid.png", dpi=150, bbox_inches="tight")
print("Saved overwatch_showcase_grid.png")
