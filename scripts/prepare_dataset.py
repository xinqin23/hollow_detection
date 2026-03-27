"""
Download CIFAR-10 and generate hollow contour (edge-only) versions.
Saves both original and contour images in directory structure for YOLO/DINO.
"""
import os
import cv2
import numpy as np
from torchvision import datasets
from tqdm import tqdm
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
CIFAR_RAW = DATA_ROOT / "cifar10_raw"
ORIGINAL_DIR = DATA_ROOT / "cifar10_original"
CONTOUR_DIR = DATA_ROOT / "cifar10_contour"

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def make_contour(img_np: np.ndarray) -> np.ndarray:
    """Convert an RGB image to hollow contour (Canny edges, white on black)."""
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    # Slight blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 50, 150)
    # Convert back to 3-channel for model compatibility
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)


def save_split(dataset, split_name):
    """Save original and contour versions of a dataset split."""
    for idx in tqdm(range(len(dataset)), desc=f"Processing {split_name}"):
        img, label = dataset[idx]
        img_np = np.array(img)
        class_name = CLASSES[label]

        # Original
        orig_dir = ORIGINAL_DIR / split_name / class_name
        orig_dir.mkdir(parents=True, exist_ok=True)
        orig_path = orig_dir / f"{idx:05d}.png"
        cv2.imwrite(str(orig_path), cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

        # Contour
        cont_dir = CONTOUR_DIR / split_name / class_name
        cont_dir.mkdir(parents=True, exist_ok=True)
        contour_img = make_contour(img_np)
        cont_path = cont_dir / f"{idx:05d}.png"
        cv2.imwrite(str(cont_path), cv2.cvtColor(contour_img, cv2.COLOR_RGB2BGR))


def main():
    print("Downloading CIFAR-10...")
    train_ds = datasets.CIFAR10(root=str(CIFAR_RAW), train=True, download=True)
    test_ds = datasets.CIFAR10(root=str(CIFAR_RAW), train=False, download=True)

    print(f"Train: {len(train_ds)}, Test: {len(test_ds)}")

    save_split(train_ds, "train")
    save_split(test_ds, "test")

    # Print sample counts
    for variant, base in [("original", ORIGINAL_DIR), ("contour", CONTOUR_DIR)]:
        for split in ["train", "test"]:
            total = sum(
                len(list((base / split / c).glob("*.png"))) for c in CLASSES
            )
            print(f"{variant}/{split}: {total} images")

    print("Done! Dataset saved to:", DATA_ROOT)


if __name__ == "__main__":
    main()
