"""
Prepare the Overwatch frames dataset for DINO and YOLO experiments.
- Splits each character class into train/test (80/20)
- Creates original and contour versions
- Handles both 'normal' and 'occluded' subfolders (merged into the class)
Output structure:
  data/overwatch_original/{train,test}/<character>/
  data/overwatch_contour/{train,test}/<character>/
"""
import os
import glob
import random
import shutil
from pathlib import Path
from PIL import Image
import cv2
import numpy as np

random.seed(42)

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
SRC = DATA_ROOT / "overwatch_frames"
OUT_ORIG = DATA_ROOT / "overwatch_original"
OUT_CONT = DATA_ROOT / "overwatch_contour"

IMG_SIZE = 224  # resize for consistency

CLASSES = sorted([d.name for d in SRC.iterdir() if d.is_dir() and not d.name.startswith('.')])
print(f"Classes ({len(CLASSES)}): {CLASSES}")

for cls in CLASSES:
    # Collect all images from normal + occluded
    cls_dir = SRC / cls
    all_images = []
    for sub in ["normal", "occluded"]:
        sub_dir = cls_dir / sub
        if sub_dir.exists():
            imgs = list(sub_dir.glob("*.png")) + list(sub_dir.glob("*.jpg"))
            all_images.extend(imgs)

    random.shuffle(all_images)
    split_idx = int(len(all_images) * 0.8)
    train_imgs = all_images[:split_idx]
    test_imgs = all_images[split_idx:]

    print(f"  {cls}: {len(all_images)} total -> {len(train_imgs)} train, {len(test_imgs)} test")

    for split_name, split_imgs in [("train", train_imgs), ("test", test_imgs)]:
        orig_dir = OUT_ORIG / split_name / cls
        cont_dir = OUT_CONT / split_name / cls
        orig_dir.mkdir(parents=True, exist_ok=True)
        cont_dir.mkdir(parents=True, exist_ok=True)

        for i, img_path in enumerate(split_imgs):
            fname = f"{i:04d}.png"

            # Read and resize original
            img = Image.open(img_path).convert("RGB")
            img_resized = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
            img_resized.save(orig_dir / fname)

            # Generate contour
            img_np = np.array(img_resized)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            # Save as 3-channel for model compatibility
            edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            Image.fromarray(edges_rgb).save(cont_dir / fname)

# Print summary
for variant in ["overwatch_original", "overwatch_contour"]:
    for split in ["train", "test"]:
        d = DATA_ROOT / variant / split
        total = sum(len(list((d / c).glob("*.png"))) for c in CLASSES)
        print(f"{variant}/{split}: {total} images")

print("\nDone!")
