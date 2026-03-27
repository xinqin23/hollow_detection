"""
Prepare the TU-Berlin Sketch Dataset (Eitz et al. 2012) for DINO and YOLO experiments.
- Downloads 20,000 human sketches across 250 object categories (~80 per category)
- Uses HuggingFace datasets (kmewhort/tu-berlin-png)
- Splits into train/test (80/20)
- Converts to 3-channel RGB at 224x224 for model compatibility

Output structure:
  data/sketch/{train,test}/<category>/
"""
import random
from pathlib import Path
from PIL import Image
from datasets import load_dataset

random.seed(42)

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = DATA_ROOT / "sketch"
IMG_SIZE = 224


def prepare_dataset():
    print("Loading TU-Berlin sketch dataset from HuggingFace...")
    ds = load_dataset("kmewhort/tu-berlin-png")

    # Combine train and test splits, then re-split ourselves (original test is tiny)
    all_samples = []
    for split_name in ds:
        for sample in ds[split_name]:
            all_samples.append((sample["image"], sample["label"]))
    print(f"Total samples: {len(all_samples)}")

    # Get class names from the dataset features
    label_names = ds["train"].features["label"].names
    print(f"Classes ({len(label_names)}): {label_names[:10]}... (showing first 10)")

    # Group by class
    class_images = {}
    for img, label_idx in all_samples:
        cls_name = label_names[label_idx]
        if cls_name not in class_images:
            class_images[cls_name] = []
        class_images[cls_name].append(img)

    total_train, total_test = 0, 0
    for cls_name in sorted(class_images.keys()):
        imgs = class_images[cls_name]
        random.shuffle(imgs)
        split_idx = int(len(imgs) * 0.8)
        train_imgs = imgs[:split_idx]
        test_imgs = imgs[split_idx:]

        for split_name, split_imgs in [("train", train_imgs), ("test", test_imgs)]:
            out_dir = OUT_DIR / split_name / cls_name
            out_dir.mkdir(parents=True, exist_ok=True)

            for i, img in enumerate(split_imgs):
                fname = f"{i:04d}.png"
                # Convert to RGB, resize
                img_rgb = img.convert("RGB")
                img_resized = img_rgb.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
                img_resized.save(out_dir / fname)

        total_train += len(train_imgs)
        total_test += len(test_imgs)

    print(f"\nTotal: {total_train} train, {total_test} test")
    return sorted(class_images.keys())


if __name__ == "__main__":
    classes = prepare_dataset()
    print(f"\nPrepared {len(classes)} classes in {OUT_DIR}")

    # Print summary
    for split in ["train", "test"]:
        d = OUT_DIR / split
        if d.exists():
            n_classes = len([c for c in d.iterdir() if c.is_dir()])
            n_imgs = sum(len(list(c.glob("*.png"))) for c in d.iterdir() if c.is_dir())
            print(f"  {split}: {n_classes} classes, {n_imgs} images")

    print("\nDone!")
