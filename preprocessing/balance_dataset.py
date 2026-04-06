import os
import random
import shutil

# =========================
# BASE CONFIG
# =========================
base_dir = "data/clahe_Result"

# limits for each split
limits = {
    "train": 5000,
    "val": 2000
}

valid_ext = (".png", ".jpg", ".jpeg")

# =========================
# FUNCTION
# =========================
def reduce_normal(split, limit):
    normal_dir = os.path.join(base_dir, split, "NORMAL")
    reduced_dir = os.path.join(base_dir, split, "NORMAL_REDUCED")

    print(f"\n📁 Processing: {split}")
    print("Source folder:", normal_dir)

    if not os.path.exists(normal_dir):
        print(f"❌ ERROR: {split} NORMAL folder not found!")
        return

    os.makedirs(reduced_dir, exist_ok=True)

    images = [img for img in os.listdir(normal_dir) if img.lower().endswith(valid_ext)]

    print("📊 Total NORMAL images found:", len(images))

    if len(images) == 0:
        print("❌ No images found!")
        return

    random.shuffle(images)
    selected_images = images[:limit]

    print(f"🚀 Copying {len(selected_images)} images...")

    for i, img in enumerate(selected_images):
        src = os.path.join(normal_dir, img)
        dst = os.path.join(reduced_dir, img)

        try:
            shutil.copy2(src, dst)
        except Exception as e:
            print(f"⚠ Skipped {img}: {e}")

        if (i + 1) % 500 == 0:
            print(f"✅ Copied {i+1}/{len(selected_images)}")

    print("🎯 DONE!")
    print(f"✔ {len(selected_images)} images saved in: {reduced_dir}")


# =========================
# RUN FOR BOTH TRAIN + VAL
# =========================
for split in ["train", "val"]:
    reduce_normal(split, limits[split])

print("\n✅ ALL DONE (TRAIN + VAL REDUCED)")