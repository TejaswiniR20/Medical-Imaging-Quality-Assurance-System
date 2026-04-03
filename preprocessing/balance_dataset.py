import os
import random
import shutil

# =========================
# PATHS
# =========================
src_base = "data/clahe_Result"
dst_base = "data/clahe_balanced"

valid_ext = (".png", ".jpg", ".jpeg")

# =========================
# TARGET BALANCE
# =========================
balance_limits = {
    "train": 1001,
    "val": 215
}

# =========================
# HELPER FUNCTION
# =========================
def get_images(folder):
    return [f for f in os.listdir(folder) if f.lower().endswith(valid_ext)]

# =========================
# MAIN FUNCTION
# =========================
def process_split(split):
    print(f"\n📁 Processing {split}...")

    src_split = os.path.join(src_base, split)
    dst_split = os.path.join(dst_base, split)

    # recreate split folder
    if os.path.exists(dst_split):
        shutil.rmtree(dst_split)

    os.makedirs(dst_split)

    # paths
    src_normal = os.path.join(src_split, "NORMAL")
    src_pneumonia = os.path.join(src_split, "PNEUMONIA")

    dst_normal = os.path.join(dst_split, "NORMAL")
    dst_pneumonia = os.path.join(dst_split, "PNEUMONIA")

    os.makedirs(dst_normal)
    os.makedirs(dst_pneumonia)

    # =========================
    # COPY PNEUMONIA (FULL)
    # =========================
    pneu_images = get_images(src_pneumonia)

    print(f"🦠 Copying PNEUMONIA: {len(pneu_images)} images")

    for img in pneu_images:
        shutil.copy2(
            os.path.join(src_pneumonia, img),
            os.path.join(dst_pneumonia, img)
        )

    # =========================
    # NORMAL HANDLING
    # =========================
    normal_images = get_images(src_normal)

    if split in balance_limits:
        limit = balance_limits[split]
        limit = min(limit, len(normal_images))

        print(f"⚖ Reducing NORMAL to: {limit}")

        selected = random.sample(normal_images, limit)
    else:
        # TEST → copy all
        print(f"📦 Copying FULL NORMAL: {len(normal_images)}")
        selected = normal_images

    for img in selected:
        shutil.copy2(
            os.path.join(src_normal, img),
            os.path.join(dst_normal, img)
        )

    print(f"✅ Done {split}")


# =========================
# RUN ALL
# =========================
for split in ["train", "val", "test"]:
    process_split(split)

print("\n🎯 BALANCED DATASET READY at:", dst_base)