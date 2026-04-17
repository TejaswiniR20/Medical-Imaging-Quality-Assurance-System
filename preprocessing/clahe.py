"""
clahe.py — CLAHE image enhancement for chest X-rays.
Provides both single-image and batch-dataset processing utilities.
"""
import cv2
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0,
                tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).

    Args:
        image: Grayscale numpy array (H, W)
        clip_limit: Threshold for contrast limiting.
        tile_grid_size: Grid size for histogram equalization.

    Returns:
        Enhanced grayscale image as numpy array.
    """
    if image is None:
        return None
    try:
        clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe_obj.apply(image)
    except Exception as e:
        logging.error(f"Error applying CLAHE: {e}")
        return None


def apply_clahe_to_file(input_path: str, output_path: str) -> bool:
    """Apply CLAHE to a single image file and save the result."""
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        logging.error(f"Cannot read image: {input_path}")
        return False

    enhanced = apply_clahe(img)
    if enhanced is None:
        return False

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, enhanced)
    return True


def process_dataset(input_base: str, output_base: str):
    """
    Recursively apply CLAHE to all images in the dataset.

    Args:
        input_base: Path to enhanced dataset.
        output_base: Output path for CLAHE images.
    """
    splits = ["train", "val", "test"]
    labels = ["NORMAL", "PNEUMONIA"]
    total_processed = total_skipped = total_failed = 0

    for split in splits:
        for label in labels:
            input_dir  = os.path.join(input_base, split, label)
            output_dir = os.path.join(output_base, split, label)

            if not os.path.exists(input_dir):
                logging.warning(f"Directory {input_dir} does not exist. Skipping.")
                continue

            os.makedirs(output_dir, exist_ok=True)
            filenames = [f for f in os.listdir(input_dir)
                         if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            logging.info(f"Applying CLAHE to {len(filenames)} images in {input_dir}")

            for filename in filenames:
                in_path  = os.path.join(input_dir, filename)
                out_path = os.path.join(output_dir, filename)
                if os.path.exists(out_path):
                    total_skipped += 1
                    continue
                if apply_clahe_to_file(in_path, out_path):
                    total_processed += 1
                else:
                    total_failed += 1

    logging.info(f"CLAHE complete — processed: {total_processed}, "
                 f"skipped: {total_skipped}, failed: {total_failed}")


if __name__ == "__main__":
    process_dataset(
        input_base=os.path.join("data", "enhanced"),
        output_base=os.path.join("data", "clahe_Result"),
    )
