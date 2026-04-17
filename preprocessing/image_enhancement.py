import os
import cv2
import numpy as np
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

INPUT_BASE_DIR = "data/processed/"
OUTPUT_BASE_DIR = "data/enhanced/"
TARGET_SIZE = (224, 224)

def enhance_image(input_path, output_path):
    """
    Standarizes an image by converting to grayscale and resizing to 224x224.
    
    Args:
        input_path (str): Path to the source image.
        output_path (str): Path where the processed image will be saved.
    """
    try:
        
        
        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            logging.warning(f"Failed to read image: {input_path}")
            return False

        
        img_resized = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)

        
        assert img_resized.ndim == 2 or img_resized.shape[-1] == 1

        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, img_resized)
        
        return True

    except Exception as e:
        logging.error(f"Error processing image {input_path}: {str(e)}")
        return False

def process_dataset():
    """
    Iterates through the dataset hierarchy and applies standardization to each image.
    Maintains the folder structure (train/val/test/NORMAL/PNEUMONIA).
    """
    if not os.path.exists(INPUT_BASE_DIR):
        logging.error(f"Input directory not found: {INPUT_BASE_DIR}")
        return

    splits = ["train", "val", "test"]
    labels = ["NORMAL", "PNEUMONIA"]

    total_skipped = 0
    total_processed = 0
    total_failed = 0

    for split in splits:
        for label in labels:
            input_dir = os.path.join(INPUT_BASE_DIR, split, label)
            output_dir = os.path.join(OUTPUT_BASE_DIR, split, label)

            if not os.path.exists(input_dir):
                logging.warning(f"Input subdirectory does not exist: {input_dir}")
                continue

            os.makedirs(output_dir, exist_ok=True)
            
            filenames = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            logging.info(f"Processing {len(filenames)} images from {input_dir}...")

            count = 0
            for filename in filenames:
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, filename)

                
                if os.path.exists(output_path):
                    total_skipped += 1
                    continue

                if enhance_image(input_path, output_path):
                    total_processed += 1
                    count += 1
                else:
                    total_failed += 1

                # Log progress every 1000 images
                if count % 1000 == 0 and count > 0:
                    logging.info(f"  - {split}/{label}: Processed {count}/{len(filenames)}")

            logging.info(f"Finished {split}/{label}: Total {count} images processed.")

    logging.info("\n--- Overall Summary ---")
    logging.info(f"Total Processed: {total_processed}")
    logging.info(f"Total Skipped (already exist): {total_skipped}")
    logging.info(f"Total Failed: {total_failed}")

if __name__ == "__main__":
    process_dataset()
