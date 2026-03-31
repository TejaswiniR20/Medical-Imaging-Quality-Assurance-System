import cv2
import numpy as np
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    
    Args:
        image (numpy.ndarray): Input grayscale image.
        clip_limit (float): Threshold for contrast limiting.
        tile_grid_size (tuple): Size of grid for histogram equalization.
        
    Returns:
        numpy.ndarray: Contrast enhanced image.
    """
    try:
        if image is None:
            return None
            
        # Initialize CLAHE object
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        
        # Apply CLAHE
        enhanced_img = clahe.apply(image)
        
        return enhanced_img
    except Exception as e:
        logging.error(f"Error applying CLAHE: {str(e)}")
        return None

def process_dataset(input_base, output_base):
    """
    Recursively apply CLAHE to all images in the dataset.
    
    Args:
        input_base (str): Path to enhanced dataset (output of enhancement step).
        output_base (str): Path where final CLAHE images will be stored.
    """
    splits = ["train", "val", "test"]
    labels = ["NORMAL", "PNEUMONIA"]

    total_processed = 0
    total_skipped = 0
    total_failed = 0

    for split in splits:
        for label in labels:
            input_dir = os.path.join(input_base, split, label)
            output_dir = os.path.join(output_base, split, label)

            if not os.path.exists(input_dir):
                logging.warning(f"Directory {input_dir} does not exist. Skipping.")
                continue

            
            os.makedirs(output_dir, exist_ok=True)

            filenames = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            logging.info(f"Applying CLAHE to {len(filenames)} images in {input_dir}...")

            count = 0
            for filename in filenames:
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, filename)

                
                if os.path.exists(output_path):
                    total_skipped += 1
                    continue

                
                img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
                
                if img is None:
                    logging.error(f"Could not read image: {input_path}")
                    total_failed += 1
                    continue

                
                enhanced_img = apply_clahe(img)

                if enhanced_img is not None:
                    
                    cv2.imwrite(output_path, enhanced_img)
                    total_processed += 1
                    count += 1
                else:
                    total_failed += 1

                
                if count % 1000 == 0 and count > 0:
                    logging.info(f"  - {split}/{label}: Enhanced {count}/{len(filenames)}")

    logging.info("CLAHE Processing Complete!")
    logging.info(f"Total Processed: {total_processed}")
    logging.info(f"Total Skipped: {total_skipped}")
    logging.info(f"Total Failed: {total_failed}")

if __name__ == "__main__":

    INPUT_DATA_DIR = os.path.join("data", "enhanced")
    OUTPUT_DATA_DIR = os.path.join("data", "clahe_Result")
    
    
    process_dataset(INPUT_DATA_DIR, OUTPUT_DATA_DIR)
