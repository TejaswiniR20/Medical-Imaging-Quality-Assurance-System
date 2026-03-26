import pandas as pd
import os
import shutil


CSV_PATH = "data/Data_Entry_2017.csv"
IMAGES_DIR = "data/images/"
OUTPUT_DIR = "data/processed/"

def prepare_dataset():
    # Load the CSV
    if not os.path.exists(CSV_PATH):
        print(f"Error: CSV file not found at {CSV_PATH}")
        return

    print(f"Loading CSV from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)

    normal_df = df[df["Finding Labels"] == "No Finding"]
    pneumonia_df = df[df["Finding Labels"].str.contains("Pneumonia", case=False, na=False)]

    print(f"Total Normal images found: {len(normal_df)}")
    print(f"Total Pneumonia images found: {len(pneumonia_df)}")

    # Create output folders for train, val, and test
    splits = ["train", "val", "test"]
    labels = ["NORMAL", "PNEUMONIA"]
    for split in splits:
        for label in labels:
            folder_path = os.path.join(OUTPUT_DIR, split, label)
            os.makedirs(folder_path, exist_ok=True)

    print("Directory structure created successfully at:", OUTPUT_DIR)

    def split_and_copy(data_df, label_folder):
        
        data_df = data_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Split (70% Train, 15% Val, 15% Test)
        num_images = len(data_df)
        train_end = int(0.7 * num_images)
        val_end = int(0.85 * num_images)
        
        splits_map = {
            "train": data_df[:train_end],
            "val": data_df[train_end:val_end],
            "test": data_df[val_end:]
        }
        
        for split, split_df in splits_map.items():
            print(f"Processing {len(split_df)} images for {split}/{label_folder}...")
            count = 0
            for idx, row in split_df.iterrows():
                filename = row["Image Index"]
                src_path = os.path.join(IMAGES_DIR, filename)
                dst_path = os.path.join(OUTPUT_DIR, split, label_folder, filename)
                
                
                if os.path.exists(src_path):
                    if not os.path.exists(dst_path):
                        shutil.copy(src_path, dst_path)
                    count += 1
                
                
                if count % 1000 == 0 and count > 0:
                    print(f"  - Progress ({split}/{label_folder}): {count}/{len(split_df)} copied.")
            
            print(f"Finished: {count} images copied to {split}/{label_folder}.")

    
    split_and_copy(normal_df, "NORMAL")
    split_and_copy(pneumonia_df, "PNEUMONIA")

    print(f"\nDataset preparation complete!")
    print(f"Train/Val/Test directories are ready in: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    prepare_dataset()