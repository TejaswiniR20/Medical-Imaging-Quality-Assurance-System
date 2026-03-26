import pandas as pd
import os
import shutil

# Paths - each teammate updates these to match their local setup
CSV_PATH = "data/Data_Entry_2017.csv"
IMAGES_DIR = "data/images/"
OUTPUT_DIR = "data/processed/"

def prepare_dataset():
    # Load the CSV
    df = pd.read_csv(CSV_PATH)

    # Filter Normal and Pneumonia only
    normal_df = df[df["Finding Labels"] == "No Finding"]
    pneumonia_df = df[df["Finding Labels"].str.contains("Pneumonia", na=False)]

    print(f"Normal images found: {len(normal_df)}")
    print(f"Pneumonia images found: {len(pneumonia_df)}")

    # Create output folders
    for split in ["train", "val", "test"]:
        for label in ["NORMAL", "PNEUMONIA"]:
            os.makedirs(f"{OUTPUT_DIR}/{split}/{label}", exist_ok=True)

    print("Folder structure created successfully!")
    print(f"Output directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    prepare_dataset()