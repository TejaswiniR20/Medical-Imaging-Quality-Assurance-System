# Dataset

This project uses the NIH ChestX-ray14 dataset.

## Download
https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/37178474737

Download the following:
- All `images_00x.tar.gz` files (12 parts)
- `Data_Entry_2017.csv` (contains image labels)

## After downloading, organize like this:
```
data/
  images/
    00000001_000.png
    00000002_000.png
    ...
  Data_Entry_2017.csv
```

---

# Data Processing Pipeline

We have implemented a three-step pipeline to prepare the data for training and evaluation.

### Step 1: Dataset Preparation
**Script:** `preprocessing/prepare_dataset.py`

Filters the NIH dataset for "No Finding" and "Pneumonia" cases and splits them into training, validation, and test sets.

**Run Command:**
```bash
python preprocessing/prepare_dataset.py
```

**Output:**
- **Location:** `data/processed/`
- **Structure:**
  - `data/processed/train/NORMAL/` & `PNEUMONIA/`
  - `data/processed/val/NORMAL/` & `PNEUMONIA/`
  - `data/processed/test/NORMAL/` & `PNEUMONIA/`

### Step 2: Image Enhancement (Standardization)
**Script:** `preprocessing/image_enhancement.py`

Reads images from `data/processed/`, converts them to **Grayscale**, and resizes them to **224x224** pixels.

**Run Command:**
```bash
python preprocessing/image_enhancement.py
```

**Output:**
- **Location:** `data/enhanced/` (Maintains split folder structure)

### Step 3: Contrast Enhancement (CLAHE)
**Script:** `preprocessing/clahe.py`

Applies **Contrast Limited Adaptive Histogram Equalization (CLAHE)** to the enhanced images to improve feature visibility for the deep learning model.

**Run Command:**
```bash
python preprocessing/clahe.py
```

**Output:**
- **Location:** `data/clahe_Result/` (Maintains split folder structure)

---

# Data Storage Architecture Summary

| Step | Source Folder | Output Folder | Action |
| --- | --- | --- | --- |
| **Preparation** | `data/images/` | `data/processed/` | Filtering & Splitting |
| **Enhancement** | `data/processed/` | `data/enhanced/` | Grayscale & Resize |
| **CLAHE** | `data/enhanced/` | `data/clahe_Result/` | Contrast Enhancement |