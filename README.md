# Medical Imaging Quality Assurance System

**Deep Learning + Computer Vision + Healthcare AI**

## Problem Statement

Medical imaging such as chest X-rays is commonly used to detect lung diseases like pneumonia. While deep learning models can automatically analyze these images, many systems only give a prediction without showing where the disease is present, and sometimes the predictions may not be reliable. This project aims to develop a Medical Imaging Quality Assurance System that uses a pretrained deep learning model to detect pneumonia in chest X-ray images, highlight the affected region in the image, and evaluate the confidence of the prediction. If the model’s confidence is low, the system will flag the case for expert review to ensure accurate and reliable diagnosis.

## Overview

The **Medical Imaging Quality Assurance System** is an AI-powered healthcare application designed to automatically screen **Chest X-ray images for Pneumonia**. The system uses deep learning and computer vision techniques to analyze X-ray images and assist medical professionals in early diagnosis.

The main goal of this project is to build a **reliable AI-assisted diagnostic pipeline** where predictions are made automatically when the model confidence is high, while **low-confidence predictions are routed to a radiologist for manual review**. This approach introduces a **human-in-the-loop system** that improves safety, reliability, and trust in AI-based medical applications.

This project integrates **deep learning, medical image preprocessing, automated decision routing, and API deployment** into a complete pipeline suitable for real-world healthcare environments.

---

# Project Objectives

* Build a **deep learning model** capable of detecting pneumonia from chest X-ray images.
* Implement a **confidence-based routing system** to decide whether a prediction should be automated or reviewed by a radiologist.
* Create an **API service using FastAPI** to serve model predictions.
* Package the system inside a **Docker container for easy deployment**.

---

# System Architecture

![System Architecture](images/architecture.jpeg)

The system workflow follows these steps:

1. Input chest X-ray image is uploaded.
2. Image undergoes preprocessing using CLAHE-based enhancement and normalization.
3. Processed image is passed to the deep learning classification model.
4. The model outputs a prediction probability.
5. If the confidence score is above a predefined threshold:
   - Prediction is returned automatically.
6. If confidence is below the threshold
    * Image is moved to a **pending_review folder** for radiologist evaluation.

---

# Key Features

### 1. Pneumonia Classification Model

A deep learning model is trained to classify chest X-ray images into:

* Pneumonia
* Normal

Transfer learning techniques are used with architectures such as:

* ResNet50
* MobileNet

A lightweight ensemble approach is used to balance performance and computational efficiency.

---

### 2. Image Preprocessing (CLAHE-Based Enhancement)

To improve the quality of chest X-ray images, Contrast Limited Adaptive Histogram Equalization (CLAHE) is used. This enhances image contrast and makes important features more visible for the model.

Preprocessing steps include:
* Image resizing
* Normalization
* CLAHE contrast enhancement
* Data augmentation (rotation, flipping, brightness adjustment)

---

### 3. Custom Dataset and DataLoader

A custom dataset pipeline is implemented to efficiently load medical images.

The dataset generator:

* Reads X-ray images
* Returns tensors ready for deep learning models

Example:

```python
from torch.utils.data import Dataset, DataLoader

dataset = Dataset(data=image_files, transform=transforms)
loader = DataLoader(dataset, batch_size=16, shuffle=True)
```

---

### 4. Confidence-Based Human-in-the-Loop System

Healthcare AI must be reliable. This project implements a **confidence threshold mechanism**.

Workflow:

* Model outputs probability score
* If confidence ≥ threshold → automatic prediction
* If confidence < threshold → send image for human review

Example logic:

```python
confidence_threshold = 0.80

if prediction_prob >= confidence_threshold:
    result = "Automated Prediction"
else:
    move_to_pending_review(image_path)
```

Images with uncertain predictions are stored in:

```
pending_review/
```

Radiologists can later examine these cases.

---

### 5. API Service with FastAPI

The trained model is deployed using **FastAPI** to allow easy integration with applications.

Example endpoint:

```
POST /predict
```

Workflow:

1. Upload chest X-ray
2. Preprocess image using CLAHE-based enhancement
3. Run model inference
4. Return prediction or route to review

Example API response:

```json
{
  "prediction": "Pneumonia",
  "confidence": 0.92,
  "status": "automated"
}
```

or

```json
{
  "status": "pending_review",
  "message": "Low confidence prediction routed to radiologist"
}
```

---

### 6. Docker Deployment

The entire application can be containerized using Docker to ensure easy deployment across systems.

Example Docker workflow:

```
Build Docker Image
Run Container
Expose FastAPI endpoint
```

This ensures the system runs consistently in different environments.

---

# Project Structure

```
medical-imaging-quality-assurance-system
│
├── data/
│   ├── images/              # Raw images from NIH dataset
│   ├── Data_Entry_2017.csv  # Metadata and labels
│   ├── processed/           # Output of prepare_dataset.py (Filtered/Split)
│   ├── enhanced/            # Output of image_enhancement.py (Grayscale/Resize)
│   └── clahe_Result/        # Output of clahe.py (Contrast Enhanched)
│
├── preprocessing/
│   ├── prepare_dataset.py   # Step 1: Filter and Split data
│   ├── image_enhancement.py # Step 2: Standardize (Resizing, Grayscale)
│   └── clahe.py             # Step 3: Enhance Contrast
│
├── models/                  # PyTorch model architectures (Placeholders)
│   ├── resnet50_model.py
│   ├── mobilenet_model.py
│   └── ensemble_model.py
│
├── app/                     # FastAPI Application (API)
│   ├── main.py
│   └── api.py
│
├── inference/               # Model inference logic
│   ├── predict.py
│   └── heatmap.py (Grad-CAM)
│
├── tests/                   # Pytest suite
│   └── test_enhancement.py
│
├── requirements.txt         # Project dependencies
├── .gitignore               # Ignored files (data/venv)
└── README.md
```

---

# Technologies Used

* Python
* PyTorch
* Grad-CAM
* OpenCV
* FastAPI
* Docker

---
<!--
# Installation

Clone the repository:

```
git clone https://github.com/yourusername/medical-imaging-quality-assurance-system.git
```

Navigate into the project:

```
cd medical-imaging-quality-assurance-system
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# Running the Training Pipeline

```
python training/train_model.py
```

This will train the pneumonia classification model and save the weights in the **models** folder.

---

# Running the API

Start the FastAPI server:

```
uvicorn api.main:app --reload
```

Open in browser:

```
http://127.0.0.1:8000/docs
```

This provides an interactive API interface.

---

# Future Improvements

* Integration with hospital PACS systems
* Explainable AI for radiology decisions
* Multi-disease detection
* Improved uncertainty estimation
* Cloud deployment for hospital environments

---
-->
## 7. Explainability using Grad-CAM

To improve trust and interpretability, Grad-CAM is used to generate heatmaps that highlight the infected regions in the chest X-ray.

This helps doctors understand why the model made a particular prediction.

# Conclusion

This project demonstrates how AI and deep learning can assist healthcare professionals in medical diagnosis. By integrating custom preprocessing techniques including CLAHE, deep learning classification, confidence-based routing, and human-in-the-loop review, the system provides a safer and more reliable AI-assisted diagnostic workflow for chest X-ray analysis.

Such systems have the potential to **improve diagnostic efficiency, reduce workload for radiologists, and enable faster detection of pneumonia in clinical settings**.

---
<!--
# License

This project is released under the MIT License.
-->
