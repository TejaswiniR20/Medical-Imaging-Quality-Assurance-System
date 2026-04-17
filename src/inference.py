"""
inference.py — Core inference pipeline for the Medical Imaging QA webapp.
Handles: CLAHE preprocessing → model prediction → Grad-CAM generation
"""

import os
import sys
import uuid
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

# Allow imports from project root when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.confidence_router import route_prediction
from src.gradcam import GradCAM

# ── Constants ────────────────────────────────────────────────────────────────
IMG_SIZE = (224, 224)
MODEL_PATH = os.getenv("MODEL_PATH", "models/best_pneumonia_model.keras")

# Lazy-loaded global model (loaded once on first call)
_model = None
_gradcam = None


def get_model():
    global _model, _gradcam
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at '{MODEL_PATH}'. "
                "Copy best_pneumonia_model.keras into the models/ folder."
            )
        print(f"[Inference] Loading model from: {MODEL_PATH}")
        _model = tf.keras.models.load_model(MODEL_PATH)
        _gradcam = GradCAM(_model)
        print("[Inference] Model loaded ✓")
    return _model, _gradcam


# ── CLAHE preprocessing ──────────────────────────────────────────────────────

def apply_clahe_to_image(image_path: str, output_path: str) -> str:
    """
    Read image as grayscale, apply CLAHE, save to output_path.
    Returns the output_path.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        # Try via PIL (handles more formats)
        pil_img = Image.open(image_path).convert("L")
        img = np.array(pil_img)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, enhanced)
    return output_path


# ── Model preprocessing ──────────────────────────────────────────────────────

def preprocess_for_model(image_path: str) -> np.ndarray:
    """
    Load image (grayscale CLAHE output), convert to RGB, resize to 224x224,
    normalize to [0,1], add batch dimension.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        pil_img = Image.open(image_path).convert("L")
        img = np.array(pil_img)

    img_resized = cv2.resize(img, IMG_SIZE)
    # Convert grayscale → 3-channel RGB (model expects 3 channels)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
    img_float = img_rgb.astype(np.float32) / 255.0
    return np.expand_dims(img_float, axis=0)   # (1, 224, 224, 3)


# ── Full pipeline ────────────────────────────────────────────────────────────

def run_full_inference(
    original_image_path: str,
    session_id: str = None,
    outputs_dir: str = "outputs"
) -> dict:
    """
    End-to-end inference:
      1. Apply CLAHE to the uploaded image
      2. Preprocess for the model
      3. Run model prediction
      4. Route the confidence
      5. Generate Grad-CAM overlay

    Args:
        original_image_path: Path to the raw uploaded X-ray image.
        session_id: Optional session identifier for naming outputs.
        outputs_dir: Root directory for saving CLAHE + GradCAM results.

    Returns:
        dict with all inference results plus file paths.
    """
    model, gradcam = get_model()
    uid = session_id or str(uuid.uuid4())[:8]
    basename = os.path.splitext(os.path.basename(original_image_path))[0]

    # 1. CLAHE
    clahe_path = os.path.join(outputs_dir, "clahe", f"{uid}_{basename}_clahe.png")
    apply_clahe_to_image(original_image_path, clahe_path)

    # 2. Preprocess
    img_array = preprocess_for_model(clahe_path)

    # 3. Predict
    raw_output = model.predict(img_array, verbose=0)
    pneumonia_prob = float(raw_output[0][0])

    # 4. Route
    routing = route_prediction(pneumonia_prob)

    # 5. Grad-CAM
    heatmap = gradcam.generate(img_array)
    overlay = gradcam.overlay_on_image(clahe_path, heatmap)
    gradcam_path = os.path.join(outputs_dir, "gradcam", f"{uid}_{basename}_gradcam.png")
    os.makedirs(os.path.dirname(gradcam_path), exist_ok=True)
    cv2.imwrite(gradcam_path, overlay)

    return {
        "session_id":       uid,
        "original_image":   original_image_path,
        "clahe_image":      clahe_path,
        "gradcam_image":    gradcam_path,
        "pneumonia_prob":   round(pneumonia_prob * 100, 2),
        "normal_prob":      round((1 - pneumonia_prob) * 100, 2),
        "predicted_class":  routing["predicted_class"],
        "confidence":       round(routing["confidence"] * 100, 2),
        "decision":         routing["decision"],
        "needs_review":     routing["needs_review"],
        # Raw values for report agent
        "pneumonia_prob_raw": pneumonia_prob,
        "normal_prob_raw":    1 - pneumonia_prob,
        "heatmap":            heatmap,
        "label":              routing["predicted_class"],
        "routing":            routing["decision"],
    }
