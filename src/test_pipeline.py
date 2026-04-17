import sys, os, numpy as np, tensorflow as tf
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from confidence_router import route_prediction
from gradcam import GradCAM
from report_agent import run_report_agent
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image

IMAGE_PATH = r"C:\Users\Mohammed_Anas\OneDrive\Desktop\pneumonia-xray.jpg"
MODEL_PATH = r"best_pneumonia_model.keras"
GROQ_KEY = os.environ.get("GROQ_API_KEY", "gsk_your_actual_key_here")

# Step 1 — Load model
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded ✓")

# Step 2 — Preprocess image
img = Image.open(IMAGE_PATH).convert("RGB").resize((224, 224))
img_array = np.expand_dims(np.array(img), axis=0)
img_array = preprocess_input(img_array.astype(np.float32))

# Step 3 — Predict
raw = float(model.predict(img_array, verbose=0)[0][0])
pneumonia_prob = raw
normal_prob    = 1 - raw
print(f"Pneumonia prob: {pneumonia_prob:.1%}, Normal prob: {normal_prob:.1%}")

# Step 4 — Confidence routing
routing_result = route_prediction(pneumonia_prob)
routing_label  = routing_result["decision"]  # extract just the string

# Step 5 — Grad-CAM (numpy array)
gradcam    = GradCAM(model)
heatmap    = gradcam.generate(img_array)
print(f"Heatmap shape: {heatmap.shape}, max: {heatmap.max():.3f}")

# Step 6 — Report
label = "Pneumonia" if pneumonia_prob >= 0.5 else "Normal"
pdf = run_report_agent(
    inference_result={
        "routing": routing_label,
        "pneumonia_prob": pneumonia_prob,
        "normal_prob":    normal_prob,
        "routing":        routing_result,
        "heatmap":        heatmap,
    },
    original_image_path=IMAGE_PATH,
    output_path="reports/real_test.pdf",
    api_key=GROQ_KEY,
)
print(f"\nDone! PDF → {pdf}")