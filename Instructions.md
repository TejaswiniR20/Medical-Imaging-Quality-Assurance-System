Setup in Spyder (Windows / macOS / Linux)
Step 1 — Create virtual environment
Open Anaconda Prompt or a terminal:
```bash
python -m venv venv
```
Activate it:
```bash
# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```
Step 2 — Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
> TensorFlow + OpenCV take ~5 min on first install.
Step 3 — Add your trained model
Copy your `.keras` model file into the `models/` folder:
```
models/
  best_mobilenetv2_balanced_model.keras   ← from the original project
```
The system searches for models in this order:
`src/best_pneumonia_model.keras`
`models/best_mobilenetv2_balanced_model.keras`
`models/best_pneumonia_model.keras`
If no model is found, the system runs in DEMO mode — predictions are
deterministic hash-based simulations so you can test the full workflow immediately.
Step 4 — Run in Spyder
Option A — Run file directly:
Open `run_server.py` in Spyder, press F5.
Option B — Spyder console:
```python
%run run_server.py
```
Option C — Terminal:
```bash
python run_server.py
```
Option D — Uvicorn directly:
```bash
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```
Step 5 — Open the web app
```
http://localhost:8000
```
---
Using the web application
 Predict tab
Upload a chest X-ray image (JPEG / PNG / BMP / TIFF, max 50 MB)
Click Analyse Image
View: prediction, probabilities, Grad-CAM heatmap, routing decision
If flagged for review, note your Prediction ID
 My Result tab
Enter your Prediction ID to check your doctor's response
Shows: AI result + radiologist diagnosis, severity, notes, recommendation
 History tab
Dashboard stats: total scans, pneumonia, normal, pending, reviewed
Full prediction table with routing status
 Doctor Portal
Grid of pending X-ray images requiring review
Click Review on any image
Fill in: doctor name, diagnosis, severity, notes, recommendation
Submit → patient's result is immediately updated
---