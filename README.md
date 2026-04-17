# 🫁 Medical Imaging Quality Assurance System
### AI-Powered Pneumonia Detection from Chest X-Rays

A full-stack web application built with **FastAPI** (backend) and **Vanilla JS** (frontend) that detects pneumonia in chest X-rays using deep learning, Grad-CAM visualization, and an agentic AI assistant.

---

## 📋 Table of Contents
1. [Features](#features)
2. [Project Structure](#project-structure)
3. [Prerequisites](#prerequisites)
4. [Setup Instructions](#setup-instructions)
5. [Running the Application](#running-the-application)
6. [Usage Guide](#usage-guide)
7. [Doctor Portal](#doctor-portal)
8. [AI Agent](#ai-agent)
9. [Configuration](#configuration)
10. [Troubleshooting](#troubleshooting)
11. [API Reference](#api-reference)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔬 **CLAHE Preprocessing** | Automatic contrast enhancement of uploaded X-rays |
| 🧠 **Deep Learning Inference** | MobileNetV2-based model for pneumonia detection |
| 🔥 **Grad-CAM Heatmap** | Visual explanation highlighting suspicious regions |
| 📊 **Confidence Routing** | Three-tier diagnosis: Confirmed / Possible / Normal |
| 🩺 **Doctor Portal** | JWT-authenticated review interface for physicians |
| 💬 **AI Agent Popup** | Groq-powered chat assistant (LLaMA 3.3 70B) |
| 📄 **PDF Reports** | Auto-generated clinical reports via ReportLab + Groq |
| 🗃️ **Case History** | SQLite-backed patient case tracking |
| 🐳 **Docker-Ready** | Single-server architecture, easy to containerize |

---

## 📁 Project Structure

```
pneumonia_webapp/
│
├── main.py                     # FastAPI application — all routes
├── database.py                 # SQLite helpers & schema
├── auth.py                     # JWT doctor authentication
├── model_accuracy.json         # Pre-computed accuracy (update via evaluate_model.py)
├── pneumonia_qa.db             # SQLite database (auto-created on first run)
│
├── src/
│   ├── __init__.py
│   ├── inference.py            # End-to-end inference pipeline
│   ├── gradcam.py              # Grad-CAM heatmap generation
│   ├── confidence_router.py    # Prediction routing logic
│   └── report_agent.py        # Groq AI agent + PDF report builder
│
├── preprocessing/
│   ├── __init__.py
│   └── clahe.py               # CLAHE image enhancement
│
├── static/
│   └── index.html             # Complete single-page application (UI)
│
├── models/                    # ← PUT YOUR .keras FILES HERE
│   └── README.txt
│
├── uploads/                   # User-uploaded X-ray images (auto-created)
├── outputs/
│   ├── clahe/                 # CLAHE-enhanced images
│   └── gradcam/               # Grad-CAM overlay images
├── reports/                   # Generated PDF reports
│
├── requirements.txt           # All Python dependencies
├── .env.example               # Environment variable template
└── README.md                  # This file
```

---

## 🔧 Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | **3.10.x** | Strictly required for TF 2.13 compatibility |
| pip | ≥ 23 | `python -m pip install --upgrade pip` |
| Git | Any | Optional |
| RAM | ≥ 4 GB | 8 GB recommended for TensorFlow |
| Groq API Key | Free | Get at https://console.groq.com/ |

> ⚠️ **Python 3.10 is required.** TensorFlow 2.13 does NOT support Python 3.11+.  
> Check your version: `python --version`

---

## 🚀 Setup Instructions

### Step 1 — Clone / Extract the Project

If you received a ZIP file, extract it. You should have a folder called `pneumonia_webapp/`.

```bash
cd pneumonia_webapp
```

### Step 2 — Create a Virtual Environment

```bash
# Create the virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# Activate it (macOS / Linux)
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

### Step 3 — Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> ⏳ This may take 3–5 minutes. TensorFlow is a large package (~600 MB).

If you encounter errors with `opencv-python`, try:
```bash
pip install opencv-python-headless==4.9.0.80
```

### Step 4 — Add Your Keras Model Files

Copy your trained model files into the `models/` folder:

```
models/
├── best_pneumonia_model.keras       ← REQUIRED (main inference model)
└── best_mobilenetv2_model.keras     ← Optional (for evaluate_model.py)
```

> 🔑 The app **will not run inference** without `best_pneumonia_model.keras`.  
> The server starts fine but returns a 503 error when you try to analyze an image.

### Step 5 — Configure Environment Variables

```bash
# Copy the template
cp .env.example .env
```

Open `.env` in any text editor and fill in:

```env
# REQUIRED — Get your free key at https://console.groq.com/
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx

# Optional — keep defaults for local development
HOST=0.0.0.0
PORT=8000
MODEL_PATH=models/best_pneumonia_model.keras
```

> 💡 The app works **without** a Groq key — the AI agent popup will show a message,  
> and PDF report generation will return an error, but all other features work normally.

### Step 6 — (Optional) Update Model Accuracy

The accuracy displayed in the top bar comes from `model_accuracy.json`.  
To update it with your actual test set accuracy, run:

```bash
# Edit evaluate_model.py to point to your test data, then:
python evaluate_model.py
```

Then manually update `model_accuracy.json`:
```json
{
  "accuracy": 91.25,
  "model_name": "best_mobilenetv2_model.keras"
}
```

---

## ▶️ Running the Application

### Method 1 — Direct Python (Recommended for Development)

```bash
# Make sure your virtual environment is activated
python main.py
```

### Method 2 — Uvicorn CLI

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Method 3 — VS Code

1. Open the `pneumonia_webapp/` folder in VS Code
2. Select your Python interpreter: `Ctrl+Shift+P` → `Python: Select Interpreter` → choose `venv`
3. Open the integrated terminal (`Ctrl+\``)
4. Run: `python main.py`

You should see:

```
[DB] Default doctor created → username: doctor1 / password: doctor123
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

Open your browser and navigate to: **http://localhost:8000**

---

## 🧭 Usage Guide

### Patient Portal

1. **Open the app** at http://localhost:8000
2. Your **Patient ID** is auto-generated and saved in your browser
3. **Upload a chest X-ray** by clicking the upload area or dragging an image
4. Click **🔬 Analyze X-Ray**
5. Wait for analysis (typically 3–10 seconds depending on hardware)
6. View results:
   - **Grad-CAM heatmap** showing suspicious regions
   - **Probability scores** for Pneumonia vs Normal
   - **Diagnosis recommendation** based on thresholds:

| Pneumonia Probability | Diagnosis | Recommendation |
|---|---|---|
| **> 85%** | 🔴 Pneumonia Detected | Immediate doctor consultation required |
| **> 50%** | 🟡 Possible Pneumonia | Doctor consultation recommended |
| **≤ 50%** | 🟢 Normal | No immediate action needed |

7. Click **📄 Generate PDF Report** to download a full clinical report (requires Groq API key)
8. View **Case History** to see all previous analyses

---

## 🩺 Doctor Portal

### Default Login Credentials

| Field | Value |
|---|---|
| Username | `doctor1` |
| Password | `doctor123` |

### How to Use

1. Click the **Doctor Portal** tab in the navigation
2. Click **🔐 Doctor Login** and enter credentials
3. The dashboard shows:
   - Total cases / Pending reviews / Completed reviews
   - Filterable table of all patient cases
4. Click **🔍 Review** on any case to open the review panel:
   - See the **patient-uploaded X-ray** and **Grad-CAM overlay** side-by-side
   - View AI analysis metrics
   - Select a **diagnosis**:
     - 🔴 Pneumonia Confirmed
     - 🟡 Possible Pneumonia
     - 🟢 Normal - No Pneumonia
     - 🔵 Requires Further Tests
     - ⚪ Inconclusive
   - Add **clinical notes**
   - Click **✅ Submit Review**
5. The review is saved and visible to the patient in their case history

### Adding More Doctor Accounts

You can add doctors directly to the SQLite database:

```python
# Run this in a Python shell inside your venv
import database as db
db.init_db()
with db.get_db() as conn:
    conn.execute(
        "INSERT INTO doctors (username, password_hash, full_name) VALUES (?,?,?)",
        ("doctor2", db.hash_password("securepassword"), "Dr. John Smith")
    )
    conn.commit()
print("Doctor added!")
```

---

## 🤖 AI Agent

The AI chat assistant (bottom-right 🤖 button) is powered by **Groq's LLaMA 3.3 70B**.

**Features:**
- Explains current analysis results in plain language
- Answers general questions about pneumonia and chest X-rays
- Provides context-aware responses based on the active case
- Reminds users that results should be confirmed by a doctor

**Requires:** `GROQ_API_KEY` in your `.env` file

**Without Groq key:** The popup still appears but shows a configuration message.

---

## ⚙️ Configuration

All configuration is in `.env`:

```env
GROQ_API_KEY=            # Groq API key (required for AI features)
HOST=0.0.0.0             # Server bind address
PORT=8000                # Server port
RELOAD=true              # Auto-reload on code changes (disable in production)
MODEL_PATH=models/best_pneumonia_model.keras
UPLOAD_DIR=uploads
OUTPUTS_DIR=outputs
REPORTS_DIR=reports
SECRET_KEY=...           # JWT signing key — change in production!
TOKEN_EXPIRE_MINUTES=480 # Doctor session duration (8 hours)
DB_PATH=pneumonia_qa.db  # SQLite database file path
```

---

## 🐛 Troubleshooting

### ❌ `ModuleNotFoundError: No module named 'tensorflow'`
```bash
# Make sure venv is activated, then:
pip install tensorflow==2.13.0
```

### ❌ `FileNotFoundError: Model not found at 'models/best_pneumonia_model.keras'`
Copy your `.keras` model file into the `models/` folder. The filename must match exactly.

### ❌ `503 Service Unavailable — Model not ready`
The model file is missing or the path in `.env` is wrong. Check `MODEL_PATH`.

### ❌ `ImportError: libGL.so.1: cannot open shared object file` (Linux)
```bash
# Install OpenGL libraries
sudo apt-get install libgl1-mesa-glx libglib2.0-0
# OR use headless OpenCV:
pip uninstall opencv-python && pip install opencv-python-headless==4.9.0.80
```

### ❌ `ValueError: Groq API key not found`
Add `GROQ_API_KEY=your_key` to your `.env` file. Get a free key at https://console.groq.com/

### ❌ Port already in use
```bash
# Change the port in .env:
PORT=8001
# Or kill the process using port 8000:
# Windows:
netstat -ano | findstr :8000
taskkill /PID <pid> /F
# Linux/Mac:
lsof -ti:8000 | xargs kill
```

### ❌ `AttributeError` on GradCAM layer `out_relu`
The GradCAM code automatically falls back to the last conv layer if `out_relu` is not found. If you still get errors, your model architecture may differ. Check your model's layer names:
```python
import tensorflow as tf
model = tf.keras.models.load_model("models/best_pneumonia_model.keras")
for layer in model.layers[-10:]:
    print(layer.name, type(layer).__name__)
```

### ❌ Doctor login returns 401
The default credentials are `doctor1` / `doctor123`.  
If the database was deleted and recreated, run the app once — the default doctor is seeded automatically.

### ❌ Images not displaying after analysis
Make sure the `uploads/`, `outputs/clahe/`, and `outputs/gradcam/` directories exist. They are created automatically on startup, but check that the app has write permissions.

---

## 📡 API Reference

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `GET` | `/` | None | Serve index.html |
| `GET` | `/api/accuracy` | None | Model accuracy metrics |
| `POST` | `/api/analyze` | None | Upload & analyze X-ray |
| `GET` | `/api/history/{session_id}` | None | Patient case history |
| `GET` | `/api/case/{case_id}` | None | Single case details |
| `POST` | `/api/report/{case_id}` | None | Generate PDF report |
| `POST` | `/api/agent/chat` | None | AI chat agent |
| `POST` | `/api/doctor/login` | None | Doctor authentication |
| `GET` | `/api/doctor/cases` | Bearer JWT | All patient cases |
| `POST` | `/api/doctor/review/{case_id}` | Bearer JWT | Submit diagnosis |
| `GET` | `/api/health` | None | Server health check |

### Interactive API Docs
FastAPI provides automatic documentation at:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:**       http://localhost:8000/redoc

---

## 🐳 Docker (Optional)

Create a `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t pneumonia-qa .
docker run -p 8000:8000 --env-file .env -v $(pwd)/models:/app/models pneumonia-qa
```

---

## 🔒 Security Notes for Production

1. **Change `SECRET_KEY`** in `.env` to a long random string
2. **Disable `RELOAD=false`** in production
3. **Use HTTPS** — run behind nginx or a reverse proxy
4. **Restrict CORS** — update `allow_origins` in `main.py`
5. **Protect model files** — do not expose the `models/` directory directly

---

## 📝 Notes

- **Session IDs** are auto-generated and stored in the browser's `localStorage`. Clearing browser storage will create a new session.
- **The SQLite database** (`pneumonia_qa.db`) is created automatically on first run.
- **Uploaded images** are stored in `uploads/`. Implement periodic cleanup for production use.
- This system is intended as a **clinical decision support tool**, not a replacement for radiologist review.

---

*Built with ❤️ using FastAPI · TensorFlow · Grad-CAM · Groq · ReportLab*
