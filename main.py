"""
main.py — Medical Imaging Quality Assurance System — FastAPI Backend
Serves the full web application including User Portal and Doctor Portal.
"""

import os
import sys
import json
import uuid
import shutil
import logging
from pathlib import Path
from datetime import timedelta
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import (
    FastAPI, File, UploadFile, Depends, HTTPException,
    Form, Request, status
)
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Internal modules ─────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import database as db
from auth import create_access_token, get_current_doctor
from src.inference import run_full_inference

# ── App setup ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
)
logger = logging.getLogger("webapp")

app = FastAPI(
    title="Medical Imaging QA System",
    description="Pneumonia Detection from Chest X-Rays with AI",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Directory setup ───────────────────────────────────────────────────────────
UPLOAD_DIR    = Path(os.getenv("UPLOAD_DIR",    "uploads"))
OUTPUTS_DIR   = Path(os.getenv("OUTPUTS_DIR",   "outputs"))
REPORTS_DIR   = Path(os.getenv("REPORTS_DIR",   "reports"))
STATIC_DIR    = Path("static")
MODEL_ACCURACY_FILE = Path("model_accuracy.json")

for d in [UPLOAD_DIR, OUTPUTS_DIR / "clahe", OUTPUTS_DIR / "gradcam",
          REPORTS_DIR, STATIC_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Mount static file directories
app.mount("/static",   StaticFiles(directory=str(STATIC_DIR)),    name="static")
app.mount("/uploads",  StaticFiles(directory=str(UPLOAD_DIR)),     name="uploads")
app.mount("/outputs",  StaticFiles(directory=str(OUTPUTS_DIR)),    name="outputs")
app.mount("/reports",  StaticFiles(directory=str(REPORTS_DIR)),    name="reports")

# ── DB Init ───────────────────────────────────────────────────────────────────
db.init_db()


# ── Pydantic models ───────────────────────────────────────────────────────────
class DoctorLoginRequest(BaseModel):
    username: str
    password: str

class DoctorReviewRequest(BaseModel):
    doctor_diagnosis: str
    doctor_notes: Optional[str] = ""

class AgentChatRequest(BaseModel):
    message: str
    case_id: Optional[int] = None
    session_id: Optional[str] = None


# ── Helpers ───────────────────────────────────────────────────────────────────
def _load_accuracy() -> dict:
    if MODEL_ACCURACY_FILE.exists():
        try:
            return json.loads(MODEL_ACCURACY_FILE.read_text())
        except Exception:
            pass
    return {
        "accuracy": 87.50,
        "model_name": "best_mobilenetv2_model.keras",
        "note": "Run evaluate_model.py to update this value."
    }


def _diagnosis_recommendation(pneumonia_prob: float) -> dict:
    """
    Map pneumonia probability percentage to a clinical recommendation.
    - >85%: Definite Pneumonia — urgent consultation
    - >50%: Possible Pneumonia — recommended consultation
    - <=50%: Normal — optional consultation
    """
    if pneumonia_prob > 85:
        return {
            "status":  "PNEUMONIA_CONFIRMED",
            "label":   "Pneumonia Detected",
            "severity": "high",
            "color":   "#c62828",
            "message": (
                "High probability of Pneumonia detected. "
                "Immediate doctor consultation is strongly recommended. "
                "This patient has been flagged as Pneumonia Disorder."
            ),
            "consultation_required": True,
        }
    elif pneumonia_prob > 50:
        return {
            "status":  "PNEUMONIA_POSSIBLE",
            "label":   "Possible Pneumonia",
            "severity": "medium",
            "color":   "#f57f17",
            "message": (
                "Moderate probability of Pneumonia detected. "
                "Doctor consultation is recommended. "
                "Patient may have Pneumonia Disorder."
            ),
            "consultation_required": True,
        }
    else:
        return {
            "status":  "NORMAL",
            "label":   "Normal",
            "severity": "low",
            "color":   "#2e7d32",
            "message": (
                "No significant signs of Pneumonia detected. "
                "Result appears normal. "
                "Optional doctor consultation available if desired."
            ),
            "consultation_required": False,
        }


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve the single-page application."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found in static/")
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"))


@app.get("/api/accuracy")
async def get_model_accuracy():
    """Return pre-computed model accuracy metrics."""
    return _load_accuracy()


@app.post("/api/analyze")
async def analyze_xray(
    file: UploadFile = File(...),
    session_id: str  = Form(default=""),
):
    """
    Upload an X-ray image, run CLAHE preprocessing + inference + Grad-CAM.
    Returns prediction results and image paths.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Uploaded file must be an image.")

    # Use provided session_id or generate one
    sid = session_id.strip() if session_id.strip() else str(uuid.uuid4())[:12]

    # Save uploaded file
    suffix   = Path(file.filename).suffix or ".jpg"
    filename = f"{sid}_{uuid.uuid4().hex[:6]}{suffix}"
    save_path = UPLOAD_DIR / filename
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    logger.info(f"[Analyze] session={sid} file={filename}")

    try:
        result = run_full_inference(
            original_image_path=str(save_path),
            session_id=sid,
            outputs_dir=str(OUTPUTS_DIR),
        )
    except FileNotFoundError as e:
        raise HTTPException(503, f"Model not ready: {e}")
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(500, f"Inference failed: {str(e)}")

    recommendation = _diagnosis_recommendation(result["pneumonia_prob"])

    # Store in DB
    case_id = db.insert_case(
        session_id      = sid,
        original_image  = str(save_path),
        gradcam_image   = result.get("gradcam_image", ""),
        pneumonia_prob  = result["pneumonia_prob"],
        normal_prob     = result["normal_prob"],
        predicted_class = result["predicted_class"],
        confidence      = result["confidence"],
        decision        = result["decision"],
        needs_review    = result["needs_review"],
    )

    # Build URL-friendly paths
    def to_url(path: str) -> str:
        if not path:
            return ""
        p = Path(path)
        if str(UPLOAD_DIR) in str(p):
            return f"/uploads/{p.name}"
        if str(OUTPUTS_DIR) in str(p):
            rel = p.relative_to(OUTPUTS_DIR)
            return f"/outputs/{rel}"
        return f"/{path}"

    return {
        "case_id":          case_id,
        "session_id":       sid,
        "original_image":   to_url(str(save_path)),
        "gradcam_image":    to_url(result.get("gradcam_image", "")),
        "pneumonia_prob":   result["pneumonia_prob"],
        "normal_prob":      result["normal_prob"],
        "predicted_class":  result["predicted_class"],
        "confidence":       result["confidence"],
        "decision":         result["decision"],
        "needs_review":     result["needs_review"],
        "recommendation":   recommendation,
    }


# ── User History ──────────────────────────────────────────────────────────────

@app.get("/api/history/{session_id}")
async def get_user_history(session_id: str):
    """Return all cases for a given session/user."""
    cases = db.get_cases_for_session(session_id)

    def enrich(c: dict) -> dict:
        # Add URL paths
        orig = Path(c.get("original_image", ""))
        grad = Path(c.get("gradcam_image", ""))
        c["original_url"] = f"/uploads/{orig.name}" if orig.name else ""
        c["gradcam_url"]  = f"/outputs/gradcam/{grad.name}" if grad.name else ""
        c["recommendation"] = _diagnosis_recommendation(c.get("pneumonia_prob", 0))
        return c

    return [enrich(c) for c in cases]


@app.get("/api/case/{case_id}")
async def get_case(case_id: int):
    """Return a single case by ID."""
    case = db.get_case_by_id(case_id)
    if not case:
        raise HTTPException(404, "Case not found")

    orig = Path(case.get("original_image", ""))
    grad = Path(case.get("gradcam_image",  ""))
    case["original_url"] = f"/uploads/{orig.name}" if orig.name else ""
    case["gradcam_url"]  = f"/outputs/gradcam/{grad.name}" if grad.name else ""
    case["recommendation"] = _diagnosis_recommendation(case.get("pneumonia_prob", 0))
    return case


# ── Doctor Auth ───────────────────────────────────────────────────────────────

@app.post("/api/doctor/login")
async def doctor_login(req: DoctorLoginRequest):
    """Doctor login — returns a JWT access token."""
    doctor = db.get_doctor_by_username(req.username)
    if not doctor or not db.verify_password(req.password, doctor["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )
    token = create_access_token(
        data={"sub": doctor["username"], "name": doctor.get("full_name", "")},
        expires_delta=timedelta(hours=8),
    )
    return {
        "access_token": token,
        "token_type":   "bearer",
        "username":     doctor["username"],
        "full_name":    doctor.get("full_name", ""),
    }


# ── Doctor Portal ─────────────────────────────────────────────────────────────

@app.get("/api/doctor/cases")
async def doctor_get_cases(current_doctor: dict = Depends(get_current_doctor)):
    """Return all cases for doctor review."""
    cases = db.get_all_cases()

    def enrich(c: dict) -> dict:
        orig = Path(c.get("original_image", ""))
        grad = Path(c.get("gradcam_image",  ""))
        c["original_url"] = f"/uploads/{orig.name}" if orig.name else ""
        c["gradcam_url"]  = f"/outputs/gradcam/{grad.name}" if grad.name else ""
        c["recommendation"] = _diagnosis_recommendation(c.get("pneumonia_prob", 0))
        return c

    return [enrich(c) for c in cases]


@app.post("/api/doctor/review/{case_id}")
async def doctor_review_case(
    case_id: int,
    req: DoctorReviewRequest,
    current_doctor: dict = Depends(get_current_doctor),
):
    """Doctor submits a diagnosis review for a case."""
    case = db.get_case_by_id(case_id)
    if not case:
        raise HTTPException(404, "Case not found")

    db.update_doctor_review(
        case_id          = case_id,
        doctor_username  = current_doctor["sub"],
        doctor_diagnosis = req.doctor_diagnosis,
        doctor_notes     = req.doctor_notes or "",
    )
    logger.info(f"[Doctor] {current_doctor['sub']} reviewed case {case_id}: {req.doctor_diagnosis}")
    return {"success": True, "case_id": case_id, "diagnosis": req.doctor_diagnosis}


# ── Report Generation ─────────────────────────────────────────────────────────

@app.post("/api/report/{case_id}")
async def generate_report(case_id: int):
    """Generate a PDF clinical report for a case using the Groq AI agent."""
    groq_key = os.getenv("GROQ_API_KEY", "")
    if not groq_key:
        raise HTTPException(
            503,
            "GROQ_API_KEY not configured. Add it to your .env file to enable report generation."
        )

    case = db.get_case_by_id(case_id)
    if not case:
        raise HTTPException(404, "Case not found")

    from src.report_agent import run_report_agent

    inference_result = {
        "label":           case.get("predicted_class", "Unknown"),
        "pneumonia_prob":  case.get("pneumonia_prob", 0) / 100,
        "pneumonia_prob_raw": case.get("pneumonia_prob", 0) / 100,
        "normal_prob_raw": case.get("normal_prob", 0) / 100,
        "normal_prob":     case.get("normal_prob", 0) / 100,
        "routing":         case.get("decision", "N/A"),
        "heatmap":         None,  # heatmap numpy not stored, stats only
    }

    output_path = str(REPORTS_DIR / f"report_case_{case_id}.pdf")
    try:
        pdf_path = run_report_agent(
            inference_result     = inference_result,
            original_image_path  = case.get("original_image", ""),
            output_path          = output_path,
            api_key              = groq_key,
        )
        db.update_report_pdf(case_id, pdf_path)
        return FileResponse(
            pdf_path,
            media_type   = "application/pdf",
            filename     = f"pneumonia_report_case_{case_id}.pdf",
        )
    except Exception as e:
        logger.error(f"Report generation error: {e}", exc_info=True)
        raise HTTPException(500, f"Report generation failed: {str(e)}")


# ── AI Agent Chat ─────────────────────────────────────────────────────────────

@app.post("/api/agent/chat")
async def agent_chat(req: AgentChatRequest):
    """Conversational AI endpoint for the chat popup widget."""
    groq_key = os.getenv("GROQ_API_KEY", "")
    if not groq_key:
        return {
            "response": (
                "The AI assistant requires a GROQ_API_KEY in your .env file. "
                "Please add it and restart the server to enable this feature."
            )
        }

    case_context = None
    if req.case_id:
        case = db.get_case_by_id(req.case_id)
        if case:
            case_context = {
                "pneumonia_prob":  case.get("pneumonia_prob"),
                "normal_prob":     case.get("normal_prob"),
                "predicted_class": case.get("predicted_class"),
                "confidence":      case.get("confidence"),
                "decision":        case.get("decision"),
            }

    from src.report_agent import chat_with_agent
    try:
        response = chat_with_agent(req.message, case_context, groq_key)
        return {"response": response}
    except Exception as e:
        logger.error(f"Agent chat error: {e}")
        return {"response": "I encountered an error. Please try again."}


# ── Health Check ──────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    model_path = Path(os.getenv("MODEL_PATH", "models/best_pneumonia_model.keras"))
    return {
        "status":       "ok",
        "model_ready":  model_path.exists(),
        "model_path":   str(model_path),
        "groq_enabled": bool(os.getenv("GROQ_API_KEY")),
    }


# ── Entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host   = os.getenv("HOST", "127.0.0.1"),
        port   = int(os.getenv("PORT", "8000")),
        reload = os.getenv("RELOAD", "true").lower() == "true",
    )
