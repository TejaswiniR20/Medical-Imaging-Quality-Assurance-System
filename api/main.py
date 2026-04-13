"""
api/main.py
===========
Medical Imaging Quality Assurance System — FastAPI Web Application

Features implemented:
  ✓  Image upload → CLAHE + MobileNetV2 preprocessing → inference
  ✓  Confidence routing (high ≥ 0.85 → automated, middle → doctor review)
  ✓  Grad-CAM heatmap returned as base64 PNG
  ✓  SQLite database for all predictions, doctor queue, reports
  ✓  Doctor portal: view pending images, submit diagnosis, response sent back to user
  ✓  Single-page frontend served from static/index.html
  ✓  All endpoints compatible with Spyder (run directly or via uvicorn)

Run from project root:
    python -m uvicorn api.main:app --reload --port 8000

Or from Spyder:
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
"""

import os
import sys
import uuid
import shutil
import logging
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# ── Path setup (works from any CWD) ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR      = PROJECT_ROOT / "src"
STATIC_DIR   = PROJECT_ROOT / "static"
PENDING_DIR  = PROJECT_ROOT / "pending_review"
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"

for _d in [PENDING_DIR, OUTPUTS_DIR, STATIC_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# Add src to sys.path so inference.py can import confidence_router / gradcam
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ── Internal imports ──────────────────────────────────────────────────────────
from src.inference import run_inference, load_model_once

sys.path.insert(0, str(PROJECT_ROOT / "database"))
from database.db import (
    init_db,
    insert_prediction,
    insert_doctor_queue,
    insert_doctor_report,
    get_all_predictions,
    get_pending_queue,
    get_all_queue,
    get_all_reports,
    get_report_for_queue,
    get_stats,
    get_prediction,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("miqas.api")

# ── Startup ───────────────────────────────────────────────────────────────────
init_db()
load_model_once()

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Medical Imaging Quality Assurance System",
    description="Pneumonia detection · MobileNetV2 · Grad-CAM · Doctor review loop",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve pending_review images as /uploads/<filename>
app.mount("/uploads", StaticFiles(directory=str(PENDING_DIR)), name="uploads")


# ═══════════════════════════════════════════════════════════════════════════════
# FRONTEND
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = STATIC_DIR / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not built yet.")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


# ═══════════════════════════════════════════════════════════════════════════════
# HEALTH
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    stats = get_stats()
    from src.inference import _demo_mode
    return {
        "status":     "ok",
        "demo_mode":  _demo_mode,
        "db":         "connected",
        "timestamp":  datetime.utcnow().isoformat(),
        **stats,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICT
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    session_id: str = Form(default=""),
):
    """
    Accept a chest X-ray image and return:
      - Pneumonia probability
      - Confidence routing decision
      - Grad-CAM heatmap (base64 PNG)
      - DB record id

    If decision == 'Review', image is saved and queued for doctor.
    """
    if not session_id:
        session_id = str(uuid.uuid4())

    # ── Validate ──────────────────────────────────────────────────────────────
    fname = file.filename or "upload.jpg"
    ext   = Path(fname).suffix.lower()
    if ext not in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}:
        raise HTTPException(400, "Invalid file type. Accepted: JPEG, PNG, BMP, TIFF.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(400, "Empty file.")
    if len(image_bytes) > 50 * 1024 * 1024:
        raise HTTPException(400, "File too large (max 50 MB).")

    # ── Inference ─────────────────────────────────────────────────────────────
    try:
        result = run_inference(image_bytes, generate_gradcam=True)
    except Exception as exc:
        log.error(f"Inference error: {exc}")
        raise HTTPException(500, f"Inference failed: {exc}")

    # ── Save image if review needed ───────────────────────────────────────────
    safe_fname    = f"{uuid.uuid4().hex}_{fname}"
    heatmap_fname = None
    saved_path    = None

    if result["needs_review"]:
        saved_path = PENDING_DIR / safe_fname
        with open(saved_path, "wb") as f_out:
            f_out.write(image_bytes)

        # Save heatmap to outputs/
        if result.get("gradcam_b64"):
            import base64, cv2, numpy as np
            heatmap_fname = f"heatmap_{safe_fname}.png"
            heatmap_bytes = base64.b64decode(result["gradcam_b64"])
            with open(OUTPUTS_DIR / heatmap_fname, "wb") as hf:
                hf.write(heatmap_bytes)

        log.info(f"[REVIEW]    {fname} → conf={result['confidence']}%  saved={safe_fname}")
    else:
        log.info(f"[AUTOMATED] {fname} → {result['predicted_class']} conf={result['confidence']}%")

    # ── Persist to DB ─────────────────────────────────────────────────────────
    pred_id = insert_prediction(
        session_id=session_id,
        original_filename=fname,
        saved_filename=safe_fname,
        pneumonia_prob=result["pneumonia_prob"] / 100,
        normal_prob=result["normal_prob"] / 100,
        predicted_class=result["predicted_class"],
        confidence=result["confidence"] / 100,
        decision=result["decision"],
        needs_review=result["needs_review"],
        heatmap_filename=heatmap_fname,
    )

    queue_id = None
    if result["needs_review"]:
        queue_id = insert_doctor_queue(
            prediction_id=pred_id,
            session_id=session_id,
            saved_filename=safe_fname,
            original_filename=fname,
            pneumonia_prob=result["pneumonia_prob"] / 100,
        )

    return JSONResponse({
        "ok":              True,
        "pred_id":         pred_id,
        "queue_id":        queue_id,
        "session_id":      session_id,
        "filename":        fname,
        "predicted_class": result["predicted_class"],
        "pneumonia_prob":  result["pneumonia_prob"],
        "normal_prob":     result["normal_prob"],
        "confidence":      result["confidence"],
        "decision":        result["decision"],
        "needs_review":    result["needs_review"],
        "gradcam_b64":     result.get("gradcam_b64", ""),
        "demo_mode":       result["demo_mode"],
        "message": (
            "Image forwarded to radiologist for review."
            if result["needs_review"] else
            f"High-confidence result: {result['predicted_class']}."
        ),
    })


# ═══════════════════════════════════════════════════════════════════════════════
# HISTORY / RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/results")
async def get_results(session_id: str = ""):
    """Return prediction history (newest first). Filter by session_id if given."""
    rows = get_all_predictions(session_id=session_id or None)
    return JSONResponse({"results": rows, "count": len(rows)})


@app.get("/stats")
async def get_dashboard_stats():
    return JSONResponse(get_stats())


# ═══════════════════════════════════════════════════════════════════════════════
# DOCTOR QUEUE
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/doctor/queue")
async def doctor_queue():
    """List all pending items in the doctor review queue."""
    items = get_pending_queue()
    return JSONResponse({"queue": items, "count": len(items)})


@app.get("/doctor/queue/all")
async def doctor_queue_all():
    """List all queue items (pending + reviewed)."""
    items = get_all_queue()
    return JSONResponse({"queue": items, "count": len(items)})


@app.get("/doctor/image/{filename}")
async def serve_pending_image(filename: str):
    """Stream a pending-review X-ray image."""
    safe = Path(filename).name
    path = PENDING_DIR / safe
    if not path.exists():
        raise HTTPException(404, "Image not found.")
    ext_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
               ".png": "image/png",  ".bmp":  "image/bmp"}
    mt = ext_map.get(path.suffix.lower(), "image/jpeg")
    return FileResponse(str(path), media_type=mt)


@app.post("/doctor/report/{queue_id}")
async def submit_doctor_report(
    queue_id: int,
    doctor_name:    str = Form(default="Radiologist"),
    diagnosis:      str = Form(...),
    severity:       str = Form(default="Unknown"),
    notes:          str = Form(default=""),
    recommendation: str = Form(default=""),
):
    """
    Doctor submits a review for a queued image.
    Stores report in DB and links it back to the prediction and session.
    """
    # Find the queue row to get prediction_id
    from database.db import get_connection
    conn = get_connection()
    row  = conn.execute(
        "SELECT * FROM doctor_queue WHERE id=?", (queue_id,)
    ).fetchone()
    conn.close()

    if not row:
        raise HTTPException(404, "Queue item not found.")

    row = dict(row)
    if row["status"] == "reviewed":
        raise HTTPException(400, "This item has already been reviewed.")

    report_id = insert_doctor_report(
        queue_id=queue_id,
        prediction_id=row["prediction_id"],
        doctor_name=doctor_name,
        diagnosis=diagnosis,
        severity=severity,
        notes=notes,
        recommendation=recommendation,
    )

    # Optionally remove from pending_review folder
    img_path = PENDING_DIR / row["saved_filename"]
    if img_path.exists():
        shutil.move(str(img_path), str(OUTPUTS_DIR / ("reviewed_" + row["saved_filename"])))

    log.info(f"[REPORT] queue_id={queue_id} → {diagnosis} by {doctor_name}")

    return JSONResponse({
        "ok":        True,
        "report_id": report_id,
        "message":   "Review submitted. Patient will be notified.",
        "diagnosis": diagnosis,
    })


# ═══════════════════════════════════════════════════════════════════════════════
# DOCTOR REPORTS (for patients to check their result)
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/doctor/reports")
async def all_reports():
    """Return all completed doctor reports."""
    reports = get_all_reports()
    return JSONResponse({"reports": reports, "count": len(reports)})


@app.get("/result/{pred_id}")
async def get_single_result(pred_id: int):
    """
    Fetch a single prediction + its doctor report (if reviewed).
    Used by the patient to check their result.
    """
    pred = get_prediction(pred_id)
    if not pred:
        raise HTTPException(404, "Prediction not found.")

    report = None
    if pred.get("doctor_report_id"):
        from database.db import get_connection
        conn = get_connection()
        r = conn.execute(
            "SELECT * FROM doctor_reports WHERE id=?", (pred["doctor_report_id"],)
        ).fetchone()
        conn.close()
        report = dict(r) if r else None

    return JSONResponse({"prediction": pred, "doctor_report": report})
