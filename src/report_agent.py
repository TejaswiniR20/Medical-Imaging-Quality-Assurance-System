"""
Report Generation Agent — Medical Imaging QA System
Uses Groq (free) for text-based clinical report generation + ReportLab for PDF output.


Usage:
    from report_agent import run_report_agent

    pdf_path = run_report_agent(
        inference_result=result,        # dict from inference.py
        original_image_path="img.jpg",  # path to the X-ray image
        output_path="reports/case_001.pdf"
    )
"""

import os
import json
import uuid
import numpy as np
from datetime import datetime
from pathlib import Path

from groq import Groq
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image as RLImage, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

# ── Constants ────────────────────────────────────────────────────────────────
GROQ_MODEL   = "llama-3.3-70b-versatile"
REPORT_DIR   = Path("reports")
PAGE_W, PAGE_H = A4

# Colour palette
NAVY    = colors.HexColor("#1a2e4a")
TEAL    = colors.HexColor("#0d7377")
GREEN   = colors.HexColor("#2e7d32")
RED_C   = colors.HexColor("#c62828")
AMBER   = colors.HexColor("#f57f17")
LIGHT   = colors.HexColor("#f4f6f9")
WHITE   = colors.white
MID     = colors.HexColor("#546e7a")


# ── 1. Heatmap Analysis ───────────────────────────────────────────────────────
def analyse_heatmap(heatmap: np.ndarray) -> dict:
    """
    Extract spatial statistics from a Grad-CAM heatmap numpy array.
    Returns a dict that we convert to plain text for Groq.
    """
    if heatmap is None:
        return {"error": "No heatmap provided"}

    h, w = heatmap.shape[:2]
    # Normalise to 0-1 if needed
    if heatmap.max() > 1.0:
        heatmap = heatmap / 255.0

    threshold = 0.6   # "high activation" threshold

    # Global stats
    mean_act  = float(np.mean(heatmap))
    max_act   = float(np.max(heatmap))
    high_pct  = float(np.mean(heatmap > threshold) * 100)

    # Quadrant breakdown
    q = {
        "top_left":     float(np.mean(heatmap[:h//2,  :w//2])),
        "top_right":    float(np.mean(heatmap[:h//2,  w//2:])),
        "bottom_left":  float(np.mean(heatmap[h//2:,  :w//2])),
        "bottom_right": float(np.mean(heatmap[h//2:,  w//2:])),
    }
    dominant_quadrant = max(q, key=q.get).replace("_", " ")

    # Peak location (row/col as % of image)
    peak_idx  = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    peak_row_pct = round(peak_idx[0] / h * 100, 1)
    peak_col_pct = round(peak_idx[1] / w * 100, 1)

    # Vertical split (upper vs lower lung fields)
    upper_act = float(np.mean(heatmap[:h//2]))
    lower_act = float(np.mean(heatmap[h//2:]))
    vertical_bias = "lower lung fields" if lower_act > upper_act else "upper lung fields"

    return {
        "mean_activation":    round(mean_act, 3),
        "max_activation":     round(max_act, 3),
        "high_activation_pct": round(high_pct, 1),
        "dominant_quadrant":  dominant_quadrant,
        "peak_location":      f"{peak_row_pct}% from top, {peak_col_pct}% from left",
        "vertical_bias":      vertical_bias,
        "quadrant_scores":    {k: round(v, 3) for k, v in q.items()},
    }


def heatmap_to_text(stats: dict) -> str:
    """Convert heatmap stats dict to a human-readable description for Groq."""
    if "error" in stats:
        return "No heatmap data available."

    return (
        f"The Grad-CAM activation heatmap shows the following spatial distribution:\n"
        f"- Mean activation intensity: {stats['mean_activation']} (scale 0–1)\n"
        f"- Peak activation: {stats['max_activation']}\n"
        f"- Area with high activation (>60%): {stats['high_activation_pct']}% of image\n"
        f"- Dominant activation quadrant: {stats['dominant_quadrant']}\n"
        f"- Peak activation location: {stats['peak_location']}\n"
        f"- Primary vertical region: {stats['vertical_bias']}\n"
        f"- Quadrant breakdown (mean): {stats['quadrant_scores']}"
    )


# ── 2. Groq API Call ──────────────────────────────────────────────────────────
def call_groq(inference_result: dict, heatmap_text: str, api_key: str) -> dict:
    """
    Send prediction + heatmap description to Groq.
    Returns a dict with 5 clinical sections.
    """
    client = Groq(api_key=api_key)

    label        = inference_result.get("label", "Unknown")
    pneumo_prob  = inference_result.get("pneumonia_prob", 0.0)
    normal_prob  = inference_result.get("normal_prob", 0.0)
    routing      = inference_result.get("routing", "Unknown")
    confidence   = max(pneumo_prob, normal_prob) * 100

    system_prompt = """You are an expert radiologist AI assistant generating structured clinical reports 
for a chest X-ray pneumonia detection system. You must respond ONLY with a valid JSON object — 
no preamble, no markdown fences, no extra text.

The JSON must have exactly these 5 keys:
{
  "clinical_summary": "...",
  "findings": "...",
  "heatmap_interpretation": "...",
  "routing_recommendation": "...",
  "disclaimer": "..."
}

Each value must be a single string (1–4 sentences). Be precise, clinical, and professional.
Never fabricate specific measurements or patient data not given to you."""

    user_prompt = f"""Generate a clinical report for the following chest X-ray AI analysis:

PREDICTION RESULT:
- Classification: {label}
- Pneumonia Probability: {pneumo_prob:.1%}
- Normal Probability:    {normal_prob:.1%}
- Confidence:            {confidence:.1f}%
- System Routing:        {routing}

GRAD-CAM HEATMAP ANALYSIS:
{heatmap_text}

Write the 5-section clinical report as a JSON object."""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=1024,
    )

    raw = response.choices[0].message.content.strip()

    # Safe JSON parse
    try:
        report = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: try to extract JSON from the string
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        report = json.loads(raw[start:end]) if start != -1 else {}

    return report


# ── 3. PDF Generation ─────────────────────────────────────────────────────────
def build_pdf(
    report_sections: dict,
    inference_result: dict,
    heatmap_stats: dict,
    original_image_path: str,
    output_path: str,
    report_id: str,
) -> str:
    """Build a professional A4 clinical report PDF using ReportLab."""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=18*mm, rightMargin=18*mm,
        topMargin=15*mm,  bottomMargin=15*mm,
    )

    styles = getSampleStyleSheet()
    story  = []

    # ── Custom styles ──
    def style(name, parent="Normal", **kw):
        return ParagraphStyle(name, parent=styles[parent], **kw)

    s_title   = style("s_title",   "Title",   fontSize=20, textColor=WHITE,   alignment=TA_CENTER, spaceAfter=2)
    s_sub     = style("s_sub",     "Normal",  fontSize=9,  textColor=colors.HexColor("#b0bec5"), alignment=TA_CENTER)
    s_h2      = style("s_h2",      "Heading2",fontSize=11, textColor=NAVY,    spaceBefore=6, spaceAfter=3)
    s_body    = style("s_body",    "Normal",  fontSize=9,  textColor=colors.HexColor("#333333"),
                      leading=14, alignment=TA_JUSTIFY, spaceAfter=4)
    s_label   = style("s_label",   "Normal",  fontSize=8,  textColor=MID,     spaceAfter=1)
    s_value   = style("s_value",   "Normal",  fontSize=10, textColor=NAVY,    spaceAfter=6, fontName="Helvetica-Bold")
    s_disc    = style("s_disc",    "Normal",  fontSize=7.5,textColor=MID,     leading=11, alignment=TA_JUSTIFY)

    # ── Header banner ──
    header_data = [[
        Paragraph(f"<b>Medical Imaging QA System</b>", s_title),
        Paragraph(f"Clinical AI Report", s_title),
    ]]
    header_tbl = Table(header_data, colWidths=["50%", "50%"])
    header_tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,-1), NAVY),
        ("ROWPADDING",  (0,0), (-1,-1), 10),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
        ("BOX",         (0,0), (-1,-1), 0, NAVY),
    ]))
    story.append(header_tbl)

    # Sub-header (report ID + timestamp)
    ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    sub_data = [[
        Paragraph(f"Report ID: {report_id}", s_sub),
        Paragraph(f"Generated: {ts}", s_sub),
    ]]
    sub_tbl = Table(sub_data, colWidths=["50%", "50%"])
    sub_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), TEAL),
        ("ROWPADDING", (0,0), (-1,-1), 5),
    ]))
    story.append(sub_tbl)
    story.append(Spacer(1, 8*mm))

    # ── Prediction card ──
    label       = inference_result.get("label", "Unknown")
    pneumo_prob = inference_result.get("pneumonia_prob", 0.0)
    normal_prob = inference_result.get("normal_prob", 0.0)
    routing     = inference_result.get("routing", "N/A")
    confidence  = max(pneumo_prob, normal_prob) * 100

    card_color = GREEN if label == "Normal" else (RED_C if label == "Pneumonia" else AMBER)

    pred_data = [
        [Paragraph("<b>DIAGNOSIS</b>", s_label),
         Paragraph("<b>PNEUMONIA PROB</b>", s_label),
         Paragraph("<b>NORMAL PROB</b>", s_label),
         Paragraph("<b>CONFIDENCE</b>", s_label),
         Paragraph("<b>ROUTING</b>", s_label)],
        [Paragraph(f"<b>{label}</b>", s_value),
         Paragraph(f"{pneumo_prob:.1%}", s_value),
         Paragraph(f"{normal_prob:.1%}", s_value),
         Paragraph(f"{confidence:.1f}%", s_value),
         Paragraph(routing, s_value)],
    ]
    pred_tbl = Table(pred_data, colWidths=["20%","20%","20%","20%","20%"])
    pred_tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,-1), LIGHT),
        ("BACKGROUND",  (0,0), (0,0),   card_color),
        ("TEXTCOLOR",   (0,0), (0,0),   WHITE),
        ("FONTNAME",    (0,0), (0,0),   "Helvetica-Bold"),
        ("ROWPADDING",  (0,0), (-1,-1), 8),
        ("BOX",         (0,0), (-1,-1), 0.5, colors.HexColor("#cfd8dc")),
        ("INNERGRID",   (0,0), (-1,-1), 0.3, colors.HexColor("#cfd8dc")),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(pred_tbl)
    story.append(Spacer(1, 6*mm))

    # ── Image panel (original X-ray + heatmap indicator) ──
    if original_image_path and Path(original_image_path).exists():
        img = RLImage(original_image_path, width=70*mm, height=70*mm)
        img_data = [[img, Paragraph(
            "<b>Grad-CAM Heatmap Summary</b><br/><br/>" +
            f"Mean Activation: {heatmap_stats.get('mean_activation', 'N/A')}<br/>" +
            f"High Activation Area: {heatmap_stats.get('high_activation_pct', 'N/A')}%<br/>" +
            f"Dominant Region: {heatmap_stats.get('dominant_quadrant', 'N/A')}<br/>" +
            f"Vertical Bias: {heatmap_stats.get('vertical_bias', 'N/A')}<br/>" +
            f"Peak Location: {heatmap_stats.get('peak_location', 'N/A')}",
            s_body
        )]]
        img_tbl = Table(img_data, colWidths=["45%", "55%"])
        img_tbl.setStyle(TableStyle([
            ("VALIGN",     (0,0), (-1,-1), "TOP"),
            ("ROWPADDING", (0,0), (-1,-1), 6),
            ("BOX",        (0,0), (-1,-1), 0.5, colors.HexColor("#cfd8dc")),
            ("BACKGROUND", (0,0), (-1,-1), LIGHT),
        ]))
        story.append(img_tbl)
        story.append(Spacer(1, 6*mm))

    # ── 5 Clinical sections ──
    section_map = [
        ("clinical_summary",      "1. Clinical Summary"),
        ("findings",              "2. Findings"),
        ("heatmap_interpretation","3. Heatmap Interpretation"),
        ("routing_recommendation","4. Routing Recommendation"),
    ]

    for key, title in section_map:
        text = report_sections.get(key, "Not available.")
        block = KeepTogether([
            Paragraph(title, s_h2),
            HRFlowable(width="100%", thickness=0.5, color=TEAL, spaceAfter=4),
            Paragraph(text, s_body),
            Spacer(1, 4*mm),
        ])
        story.append(block)

    # ── Disclaimer ──
    story.append(HRFlowable(width="100%", thickness=1, color=NAVY, spaceBefore=4, spaceAfter=4))
    disc_text = report_sections.get(
        "disclaimer",
        "This report is generated by an AI system and is intended to assist, not replace, "
        "qualified medical professionals. All findings must be reviewed and verified by a "
        "licensed radiologist before any clinical decisions are made."
    )
    story.append(Paragraph("<b>Disclaimer</b>", s_h2))
    story.append(Paragraph(disc_text, s_disc))

    doc.build(story)
    return output_path


# ── 4. Main Entry Point ───────────────────────────────────────────────────────
def run_report_agent(
    inference_result: dict,
    original_image_path: str = None,
    output_path: str = None,
    api_key: str = None,
) -> str:
    """
    Full pipeline: heatmap analysis → Groq report → PDF.

    Args:
        inference_result:    Dict from inference.py. Must contain:
                             label, pneumonia_prob, normal_prob, routing, heatmap (np.ndarray)
        original_image_path: Path to the input X-ray image (optional, for PDF display)
        output_path:         Where to save the PDF. Defaults to reports/<report_id>.pdf
        api_key:             Groq API key. Falls back to GROQ_API_KEY env var.

    Returns:
        Path to the generated PDF as a string.
    """
    # Resolve API key
    api_key = api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Groq API key not found. Pass api_key= or set GROQ_API_KEY env var.")

    # Generate report ID
    report_id = f"MIQ-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{str(uuid.uuid4())[:6].upper()}"

    # Default output path
    if output_path is None:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = str(REPORT_DIR / f"{report_id}.pdf")

    print(f"[ReportAgent] Report ID : {report_id}")
    print(f"[ReportAgent] Prediction: {inference_result.get('label')} "
          f"(pneumonia={inference_result.get('pneumonia_prob', 0):.1%})")

    # Step 1 — Analyse heatmap
    heatmap = inference_result.get("heatmap")
    heatmap_stats = analyse_heatmap(heatmap) if heatmap is not None else {}
    heatmap_text  = heatmap_to_text(heatmap_stats)
    print(f"[ReportAgent] Heatmap   : dominant={heatmap_stats.get('dominant_quadrant','N/A')}, "
          f"high_act={heatmap_stats.get('high_activation_pct','N/A')}%")

    # Step 2 — Call Groq
    print("[ReportAgent] Calling Groq API...")
    report_sections = call_groq(inference_result, heatmap_text, api_key)
    print("[ReportAgent] Report sections received ✓")

    # Step 3 — Build PDF
    print(f"[ReportAgent] Building PDF → {output_path}")
    pdf_path = build_pdf(
        report_sections=report_sections,
        inference_result=inference_result,
        heatmap_stats=heatmap_stats,
        original_image_path=original_image_path,
        output_path=output_path,
        report_id=report_id,
    )
    print(f"[ReportAgent] Done ✓  PDF saved to: {pdf_path}")
    return pdf_path


# ── Quick smoke-test (run directly) ──────────────────────────────────────────
if __name__ == "__main__":
    import sys

    # Mock inference result — replace with real output from inference.py
    mock_heatmap = np.random.rand(224, 224).astype(np.float32)
    mock_heatmap[140:200, 100:180] = 0.9   # simulate hot zone in bottom-right

    mock_result = {
        "label":          "Pneumonia",
        "pneumonia_prob": 0.87,
        "normal_prob":    0.13,
        "routing":        "Senior Radiologist Review",
        "heatmap":        mock_heatmap,
    }

    api_key = os.environ.get("GROQ_API_KEY") or (sys.argv[1] if len(sys.argv) > 1 else None)

    pdf = run_report_agent(
        inference_result=mock_result,
        original_image_path=None,          # swap with real image path
        output_path="reports/test_report.pdf",
        api_key=api_key,
    )
    print(f"\nSample report saved → {pdf}")