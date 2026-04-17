"""
report_agent.py — Medical Imaging QA Report Generation Agent
Uses Groq (free) for clinical text generation + ReportLab for PDF output.

Provides:
  - run_report_agent(...)  → generates a full PDF clinical report
  - chat_with_agent(...)   → conversational AI for the chat popup
"""

import os
import json
import uuid
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional

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
GROQ_MODEL = "llama-3.3-70b-versatile"
REPORT_DIR = Path("reports")
PAGE_W, PAGE_H = A4

NAVY  = colors.HexColor("#1a2e4a")
TEAL  = colors.HexColor("#0d7377")
GREEN = colors.HexColor("#2e7d32")
RED_C = colors.HexColor("#c62828")
AMBER = colors.HexColor("#f57f17")
LIGHT = colors.HexColor("#f4f6f9")
WHITE = colors.white
MID   = colors.HexColor("#546e7a")


# ── Heatmap Analysis ─────────────────────────────────────────────────────────

def analyse_heatmap(heatmap: np.ndarray) -> dict:
    if heatmap is None:
        return {"error": "No heatmap provided"}

    h, w = heatmap.shape[:2]
    if heatmap.max() > 1.0:
        heatmap = heatmap / 255.0

    threshold = 0.6
    mean_act  = float(np.mean(heatmap))
    max_act   = float(np.max(heatmap))
    high_pct  = float(np.mean(heatmap > threshold) * 100)

    q = {
        "top_left":     float(np.mean(heatmap[:h//2,  :w//2])),
        "top_right":    float(np.mean(heatmap[:h//2,  w//2:])),
        "bottom_left":  float(np.mean(heatmap[h//2:,  :w//2])),
        "bottom_right": float(np.mean(heatmap[h//2:,  w//2:])),
    }
    dominant_quadrant = max(q, key=q.get).replace("_", " ")

    peak_idx     = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    peak_row_pct = round(peak_idx[0] / h * 100, 1)
    peak_col_pct = round(peak_idx[1] / w * 100, 1)

    upper_act = float(np.mean(heatmap[:h//2]))
    lower_act = float(np.mean(heatmap[h//2:]))
    vertical_bias = "lower lung fields" if lower_act > upper_act else "upper lung fields"

    return {
        "mean_activation":     round(mean_act, 3),
        "max_activation":      round(max_act, 3),
        "high_activation_pct": round(high_pct, 1),
        "dominant_quadrant":   dominant_quadrant,
        "peak_location":       f"{peak_row_pct}% from top, {peak_col_pct}% from left",
        "vertical_bias":       vertical_bias,
        "quadrant_scores":     {k: round(v, 3) for k, v in q.items()},
    }


def heatmap_to_text(stats: dict) -> str:
    if not stats or "error" in stats:
        return "No heatmap data available for this case (heatmap was not retained in the database)."
    return (
        f"The Grad-CAM activation heatmap shows:\n"
        f"- Mean activation intensity: {stats['mean_activation']} (scale 0–1)\n"
        f"- Peak activation: {stats['max_activation']}\n"
        f"- High activation area (>60%): {stats['high_activation_pct']}% of image\n"
        f"- Dominant quadrant: {stats['dominant_quadrant']}\n"
        f"- Peak location: {stats['peak_location']}\n"
        f"- Primary vertical region: {stats['vertical_bias']}\n"
        f"- Quadrant breakdown: {stats['quadrant_scores']}"
    )


# ── Groq API Call ─────────────────────────────────────────────────────────────

def call_groq_report(inference_result: dict, heatmap_text: str, api_key: str) -> dict:
    client = Groq(api_key=api_key)

    label       = inference_result.get("label", "Unknown")
    pneumo_prob = inference_result.get("pneumonia_prob_raw",
                  inference_result.get("pneumonia_prob", 0.0) / 100)
    normal_prob = inference_result.get("normal_prob_raw",
                  inference_result.get("normal_prob", 0.0) / 100)
    routing     = inference_result.get("routing", "Unknown")
    confidence  = max(pneumo_prob, normal_prob) * 100

    system_prompt = """You are an expert radiologist AI assistant generating structured clinical reports
for a chest X-ray pneumonia detection system. Respond ONLY with a valid JSON object —
no preamble, no markdown fences, no extra text.

The JSON must have exactly these 5 keys:
{
  "clinical_summary": "...",
  "findings": "...",
  "heatmap_interpretation": "...",
  "routing_recommendation": "...",
  "disclaimer": "..."
}

Each value must be a single string (1–4 sentences). Be precise, clinical, and professional."""

    user_prompt = f"""Generate a clinical report for this chest X-ray AI analysis:

PREDICTION:
- Classification: {label}
- Pneumonia Probability: {pneumo_prob:.1%}
- Normal Probability:    {normal_prob:.1%}
- Confidence:            {confidence:.1f}%
- Routing:               {routing}

GRAD-CAM HEATMAP ANALYSIS:
{heatmap_text}

Return the 5-section clinical report as a JSON object."""

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
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        return json.loads(raw[start:end]) if start != -1 else {}


# ── Chat endpoint ─────────────────────────────────────────────────────────────

def chat_with_agent(message: str, case_context: Optional[dict], api_key: str) -> str:
    """
    Conversational AI endpoint for the popup chat widget.
    Provides clinical insights about the uploaded X-ray and general pneumonia info.
    """
    client = Groq(api_key=api_key)

    system_prompt = """You are a helpful medical AI assistant for a Pneumonia Detection System.
You help patients and doctors understand chest X-ray analysis results.
Be empathetic, clear, and informative. Always remind users that this is an AI tool
and results must be confirmed by a qualified medical professional.
Keep responses concise (2-4 sentences unless more detail is needed).
Do not diagnose — only explain the AI analysis results."""

    context_str = ""
    if case_context:
        pneumo = case_context.get("pneumonia_prob", "N/A")
        normal = case_context.get("normal_prob", "N/A")
        label  = case_context.get("predicted_class", "N/A")
        conf   = case_context.get("confidence", "N/A")
        context_str = f"""

Current analysis context:
- AI Prediction: {label}
- Pneumonia Probability: {pneumo}%
- Normal Probability: {normal}%
- Confidence: {conf}%
"""

    messages = [
        {"role": "system", "content": system_prompt + context_str},
        {"role": "user",   "content": message},
    ]

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.5,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()


# ── PDF Builder ──────────────────────────────────────────────────────────────

def build_pdf(report_sections, inference_result, heatmap_stats,
              original_image_path, output_path, report_id) -> str:

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=18*mm, rightMargin=18*mm,
        topMargin=15*mm,  bottomMargin=15*mm,
    )

    styles = getSampleStyleSheet()
    story  = []

    def style(name, parent="Normal", **kw):
        return ParagraphStyle(name, parent=styles[parent], **kw)

    s_title = style("s_title", "Title",   fontSize=20, textColor=WHITE, alignment=TA_CENTER)
    s_sub   = style("s_sub",   "Normal",  fontSize=9,  textColor=colors.HexColor("#b0bec5"), alignment=TA_CENTER)
    s_h2    = style("s_h2",    "Heading2",fontSize=11, textColor=NAVY, spaceBefore=6, spaceAfter=3)
    s_body  = style("s_body",  "Normal",  fontSize=9,  textColor=colors.HexColor("#333333"),
                    leading=14, alignment=TA_JUSTIFY, spaceAfter=4)
    s_label = style("s_label", "Normal",  fontSize=8,  textColor=MID, spaceAfter=1)
    s_value = style("s_value", "Normal",  fontSize=10, textColor=NAVY, spaceAfter=6, fontName="Helvetica-Bold")
    s_disc  = style("s_disc",  "Normal",  fontSize=7.5,textColor=MID, leading=11, alignment=TA_JUSTIFY)

    # Header
    header_data = [[
        Paragraph("<b>Medical Imaging QA System</b>", s_title),
        Paragraph("Clinical AI Report", s_title),
    ]]
    header_tbl = Table(header_data, colWidths=["50%", "50%"])
    header_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), NAVY),
        ("ROWPADDING", (0,0), (-1,-1), 10),
        ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(header_tbl)

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

    # Prediction card
    label       = inference_result.get("label", inference_result.get("predicted_class", "Unknown"))
    pneumo_prob = inference_result.get("pneumonia_prob_raw",
                  inference_result.get("pneumonia_prob", 0.0) / 100)
    normal_prob = inference_result.get("normal_prob_raw",
                  inference_result.get("normal_prob", 0.0) / 100)
    routing_val = inference_result.get("routing", inference_result.get("decision", "N/A"))
    confidence  = max(pneumo_prob, normal_prob) * 100
    card_color  = GREEN if label == "Normal" else (RED_C if label == "Pneumonia" else AMBER)

    pred_data = [
        [Paragraph("<b>DIAGNOSIS</b>", s_label), Paragraph("<b>PNEUMONIA PROB</b>", s_label),
         Paragraph("<b>NORMAL PROB</b>", s_label), Paragraph("<b>CONFIDENCE</b>", s_label),
         Paragraph("<b>ROUTING</b>", s_label)],
        [Paragraph(f"<b>{label}</b>", s_value), Paragraph(f"{pneumo_prob:.1%}", s_value),
         Paragraph(f"{normal_prob:.1%}", s_value), Paragraph(f"{confidence:.1f}%", s_value),
         Paragraph(str(routing_val), s_value)],
    ]
    pred_tbl = Table(pred_data, colWidths=["20%","20%","20%","20%","20%"])
    pred_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), LIGHT),
        ("BACKGROUND", (0,0), (0,1),   card_color),
        ("TEXTCOLOR",  (0,0), (0,1),   WHITE),
        ("ROWPADDING", (0,0), (-1,-1), 8),
        ("BOX",        (0,0), (-1,-1), 0.5, colors.HexColor("#cfd8dc")),
        ("INNERGRID",  (0,0), (-1,-1), 0.3, colors.HexColor("#cfd8dc")),
        ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(pred_tbl)
    story.append(Spacer(1, 6*mm))

    # Image panel
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

    # Clinical sections
    sections = [
        ("clinical_summary",       "1. Clinical Summary"),
        ("findings",               "2. Findings"),
        ("heatmap_interpretation", "3. Heatmap Interpretation"),
        ("routing_recommendation", "4. Routing Recommendation"),
    ]
    for key, title in sections:
        text = report_sections.get(key, "Not available.")
        story.append(KeepTogether([
            Paragraph(title, s_h2),
            HRFlowable(width="100%", thickness=0.5, color=TEAL, spaceAfter=4),
            Paragraph(text, s_body),
            Spacer(1, 4*mm),
        ]))

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


# ── Main Entry ────────────────────────────────────────────────────────────────

def run_report_agent(
    inference_result: dict,
    original_image_path: str = None,
    output_path: str = None,
    api_key: str = None,
) -> str:
    api_key = api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Groq API key not found. Set GROQ_API_KEY in .env")

    report_id = f"MIQ-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{str(uuid.uuid4())[:6].upper()}"

    if output_path is None:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = str(REPORT_DIR / f"{report_id}.pdf")

    heatmap       = inference_result.get("heatmap")
    heatmap_stats = analyse_heatmap(heatmap) if heatmap is not None else {}
    heatmap_text  = heatmap_to_text(heatmap_stats)

    report_sections = call_groq_report(inference_result, heatmap_text, api_key)

    pdf_path = build_pdf(
        report_sections=report_sections,
        inference_result=inference_result,
        heatmap_stats=heatmap_stats,
        original_image_path=original_image_path,
        output_path=output_path,
        report_id=report_id,
    )
    return pdf_path
