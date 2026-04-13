"""
run_server.py
=============
Spyder-compatible launcher for the Medical Imaging QA System.

HOW TO RUN IN SPYDER:
    1. Open this file in Spyder
    2. Press F5 (Run) OR run in the Spyder console:
           %run run_server.py
    3. Open browser: http://localhost:8000

HOW TO RUN FROM TERMINAL:
    python run_server.py
    
    OR using uvicorn directly:
    python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""

import os
import sys

# ── Add project root to path ──────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR      = os.path.join(PROJECT_ROOT, "src")

for _p in [PROJECT_ROOT, SRC_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Launch ────────────────────────────────────────────────────────────────────
import uvicorn

print("=" * 55)
print("  Medical Imaging Quality Assurance System")
print("  URL: http://localhost:8000")
print("  Press Ctrl+C to stop")
print("=" * 55)

uvicorn.run(
    "api.main:app",
    host="0.0.0.0",
    port=8000,
    reload=False,      # Keep False for Spyder compatibility
    log_level="info",
)
