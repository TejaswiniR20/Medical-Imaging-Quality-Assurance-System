"""
database.py — SQLite database initialization and helpers
"""
import sqlite3
import os
from pathlib import Path
from passlib.context import CryptContext

DB_PATH = os.getenv("DB_PATH", "pneumonia_qa.db")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    with get_db() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS cases (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id      TEXT    NOT NULL,
            original_image  TEXT    NOT NULL,
            gradcam_image   TEXT,
            pneumonia_prob  REAL,
            normal_prob     REAL,
            predicted_class TEXT,
            confidence      REAL,
            decision        TEXT,
            needs_review    INTEGER DEFAULT 0,
            doctor_username TEXT,
            doctor_diagnosis TEXT,
            doctor_notes    TEXT,
            reviewed_at     TEXT,
            report_pdf      TEXT,
            created_at      TEXT    DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS doctors (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT    UNIQUE NOT NULL,
            password_hash TEXT    NOT NULL,
            full_name     TEXT,
            created_at    TEXT    DEFAULT (datetime('now'))
        );
        """)
        conn.commit()
        _seed_default_doctor(conn)


def _seed_default_doctor(conn):
    """Create a default doctor account if none exists."""
    row = conn.execute("SELECT id FROM doctors WHERE username='doctor1'").fetchone()
    if not row:
        hashed = pwd_context.hash("doctor123")
        conn.execute(
            "INSERT INTO doctors (username, password_hash, full_name) VALUES (?,?,?)",
            ("doctor1", hashed, "Dr. Sarah Johnson")
        )
        conn.commit()
        print("[DB] Default doctor created → username: doctor1 / password: doctor123")


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def hash_password(plain: str) -> str:
    return pwd_context.hash(plain)


def insert_case(session_id, original_image, gradcam_image,
                pneumonia_prob, normal_prob, predicted_class,
                confidence, decision, needs_review) -> int:
    with get_db() as conn:
        cur = conn.execute(
            """INSERT INTO cases
               (session_id, original_image, gradcam_image,
                pneumonia_prob, normal_prob, predicted_class,
                confidence, decision, needs_review)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (session_id, original_image, gradcam_image,
             pneumonia_prob, normal_prob, predicted_class,
             confidence, decision, int(needs_review))
        )
        conn.commit()
        return cur.lastrowid


def get_cases_for_session(session_id: str):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM cases WHERE session_id=? ORDER BY created_at DESC",
            (session_id,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_all_cases():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM cases ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def get_case_by_id(case_id: int):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM cases WHERE id=?", (case_id,)).fetchone()
        return dict(row) if row else None


def update_doctor_review(case_id: int, doctor_username: str,
                          doctor_diagnosis: str, doctor_notes: str):
    with get_db() as conn:
        conn.execute(
            """UPDATE cases SET
               doctor_username=?, doctor_diagnosis=?, doctor_notes=?,
               reviewed_at=datetime('now')
               WHERE id=?""",
            (doctor_username, doctor_diagnosis, doctor_notes, case_id)
        )
        conn.commit()


def update_report_pdf(case_id: int, pdf_path: str):
    with get_db() as conn:
        conn.execute("UPDATE cases SET report_pdf=? WHERE id=?", (pdf_path, case_id))
        conn.commit()


def get_doctor_by_username(username: str):
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM doctors WHERE username=?", (username,)
        ).fetchone()
        return dict(row) if row else None
