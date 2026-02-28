import sqlite3
import logging
from contextlib import contextmanager

from config import DATABASE_PATH

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS transcriptions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    filename        TEXT NOT NULL,
    model           TEXT NOT NULL,
    text            TEXT,
    wer             REAL,
    cer             REAL,
    bleu            REAL,
    duration        REAL,
    confidence      REAL,
    reference       TEXT,
    translation     TEXT,
    source_language TEXT,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""


def init_db():
    """Create database tables if they do not exist yet."""
    with _get_connection() as conn:
        conn.executescript(SCHEMA)
    _migrate_db()
    logger.info("Database initialised at %s", DATABASE_PATH)


def _migrate_db():
    """Add new columns to pre-existing databases (forward-compatible).

    Column names and types are taken from a hardcoded allowlist — they are
    never derived from user input — so constructing the ALTER TABLE statement
    with string formatting is safe here.
    """
    # Allowlist: only these column definitions may ever be added.
    _ALLOWED_COLUMNS = {
        'translation': 'TEXT',
        'source_language': 'TEXT',
        'bleu': 'REAL',
    }
    with _get_connection() as conn:
        for col, col_type in _ALLOWED_COLUMNS.items():
            try:
                conn.execute(f"ALTER TABLE transcriptions ADD COLUMN {col} {col_type}")
            except Exception:
                pass  # Column already exists — safe to ignore


@contextmanager
def _get_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def save_transcription(filename: str, model: str, text: str,
                       duration: float, confidence: float,
                        wer: float = None, cer: float = None,
                       bleu: float = None,
                       reference: str = None,
                       translation: str = None,
                       source_language: str = None) -> int:
    """Insert a transcription record and return its id."""
    sql = """
        INSERT INTO transcriptions
            (filename, model, text, wer, cer, bleu, duration, confidence, reference,
             translation, source_language)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with _get_connection() as conn:
        cursor = conn.execute(
            sql,
            (filename, model, text, wer, cer, bleu, duration, confidence, reference,
             translation, source_language),
        )
        return cursor.lastrowid


def get_transcription(record_id: int) -> dict:
    """Return a single transcription record as a dict or None."""
    sql = "SELECT * FROM transcriptions WHERE id = ?"
    with _get_connection() as conn:
        row = conn.execute(sql, (record_id,)).fetchone()
    return dict(row) if row else None


def get_all_transcriptions(limit: int = 100) -> list:
    """Return all transcription records ordered by newest first."""
    sql = "SELECT * FROM transcriptions ORDER BY created_at DESC LIMIT ?"
    with _get_connection() as conn:
        rows = conn.execute(sql, (limit,)).fetchall()
    return [dict(r) for r in rows]


def delete_old_files(days: int = 7):
    """Remove transcription records older than `days` days."""
    sql = "DELETE FROM transcriptions WHERE created_at < datetime('now', ?)"
    with _get_connection() as conn:
        conn.execute(sql, (f'-{abs(days)} days',))
    logger.info("Deleted transcriptions older than %d days", days)
