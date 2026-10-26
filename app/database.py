import os
import sqlite3
from datetime import datetime

def get_connection(db_path: str = None):
    """
    Returns an active database connection and engine type ('postgres' or 'sqlite').
    Dynamically switches based on the DATABASE_URL environment variable.
    """
    url = os.getenv("DATABASE_URL")
    if url and (url.startswith("postgresql://") or url.startswith("postgres://")):
        try:
            import psycopg2
            import psycopg2.extras
            conn = psycopg2.connect(url)
            return conn, "postgres"
        except ImportError:
            print("[Database Warning] psycopg2 not found, falling back to SQLite.")
    
    path = db_path or os.getenv("DATABASE_PATH", "database/visionguard_history.db")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn, "sqlite"

def check_connection():
    """Pings the database to verify live connectivity (works for SQLite and PostgreSQL)."""
    try:
        conn, engine = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1;")
        conn.close()
        return True
    except Exception as e:
        print(f"[Database Ping Error]: {e}")
        return False


def init_db(db_path: str = None):
    """Initializes the audit schema across SQLite or PostgreSQL."""
    conn, engine = get_connection(db_path)
    cursor = conn.cursor()
    if engine == "postgres":
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
                timestamp VARCHAR(64) NOT NULL,
                filename VARCHAR(256) NOT NULL,
                verdict VARCHAR(64) NOT NULL,
                verdict_code VARCHAR(16) NOT NULL,
                confidence REAL NOT NULL,
                raw_probability REAL NOT NULL,
                latency_ms REAL NOT NULL,
                model_version VARCHAR(64) NOT NULL
            );
        """)
        conn.commit()
        print(f"[Database] PostgreSQL schema verified on external service.")
    else:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                filename TEXT NOT NULL,
                verdict TEXT NOT NULL,
                verdict_code TEXT NOT NULL,
                confidence REAL NOT NULL,
                raw_probability REAL NOT NULL,
                latency_ms REAL NOT NULL,
                model_version TEXT NOT NULL
            );
        """)
        conn.commit()
        print(f"[Database] SQLite schema verified at: {db_path or 'database/visionguard_history.db'}")
    conn.close()

def log_prediction(db_path: str, filename: str, verdict: str, verdict_code: str,
                   confidence: float, raw_prob: float, latency_ms: float, model_version: str):
    """Inserts an immutable audit record for an inference request (Task 5 requirement)."""
    conn, engine = get_connection(db_path)
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if engine == "postgres":
        cursor.execute("""
            INSERT INTO predictions (
                timestamp, filename, verdict, verdict_code,
                confidence, raw_probability, latency_ms, model_version
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
        """, (now_str, filename, verdict, verdict_code, confidence, raw_prob, latency_ms, model_version))
    else:
        cursor.execute("""
            INSERT INTO predictions (
                timestamp, filename, verdict, verdict_code,
                confidence, raw_probability, latency_ms, model_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """, (now_str, filename, verdict, verdict_code, confidence, raw_prob, latency_ms, model_version))
        
    conn.commit()
    conn.close()

def get_history(db_path: str = None, limit: int = 50):
    """Retrieves recent prediction records for frontend history visualization."""
    conn, engine = get_connection(db_path)
    cursor = conn.cursor()
    if engine == "postgres":
        import psycopg2.extras
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as dict_cur:
            dict_cur.execute("""
                SELECT id, timestamp, filename, verdict, verdict_code,
                       confidence, raw_probability, latency_ms, model_version
                FROM predictions
                ORDER BY id DESC
                LIMIT %s;
            """, (limit,))
            rows = [dict(r) for r in dict_cur.fetchall()]
        conn.close()
        return rows
    else:
        cursor.execute("""
            SELECT id, timestamp, filename, verdict, verdict_code,
                   confidence, raw_probability, latency_ms, model_version
            FROM predictions
            ORDER BY id DESC
            LIMIT ?;
        """, (limit,))
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return rows

def get_stats(db_path: str = None):
    """Computes summary statistics for the history view."""
    conn, engine = get_connection(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) AS total FROM predictions;")
    total = cursor.fetchone()[0]
    
    if engine == "postgres":
        cursor.execute("SELECT COUNT(*) FROM predictions WHERE verdict_code = 'REAL';")
        real_cnt = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM predictions WHERE verdict_code = 'AI';")
        ai_cnt = cursor.fetchone()[0]
        cursor.execute("SELECT AVG(latency_ms) FROM predictions;")
        avg_lat = cursor.fetchone()[0] or 0.0
    else:
        cursor.execute("SELECT COUNT(*) FROM predictions WHERE verdict_code = 'REAL';")
        real_cnt = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM predictions WHERE verdict_code = 'AI';")
        ai_cnt = cursor.fetchone()[0]
        cursor.execute("SELECT AVG(latency_ms) FROM predictions;")
        avg_lat = cursor.fetchone()[0] or 0.0
        
    conn.close()
    return {
        "total_queries": total,
        "real_count": real_cnt,
        "ai_count": ai_cnt,
        "avg_latency_ms": round(float(avg_lat), 2)
    }
