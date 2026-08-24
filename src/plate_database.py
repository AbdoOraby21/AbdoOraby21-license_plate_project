import sqlite3
from pathlib import Path
from datetime import datetime


# =========================
# Database Path
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent

DATABASE_PATH = (
    BASE_DIR
    / "runs"
    / "license_plates.db"
)

DATABASE_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# =========================
# Initialize Database
# =========================

def init_database():

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detected_plates (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            plate_number TEXT,

            confidence REAL,

            image_name TEXT,

            detected_at TEXT

        )
    """)

    connection.commit()

    connection.close()


# =========================
# Save Detection
# =========================

def save_plate(
    plate_number,
    confidence,
    image_name
):

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = connection.cursor()

    detected_at = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    cursor.execute(
        """
        INSERT INTO detected_plates
        (
            plate_number,
            confidence,
            image_name,
            detected_at
        )
        VALUES (?, ?, ?, ?)
        """,
        (
            plate_number,
            confidence,
            image_name,
            detected_at
        )
    )

    connection.commit()

    connection.close()


# =========================
# Read All Plates
# =========================

def get_all_plates():

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            id,
            plate_number,
            confidence,
            image_name,
            detected_at
        FROM detected_plates
        ORDER BY id DESC
        """
    )

    rows = cursor.fetchall()

    connection.close()

    return rows


# =========================
# Main Test
# =========================

if __name__ == "__main__":

    init_database()

    print(
        "Database initialized:"
    )

    print(
        DATABASE_PATH
    )