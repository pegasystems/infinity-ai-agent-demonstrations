"""Create the qa_results.db SQLite database and initialise its schema."""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "qa_results.db"
SCHEMA_PATH = Path(__file__).parent / "sql" / "sqlite_schema.sql"


def create_database():
    schema = SCHEMA_PATH.read_text()
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(schema)
    conn.close()
    print(f"Database created at {DB_PATH}")


if __name__ == "__main__":
    create_database()
