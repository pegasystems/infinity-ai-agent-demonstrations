#!/usr/bin/env python3
"""Insert evaluation results into the qa_results.db SQLite database.

Reads a pytest results JSON file and inserts the structured data into three
tables: runs, test_scores, and turn_metrics.  Uses the SQL scripts in sql/
for each INSERT statement.

Usage:
    # Insert from a specific results JSON file
    python3 insert_results.py --pytest-results test_results/_pytest_results_xyz.json

    # Override the run ID
    python3 insert_results.py --pytest-results results.json --run-id 20260303_120000

    # Dry run — show what would be inserted
    python3 insert_results.py --pytest-results results.json --dry-run
"""

import json
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

_this_dir = Path(__file__).resolve().parent

# ── Paths ──────────────────────────────────────────────────────────────
DB_PATH = _this_dir / "qa_results.db"
SQL_DIR = _this_dir / "sql"
SCHEMA_SQL = SQL_DIR / "sqlite_schema.sql"
INSERT_RUN_SQL = SQL_DIR / "insert_run.sql"
INSERT_TEST_SCORE_SQL = SQL_DIR / "insert_test_score.sql"
INSERT_TURN_METRIC_SQL = SQL_DIR / "insert_turn_metric.sql"

# ── Test classification ────────────────────────────────────────────────
LLM_JUDGE_TESTS = {
    "test_knowledge_retention",
    "test_conversation_completeness",
    "test_role_adherence",
    "test_no_hallucination_per_turn",
}
LOGIC_TESTS = {
    "test_tool_invocations_match_golden",
    "test_latency_regression",
    "test_case_lifecycle",
    "test_step_agents_detected",
}


# ── SQL loader ─────────────────────────────────────────────────────────
def _load_sql(path: Path) -> str:
    """Read a .sql file and return its contents."""
    return path.read_text()


# ── Database helpers ───────────────────────────────────────────────────
def get_connection() -> sqlite3.Connection:
    """Return a connection, initialising the schema if the DB is new."""
    is_new = not DB_PATH.exists()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    if is_new:
        init_schema(conn)
    return conn


def init_schema(conn: sqlite3.Connection | None = None):
    """Create / reset tables from sql/sqlite_schema.sql."""
    close_after = conn is None
    if conn is None:
        conn = sqlite3.connect(DB_PATH)

    conn.executescript(_load_sql(SCHEMA_SQL))
    conn.commit()
    print(f"[insert_results] Schema initialised → {DB_PATH}")

    if close_after:
        conn.close()


# ── Stdout parsers (re-used from db_etl logic) ────────────────────────
def _parse_latency_turns(stdout: str) -> list[dict]:
    rows = []
    for m in re.finditer(
        r"^\s*(\d+)\s+(.+?)\s+(\d+)ms\s+(\d+)ms\s+\d+ms\s+\w+",
        stdout, re.MULTILINE,
    ):
        rows.append({
            "turn_number": int(m.group(1)),
            "turn_label": m.group(2).strip(),
            "actual_ms": int(m.group(3)),
            "golden_ms": int(m.group(4)),
        })
    return rows


def _parse_hallucination_turns(stdout: str) -> list[dict]:
    rows = []
    for m in re.finditer(
        r"^\s*(\d+)\s+(.+?)\s+([\d.]+)\s+(?:ok|FAIL)\s*(.*)",
        stdout, re.MULTILINE,
    ):
        rows.append({
            "turn_number": int(m.group(1)),
            "turn_label": m.group(2).strip(),
            "hallucination_score": float(m.group(3)),
            "hallucination_reason": m.group(4).strip()[:500] or None,
        })
    return rows


def _parse_tools_per_turn(stdout: str) -> dict[int, list[str]]:
    result = {}
    for m in re.finditer(r"Turn\s+(\d+):\s+tools=(.+)", stdout):
        turn = int(m.group(1))
        raw = m.group(2).strip()
        result[turn] = [] if raw == "none" else re.findall(r"'([^']+)'", raw)
    return result


def _parse_step_agents(stdout: str) -> list[str]:
    return [m.group(1) for m in re.finditer(r"-\s+(\w+)\s+\(source=", stdout)]


def _extract_latency_ratio(stdout: str) -> float | None:
    m = re.search(r"Ratio:\s*([\d.]+)", stdout)
    return float(m.group(1)) if m else None


def _extract_score(test_name: str, stdout: str) -> str:
    if test_name == "test_latency_regression":
        ratio = _extract_latency_ratio(stdout)
        return f"{ratio:.2f}x" if ratio else "n/a"
    if test_name in ("test_tool_invocations_match_golden", "test_no_hallucination_per_turn"):
        return f"{len(re.findall(r'\\bFAIL\\b', stdout))} fails"
    if test_name == "test_step_agents_detected":
        m = re.search(r"Actual step agents:\s*(\d+)", stdout)
        return m.group(1) if m else "n/a"
    if test_name == "test_case_lifecycle":
        m = re.search(r"Appeared in (\d+/\d+)", stdout)
        return m.group(1) if m else "n/a"
    return "1.00"


# ── Core insert logic ─────────────────────────────────────────────────
def insert_results(pytest_results: list[dict], *, run_id: str | None = None,
                   golden_file: str | None = None) -> bool:
    """Parse pytest results and insert into qa_results.db.

    Args:
        pytest_results: List of dicts with keys: nodeid, outcome, duration,
                        longrepr, stdout.
        run_id:         Override run ID (default: current timestamp).
        golden_file:    Path to the golden session file used, if known.

    Returns:
        True on success, False on error.
    """
    conn = get_connection()
    cur = conn.cursor()

    if not run_id:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_date = datetime.now(timezone.utc).isoformat()

    # Load SQL templates
    insert_run_sql = _load_sql(INSERT_RUN_SQL)
    insert_score_sql = _load_sql(INSERT_TEST_SCORE_SQL)
    insert_turn_sql = _load_sql(INSERT_TURN_METRIC_SQL)

    # ── Build test_scores rows ─────────────────────────────────────────
    score_rows = []
    total_duration = 0.0
    for r in pytest_results:
        name = r["nodeid"].split("::")[-1]
        method = "LLM Judge" if name in LLM_JUDGE_TESTS else "Logic"
        stdout = r.get("stdout") or ""
        score_rows.append((
            run_id, name, method, r["outcome"],
            _extract_score(name, stdout),
            r.get("duration", 0.0),
            r.get("longrepr"),
        ))
        total_duration += r.get("duration", 0.0)

    # ── Build turn_metrics rows ────────────────────────────────────────
    def _stdout_for(keyword):
        return next(
            (r["stdout"] for r in pytest_results
             if keyword in r["nodeid"] and r.get("stdout")), ""
        )

    lat_turns = _parse_latency_turns(_stdout_for("test_latency_regression"))
    hal_turns = _parse_hallucination_turns(_stdout_for("test_no_hallucination_per_turn"))
    tools_map = _parse_tools_per_turn(_stdout_for("test_tool_invocations_match_golden"))
    agents = _parse_step_agents(_stdout_for("test_step_agents_detected"))
    latency_ratio = _extract_latency_ratio(_stdout_for("test_latency_regression"))

    lat_map = {t["turn_number"]: t for t in lat_turns}
    hal_map = {t["turn_number"]: t for t in hal_turns}
    all_turns = sorted({t["turn_number"] for t in lat_turns}
                       | {t["turn_number"] for t in hal_turns})

    turn_rows = []
    for tn in all_turns:
        lat = lat_map.get(tn, {})
        hal = hal_map.get(tn, {})
        turn_rows.append((
            run_id, tn,
            lat.get("turn_label") or hal.get("turn_label") or f"Turn {tn}",
            lat.get("actual_ms"), lat.get("golden_ms"),
            hal.get("hallucination_score"), hal.get("hallucination_reason"),
            json.dumps(tools_map.get(tn, [])),
            json.dumps(agents if tn == 1 else []),
        ))

    # ── Build runs row ─────────────────────────────────────────────────
    passed = sum(1 for r in pytest_results if r["outcome"] == "passed")
    failed = sum(1 for r in pytest_results if r["outcome"] in ("failed", "error"))
    total = len(pytest_results)
    run_row = (
        run_id, run_date, total, passed, failed,
        round(passed / total * 100, 2) if total else 0.0,
        round(total_duration, 3), latency_ratio,
        golden_file, f"QA_Report_{run_id}.md",
    )

    # ── Execute inserts ────────────────────────────────────────────────
    try:
        cur.execute(insert_run_sql, run_row)
        for row in score_rows:
            cur.execute(insert_score_sql, row)
        for row in turn_rows:
            cur.execute(insert_turn_sql, row)

        conn.commit()
        print(f"[insert_results] ✅ Run {run_id} → "
              f"{total} tests, {len(turn_rows)} turns inserted into {DB_PATH.name}")
        return True

    except Exception as exc:
        conn.rollback()
        print(f"[insert_results] ❌ {exc}", file=sys.stderr)
        return False

    finally:
        conn.close()


# ── CLI ────────────────────────────────────────────────────────────────
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Insert evaluation results into qa_results.db",
    )
    parser.add_argument(
        "--pytest-results",
        help="Path to the pytest results JSON file",
    )
    parser.add_argument("--run-id", default=None, help="Override run ID")
    parser.add_argument("--golden-file", default=None, help="Golden session file path")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Parse and display what would be inserted without writing to DB",
    )
    parser.add_argument(
        "--init-db", action="store_true",
        help="Initialise the database schema and exit",
    )
    args = parser.parse_args()

    if args.init_db:
        init_schema()
        if not args.pytest_results:
            return

    if not args.pytest_results:
        parser.error("--pytest-results is required unless --init-db is used")

    results_path = Path(args.pytest_results)
    if not results_path.exists():
        parser.error(f"File not found: {results_path}")

    with open(results_path) as f:
        results = json.load(f)

    rid = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.dry_run:
        print(f"DRY RUN — run_id: {rid}")
        print(f"  Tests: {len(results)}")
        for r in results:
            name = r["nodeid"].split("::")[-1]
            print(f"    {name}: {r['outcome']} ({r.get('duration', 0):.3f}s)")
        lat_stdout = next(
            (r["stdout"] for r in results
             if "test_latency_regression" in r["nodeid"] and r.get("stdout")), ""
        )
        turns = _parse_latency_turns(lat_stdout)
        print(f"  Latency turns: {len(turns)}")
        hal_stdout = next(
            (r["stdout"] for r in results
             if "test_no_hallucination_per_turn" in r["nodeid"] and r.get("stdout")), ""
        )
        hturns = _parse_hallucination_turns(hal_stdout)
        print(f"  Hallucination turns: {len(hturns)}")
    else:
        ok = insert_results(results, run_id=rid, golden_file=args.golden_file)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
