#!/usr/bin/env python3
"""ETL: Push QA test results to SQLite for agent-queryable analytics.

Usage (standalone):
    python3 db_etl.py --pytest-results _pytest_results_from_log.json
    python3 db_etl.py --init-db  # Initialize the database schema

Called automatically by conftest.py after each test run.

Tables populated:
    runs          — one row per suite execution
    test_scores   — one row per test per run
    turn_metrics  — one row per conversation turn per run
"""

import json
import os
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

_this_dir = Path(__file__).resolve().parent

# ── Database config ─────────────────────────────────────────────────────
DB_PATH = _this_dir / "qa_results.db"
SCHEMA_PATH = _this_dir / "sql" / "sqlite_schema.sql"

# Method classification
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


def _get_db_connection() -> sqlite3.Connection:
    """Get a SQLite database connection, initializing if needed."""
    db_exists = DB_PATH.exists()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Enable dict-like row access

    if not db_exists:
        _init_schema(conn)

    # Additive migration: add tools_source_summary if not present
    try:
        conn.execute("ALTER TABLE turn_metrics ADD COLUMN tools_source_summary TEXT")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # Column already exists

    return conn


def _init_schema(conn: sqlite3.Connection = None):
    """Initialize the database schema from sqlite_schema.sql."""
    close_after = False
    if conn is None:
        conn = sqlite3.connect(DB_PATH)
        close_after = True
    
    if SCHEMA_PATH.exists():
        schema_sql = SCHEMA_PATH.read_text()
        conn.executescript(schema_sql)
        conn.commit()
        print(f"[DB ETL] ✅ Schema initialized: {DB_PATH}")
    else:
        # Fallback: create tables inline
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
              run_id              TEXT    PRIMARY KEY,
              run_date            TEXT    NOT NULL,
              total_tests         INTEGER,
              passed              INTEGER,
              failed              INTEGER,
              pass_rate           REAL,
              session_duration_s  REAL,
              latency_ratio       REAL,
              golden_file         TEXT,
              report_file         TEXT
            );
            
            CREATE TABLE IF NOT EXISTS test_scores (
              id            INTEGER PRIMARY KEY AUTOINCREMENT,
              run_id        TEXT    NOT NULL,
              test_name     TEXT    NOT NULL,
              method        TEXT,
              outcome       TEXT,
              score         TEXT,
              duration_s    REAL,
              failure_reason TEXT,
              FOREIGN KEY (run_id) REFERENCES runs(run_id)
            );
            
            CREATE TABLE IF NOT EXISTS turn_metrics (
              id                  INTEGER PRIMARY KEY AUTOINCREMENT,
              run_id              TEXT    NOT NULL,
              turn_number         INTEGER NOT NULL,
              turn_label          TEXT,
              actual_ms           INTEGER,
              golden_ms           INTEGER,
              hallucination_score REAL,
              hallucination_reason TEXT,
              tools_called        TEXT,
              step_agents         TEXT,
              FOREIGN KEY (run_id) REFERENCES runs(run_id)
            );
            
            CREATE INDEX IF NOT EXISTS idx_test_scores_run_id ON test_scores(run_id);
            CREATE INDEX IF NOT EXISTS idx_turn_metrics_run_id ON turn_metrics(run_id);
            CREATE INDEX IF NOT EXISTS idx_runs_run_date ON runs(run_date);
        """)
        conn.commit()
        print(f"[DB ETL] ✅ Schema initialized (inline): {DB_PATH}")
    
    if close_after:
        conn.close()


def _parse_latency_stdout(stdout: str) -> list[dict]:
    """Extract per-turn latency rows from test_latency_regression stdout.

    Expected format per line:
        1      Turn 1 — Case creation           4470ms     11851ms    120000ms       ok
    """
    rows = []
    for m in re.finditer(
        r"^\s*(\d+)\s+(.+?)\s+(\d+)ms\s+(\d+)ms\s+\d+ms\s+\w+",
        stdout,
        re.MULTILINE,
    ):
        rows.append({
            "turn_number": int(m.group(1)),
            "turn_label": m.group(2).strip(),
            "actual_ms": int(m.group(3)),
            "golden_ms": int(m.group(4)),
        })
    return rows


def _parse_hallucination_stdout(stdout: str) -> list[dict]:
    """Extract per-turn hallucination scores from test_no_hallucination_per_turn stdout.

    Expected format per line:
        1      Turn 1 — Case creation            0.00     ok The hallucination score ...
    """
    rows = []
    for m in re.finditer(
        r"^\s*(\d+)\s+(.+?)\s+([\d.]+)\s+(?:ok|FAIL)\s*(.*)",
        stdout,
        re.MULTILINE,
    ):
        rows.append({
            "turn_number": int(m.group(1)),
            "turn_label": m.group(2).strip(),
            "hallucination_score": float(m.group(3)),
            "hallucination_reason": m.group(4).strip()[:500] or None,
        })
    return rows


def _parse_tools_stdout(stdout: str) -> tuple[dict[int, list[str]], dict[int, str]]:
    """Extract per-turn tools and source summaries from test_tool_invocations_match_golden stdout.

    Supports two formats:
      Legacy:  Turn 1: tools=['pxPerformAssignment']  or  Turn 1: tools=none
      New:     Turn 1 (...): tools=tool_calls:['CreateCasePlug']; regex:['glossary']

    Returns:
      tools_per_turn    — {turn_num: [tool_name, ...]}
      source_per_turn   — {turn_num: "tool_calls:N,regex:M"} (empty string if no data)
    """
    tools_per_turn: dict[int, list[str]] = {}
    source_per_turn: dict[int, str] = {}

    for m in re.finditer(r"Turn\s+(\d+)(?:\s+[^:]+)?:\s+tools=(.+)", stdout):
        turn = int(m.group(1))
        raw = m.group(2).strip()

        if raw == "none":
            tools_per_turn[turn] = []
            source_per_turn[turn] = ""
            continue

        # New format: source:['name1', 'name2']; source2:['name3']
        if re.search(r"(?:tool_calls|regex|plugins):\[", raw):
            all_names: list[str] = []
            summary_parts: list[str] = []
            for src_match in re.finditer(r"(tool_calls|plugins|regex):\[([^\]]*)\]", raw):
                src = src_match.group(1)
                names = re.findall(r"'([^']+)'", src_match.group(2))
                all_names.extend(names)
                if names:
                    summary_parts.append(f"{src}:{len(names)}")
            tools_per_turn[turn] = all_names
            source_per_turn[turn] = ",".join(summary_parts)
        else:
            # Legacy format: ['tool1', 'tool2']
            tools_per_turn[turn] = re.findall(r"'([^']+)'", raw)
            source_per_turn[turn] = ""

    return tools_per_turn, source_per_turn


def _parse_step_agents_stdout(stdout: str) -> list[str]:
    """Extract step agents from test_step_agents_detected stdout.

    Lines like:
        - field_prefill_agent (source=field_prefill, status=completed)
    """
    agents = []
    for m in re.finditer(r"-\s+(\w+)\s+\(source=", stdout):
        agents.append(m.group(1))
    # Step agents are session-level, not per-turn — return as session list
    return agents


def _extract_latency_ratio(stdout: str) -> float | None:
    """Extract ratio from latency test: 'Ratio: 1.16×'"""
    m = re.search(r"Ratio:\s*([\d.]+)", stdout)
    return float(m.group(1)) if m else None


def _extract_score(test_name: str, stdout: str) -> str:
    """Derive a short score string for the scorecard."""
    if test_name == "test_latency_regression":
        ratio = _extract_latency_ratio(stdout)
        return f"{ratio:.2f}x" if ratio else "n/a"
    elif test_name in ("test_tool_invocations_match_golden", "test_no_hallucination_per_turn"):
        # Count actual FAIL markers (uppercase, not header text)
        fail_count = len(re.findall(r"\bFAIL\b", stdout))
        return f"{fail_count} fails"
    elif test_name == "test_step_agents_detected":
        m = re.search(r"Actual step agents:\s*(\d+)", stdout)
        return m.group(1) if m else "n/a"
    elif test_name == "test_case_lifecycle":
        m = re.search(r"Appeared in (\d+/\d+)", stdout)
        return m.group(1) if m else "n/a"
    else:
        # DeepEval LLM-judge tests — score comes from DeepEval JSON, not stdout
        return "1.00"  # default for now; overridden when deepeval JSON is available


def push_to_database(pytest_results: list[dict], run_id: str = None):
    """Push a full test run to SQLite.

    Args:
        pytest_results: List of dicts from conftest.py or _parse_log_to_results.py
                        Each has: nodeid, outcome, duration, longrepr, stdout
        run_id:         Override run ID (default: current timestamp)
    """
    conn = _get_db_connection()
    cursor = conn.cursor()

    if not run_id:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_date = datetime.now(timezone.utc).isoformat()

    # ── 1. Build test_scores rows ─────────────────────────────────────
    test_score_rows = []
    total_duration = 0.0

    for r in pytest_results:
        name = r["nodeid"].split("::")[-1]
        method = "LLM Judge" if name in LLM_JUDGE_TESTS else "Logic"
        stdout = r.get("stdout") or ""
        score = _extract_score(name, stdout)
        dur = r.get("duration", 0.0)
        total_duration += dur

        test_score_rows.append({
            "run_id": run_id,
            "test_name": name,
            "method": method,
            "outcome": r["outcome"],
            "score": score,
            "duration_s": dur,
            "failure_reason": r.get("longrepr"),
        })

    # ── 2. Build turn_metrics rows ────────────────────────────────────
    turn_rows = []

    # Get latency data
    latency_stdout = next(
        (r["stdout"] for r in pytest_results
         if "test_latency_regression" in r["nodeid"] and r.get("stdout")),
        "",
    )
    latency_turns = _parse_latency_stdout(latency_stdout)

    # Get hallucination data
    hallucination_stdout = next(
        (r["stdout"] for r in pytest_results
         if "test_no_hallucination_per_turn" in r["nodeid"] and r.get("stdout")),
        "",
    )
    hallucination_turns = _parse_hallucination_stdout(hallucination_stdout)

    # Get tool invocation data
    tools_stdout = next(
        (r["stdout"] for r in pytest_results
         if "test_tool_invocations_match_golden" in r["nodeid"] and r.get("stdout")),
        "",
    )
    tools_per_turn, tools_source_per_turn = _parse_tools_stdout(tools_stdout)

    # Get step agents (session-level)
    agents_stdout = next(
        (r["stdout"] for r in pytest_results
         if "test_step_agents_detected" in r["nodeid"] and r.get("stdout")),
        "",
    )
    session_step_agents = _parse_step_agents_stdout(agents_stdout)

    # Merge turn data (keyed by turn number)
    all_turn_nums = sorted(set(
        [t["turn_number"] for t in latency_turns]
        + [t["turn_number"] for t in hallucination_turns]
    ))

    latency_map = {t["turn_number"]: t for t in latency_turns}
    halluc_map = {t["turn_number"]: t for t in hallucination_turns}

    for turn_num in all_turn_nums:
        lat = latency_map.get(turn_num, {})
        hal = halluc_map.get(turn_num, {})

        turn_rows.append({
            "run_id": run_id,
            "turn_number": turn_num,
            "turn_label": lat.get("turn_label") or hal.get("turn_label") or f"Turn {turn_num}",
            "actual_ms": lat.get("actual_ms"),
            "golden_ms": lat.get("golden_ms"),
            "hallucination_score": hal.get("hallucination_score"),
            "hallucination_reason": hal.get("hallucination_reason"),
            "tools_called": json.dumps(tools_per_turn.get(turn_num, [])),
            "tools_source_summary": tools_source_per_turn.get(turn_num, "") or None,
            "step_agents": json.dumps(session_step_agents if turn_num == 1 else []),
        })

    # ── 3. Build runs row ─────────────────────────────────────────────
    passed = sum(1 for r in pytest_results if r["outcome"] == "passed")
    failed = sum(1 for r in pytest_results if r["outcome"] in ("failed", "error"))
    total = len(pytest_results)
    latency_ratio = _extract_latency_ratio(latency_stdout)

    runs_row = {
        "run_id": run_id,
        "run_date": run_date,
        "total_tests": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": round(passed / total * 100, 2) if total else 0.0,
        "session_duration_s": round(total_duration, 3),
        "latency_ratio": latency_ratio,
        "golden_file": None,  # Set by caller if known
        "report_file": f"QA_Report_{run_id}.md",
    }

    # ── 4. Insert into SQLite ─────────────────────────────────────────
    try:
        # Insert runs row
        cursor.execute("""
            INSERT OR REPLACE INTO runs 
            (run_id, run_date, total_tests, passed, failed, pass_rate, 
             session_duration_s, latency_ratio, golden_file, report_file)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            runs_row["run_id"], runs_row["run_date"], runs_row["total_tests"],
            runs_row["passed"], runs_row["failed"], runs_row["pass_rate"],
            runs_row["session_duration_s"], runs_row["latency_ratio"],
            runs_row["golden_file"], runs_row["report_file"]
        ))
        print(f"  [DB ETL] ✅ runs: 1 row inserted")

        # Insert test_scores rows
        for row in test_score_rows:
            cursor.execute("""
                INSERT INTO test_scores 
                (run_id, test_name, method, outcome, score, duration_s, failure_reason)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                row["run_id"], row["test_name"], row["method"], row["outcome"],
                row["score"], row["duration_s"], row["failure_reason"]
            ))
        print(f"  [DB ETL] ✅ test_scores: {len(test_score_rows)} rows inserted")

        # Insert turn_metrics rows
        for row in turn_rows:
            cursor.execute("""
                INSERT INTO turn_metrics
                (run_id, turn_number, turn_label, actual_ms, golden_ms,
                 hallucination_score, hallucination_reason, tools_called,
                 tools_source_summary, step_agents)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row["run_id"], row["turn_number"], row["turn_label"],
                row["actual_ms"], row["golden_ms"], row["hallucination_score"],
                row["hallucination_reason"], row["tools_called"],
                row.get("tools_source_summary"), row["step_agents"]
            ))
        print(f"  [DB ETL] ✅ turn_metrics: {len(turn_rows)} rows inserted")

        conn.commit()
        print(f"  [DB ETL] ✅ Run {run_id} pushed to SQLite ({total} tests, {len(turn_rows)} turns)")
        success = True

    except Exception as e:
        conn.rollback()
        print(f"  [DB ETL] ❌ Error: {e}")
        success = False

    finally:
        conn.close()

    return success


# Backwards compatibility alias
push_to_bigquery = push_to_database


# ── CLI ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Push QA results to SQLite database")
    parser.add_argument(
        "--pytest-results",
        dest="pytest_results",
        help="Path to _pytest_results JSON file",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        dest="run_id",
        help="Override run ID (default: current timestamp)",
    )
    parser.add_argument(
        "--init-db",
        action="store_true",
        dest="init_db",
        help="Initialize the database schema",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Parse and print what would be inserted without writing to database",
    )
    args = parser.parse_args()

    if args.init_db:
        _init_schema()
        sys.exit(0)

    if not args.pytest_results:
        parser.error("--pytest-results is required unless --init-db is specified")

    with open(args.pytest_results) as f:
        results = json.load(f)

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.dry_run:
        print(f"DRY RUN — run_id: {run_id}")
        print(f"  Tests: {len(results)}")
        for r in results:
            name = r["nodeid"].split("::")[-1]
            print(f"    {name}: {r['outcome']} ({r.get('duration', 0)}s)")

        # Parse and display turn data
        latency_stdout = next(
            (r["stdout"] for r in results
             if "test_latency_regression" in r["nodeid"] and r.get("stdout")), ""
        )
        latency_turns = _parse_latency_stdout(latency_stdout)
        print(f"\n  Latency turns parsed: {len(latency_turns)}")
        for t in latency_turns:
            print(f"    Turn {t['turn_number']}: {t['actual_ms']}ms (golden: {t['golden_ms']}ms)")

        halluc_stdout = next(
            (r["stdout"] for r in results
             if "test_no_hallucination_per_turn" in r["nodeid"] and r.get("stdout")), ""
        )
        halluc_turns = _parse_hallucination_stdout(halluc_stdout)
        print(f"\n  Hallucination turns parsed: {len(halluc_turns)}")
        for t in halluc_turns:
            print(f"    Turn {t['turn_number']}: score={t['hallucination_score']:.2f}")

        tools_stdout = next(
            (r["stdout"] for r in results
             if "test_tool_invocations_match_golden" in r["nodeid"] and r.get("stdout")), ""
        )
        tools, tools_src = _parse_tools_stdout(tools_stdout)
        print(f"\n  Tool invocation turns parsed: {len(tools)}")

        agents_stdout = next(
            (r["stdout"] for r in results
             if "test_step_agents_detected" in r["nodeid"] and r.get("stdout")), ""
        )
        agents = _parse_step_agents_stdout(agents_stdout)
        print(f"  Step agents detected: {agents}")
    else:
        push_to_database(results, run_id=run_id)
