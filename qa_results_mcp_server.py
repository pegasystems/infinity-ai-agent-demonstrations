#!/usr/bin/env python3
"""MCP Server: QA Results Agent

An MCP-powered agent that lets you chat with your QA test results.
Queries SQLite for structured metrics and reads local report files.

Usage:
    # Local dev (stdio for Claude Desktop)
    python3 qa_results_mcp_server.py

    # HTTP transport (for Pega / remote clients)
    PORT=8090 python3 qa_results_mcp_server.py --transport http

    # FastMCP dev inspector
    fastmcp dev qa_results_mcp_server.py
"""

import os
import re
import json
import sqlite3
from pathlib import Path
from datetime import datetime

from fastmcp import FastMCP

# ── Config ──────────────────────────────────────────────────────────────
_this_dir = Path(__file__).resolve().parent
DB_PATH = _this_dir / "qa_results.db"


# ── MCP Server ──────────────────────────────────────────────────────────
mcp = FastMCP(
    name="QA Results Agent",
    instructions=(
        "You are a QA analytics agent. "
        "You help engineers and stakeholders understand test results, identify trends, "
        "and diagnose failures in the Pega agent's behavior.\n\n"
        "You have access to a SQLite database with historical test results and local QA report files.\n\n"
        "When answering questions:\n"
        "- Always query the database for quantitative data (scores, latencies, pass rates)\n"
        "- Use get_report_section() for narrative analysis from the latest report\n"
        "- Compare current vs historical data when asked about trends\n"
        "- Be specific: cite turn numbers, scores, and timestamps\n"
        "- If data is missing, say so rather than guessing\n\n"
        "Database tables:\n"
        "- runs: one row per test suite execution\n"
        "- test_scores: one row per test per run\n"
        "- turn_metrics: one row per conversation turn per run"
    ),
)


def _get_db_connection() -> sqlite3.Connection:
    """Get a SQLite database connection."""
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}. Run tests first to create it.")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _rows_to_markdown(cursor: sqlite3.Cursor) -> str:
    """Convert SQLite query results to a Markdown table."""
    rows = cursor.fetchall()
    if not rows:
        return "Query returned 0 rows."

    # Get column names from cursor description
    columns = [desc[0] for desc in cursor.description]

    # Build markdown table
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, separator]

    for row in rows[:100]:
        vals = []
        for col in columns:
            v = row[col]
            if v is None:
                vals.append("")
            elif isinstance(v, str) and v.startswith("["):
                # JSON array - format nicely
                try:
                    arr = json.loads(v)
                    vals.append(", ".join(str(x) for x in arr))
                except json.JSONDecodeError:
                    vals.append(str(v))
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")

    result = "\n".join(lines)
    if len(rows) > 100:
        result += f"\n\n... ({len(rows)} total rows, showing first 100)"
    return result


# ── Tool: Query SQLite Database ─────────────────────────────────────────

@mcp.tool
def query_qa_data(sql: str) -> str:
    """Run a read-only SQL query against the QA results SQLite database.

    Available tables:
    - runs (run_id, run_date, total_tests, passed, failed, pass_rate, session_duration_s, latency_ratio, golden_file, report_file)
    - test_scores (run_id, test_name, method, outcome, score, duration_s, failure_reason)
    - turn_metrics (run_id, turn_number, turn_label, actual_ms, golden_ms, hallucination_score, hallucination_reason, tools_called, step_agents)

    Examples:
    - "SELECT * FROM runs ORDER BY run_date DESC LIMIT 5"
    - "SELECT turn_number, actual_ms, golden_ms FROM turn_metrics WHERE actual_ms > 30000"
    - "SELECT test_name, outcome, score FROM test_scores WHERE run_id = '20260223_114609'"
    """
    # Safety: only allow SELECT statements
    stripped = sql.strip().upper()
    if not stripped.startswith("SELECT"):
        return "Error: Only SELECT queries are allowed. No INSERT, UPDATE, DELETE, DROP, or DDL."

    # Block dangerous patterns
    dangerous = re.search(r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|MERGE)\b", stripped)
    if dangerous:
        return f"Error: {dangerous.group()} statements are not allowed."

    try:
        conn = _get_db_connection()
        cursor = conn.execute(sql)
        result = _rows_to_markdown(cursor)
        conn.close()
        return result
    except Exception as e:
        return f"Database error: {e}"


# ── Tool: Get Report Section ────────────────────────────────────────────

@mcp.tool
def get_report_section(section_name: str) -> str:
    """Get a specific section from the latest QA report (latest_qa_report.md).

    Section names (case-insensitive, partial match):
    - "summary" → Executive Summary
    - "scorecard" → Full Test Scorecard
    - "hallucination" → Hallucination Analysis
    - "deepeval" or "knowledge" or "completeness" or "role" → DeepEval Deep-Dives
    - "latency" → Latency Analysis
    - "tool" → Tool Invocation Correctness
    - "step agent" → Step Agent Detection
    - "lifecycle" or "case" → Business Case Lifecycle
    - "failure" → Failure Deep-Dives
    - "flow" → Conversation Flow Summary
    - "risk" → Regression Risk Assessment
    - "actions" or "recommended" → Recommended Actions
    - "all" → Full report
    """
    report_path = _this_dir / "latest_qa_report.md"
    if not report_path.exists():
        return "No report found. Run the test suite first."

    content = report_path.read_text()

    if section_name.lower() == "all":
        # Truncate if too large
        if len(content) > 15000:
            return content[:15000] + "\n\n... (truncated, use a specific section name for details)"
        return content

    # Split by ## headers and find matching section
    sections = re.split(r"(?=^## )", content, flags=re.MULTILINE)
    search = section_name.lower()

    for section in sections:
        header_match = re.match(r"^## \d*\.?\s*(.*)", section)
        if header_match and search in header_match.group(1).lower():
            return section.strip()

    # Fuzzy fallback: search any section containing the keyword
    for section in sections:
        if search in section.lower():
            return section.strip()

    return f"Section '{section_name}' not found. Available sections: {[re.match(r'^## (.*)', s).group(1) for s in sections if re.match(r'^## ', s)]}"


# ── Tool: List Runs ─────────────────────────────────────────────────────

@mcp.tool
def list_recent_runs(limit: int = 10) -> str:
    """List the most recent QA test runs with summary info.

    Returns run_id, date, pass_rate, latency_ratio, and session_duration.
    """
    sql = f"""
    SELECT run_id, run_date, passed, failed, pass_rate,
           session_duration_s, latency_ratio, report_file
    FROM runs
    ORDER BY run_date DESC
    LIMIT {min(limit, 50)}
    """
    try:
        conn = _get_db_connection()
        cursor = conn.execute(sql)
        result = _rows_to_markdown(cursor)
        conn.close()
        return result
    except Exception as e:
        return f"Error: {e}"


# ── Tool: Get Slow Turns ────────────────────────────────────────────────

@mcp.tool
def get_slow_turns(threshold_ms: int = 30000, run_id: str = None) -> str:
    """Find conversation turns that exceeded a latency threshold.

    Args:
        threshold_ms: Minimum latency in milliseconds (default: 30000 = 30s)
        run_id: Specific run to check. If omitted, uses the latest run.
    """
    run_filter = f"run_id = '{run_id}'" if run_id else "run_id = (SELECT run_id FROM runs ORDER BY run_date DESC LIMIT 1)"
    sql = f"""
    SELECT turn_number, turn_label, actual_ms, golden_ms,
           ROUND(CAST(actual_ms AS REAL) / NULLIF(golden_ms, 0), 2) as ratio
    FROM turn_metrics
    WHERE {run_filter} AND actual_ms > {threshold_ms}
    ORDER BY actual_ms DESC
    """
    try:
        conn = _get_db_connection()
        cursor = conn.execute(sql)
        result = _rows_to_markdown(cursor)
        conn.close()
        return result
    except Exception as e:
        return f"Error: {e}"


# ── Tool: Get Hallucination Details ─────────────────────────────────────

@mcp.tool
def get_hallucination_details(run_id: str = None) -> str:
    """Get per-turn hallucination scores and reasons.

    Args:
        run_id: Specific run to check. If omitted, uses the latest run.
    """
    run_filter = f"run_id = '{run_id}'" if run_id else "run_id = (SELECT run_id FROM runs ORDER BY run_date DESC LIMIT 1)"
    sql = f"""
    SELECT turn_number, turn_label, hallucination_score, hallucination_reason
    FROM turn_metrics
    WHERE {run_filter} AND hallucination_score IS NOT NULL
    ORDER BY hallucination_score DESC, turn_number
    """
    try:
        conn = _get_db_connection()
        cursor = conn.execute(sql)
        result = _rows_to_markdown(cursor)
        conn.close()
        return result
    except Exception as e:
        return f"Error: {e}"


# ── Tool: Compare Runs ──────────────────────────────────────────────────

@mcp.tool
def compare_runs(run_a: str, run_b: str) -> str:
    """Compare two test runs side by side.

    Shows differences in pass rates, latency ratios, and per-test scores.

    Args:
        run_a: First run_id (e.g., '20260222_200415')
        run_b: Second run_id (e.g., '20260223_114609')
    """
    # SQLite doesn't support FULL OUTER JOIN, use LEFT JOIN + UNION
    sql = f"""
    SELECT COALESCE(a.test_name, b.test_name) as test_name,
           a.outcome as outcome_a, a.score as score_a, a.duration_s as dur_a,
           b.outcome as outcome_b, b.score as score_b, b.duration_s as dur_b
    FROM (SELECT test_name, outcome, score, duration_s FROM test_scores WHERE run_id = '{run_a}') a
    LEFT JOIN (SELECT test_name, outcome, score, duration_s FROM test_scores WHERE run_id = '{run_b}') b
    ON a.test_name = b.test_name
    UNION
    SELECT COALESCE(a.test_name, b.test_name) as test_name,
           a.outcome as outcome_a, a.score as score_a, a.duration_s as dur_a,
           b.outcome as outcome_b, b.score as score_b, b.duration_s as dur_b
    FROM (SELECT test_name, outcome, score, duration_s FROM test_scores WHERE run_id = '{run_b}') b
    LEFT JOIN (SELECT test_name, outcome, score, duration_s FROM test_scores WHERE run_id = '{run_a}') a
    ON a.test_name = b.test_name
    WHERE a.test_name IS NULL
    ORDER BY test_name
    """
    try:
        conn = _get_db_connection()

        # Run-level comparison
        runs_sql = f"""
        SELECT run_id, run_date, pass_rate, latency_ratio, session_duration_s
        FROM runs
        WHERE run_id IN ('{run_a}', '{run_b}')
        ORDER BY run_date
        """
        runs_cursor = conn.execute(runs_sql)
        runs_result = _rows_to_markdown(runs_cursor)
        
        scores_cursor = conn.execute(sql)
        scores_result = _rows_to_markdown(scores_cursor)
        
        conn.close()

        result = "### Run-Level Comparison\n"
        result += runs_result
        result += "\n\n### Per-Test Comparison\n"
        result += scores_result
        return result
    except Exception as e:
        return f"Error: {e}"


# ── Tool: Get Failed Tests ──────────────────────────────────────────────

@mcp.tool
def get_failed_tests(run_id: str = None) -> str:
    """Get details about any failed tests.

    Args:
        run_id: Specific run to check. If omitted, searches all runs.
    """
    where = f"WHERE s.run_id = '{run_id}'" if run_id else ""
    and_clause = "AND" if run_id else "WHERE"
    sql = f"""
    SELECT s.run_id, r.run_date, s.test_name, s.method, s.score,
           s.failure_reason
    FROM test_scores s
    JOIN runs r ON s.run_id = r.run_id
    {where}
    {and_clause} s.outcome != 'passed'
    ORDER BY r.run_date DESC
    LIMIT 20
    """
    try:
        conn = _get_db_connection()
        cursor = conn.execute(sql)
        result = _rows_to_markdown(cursor)
        conn.close()
        return result
    except Exception as e:
        return f"Error: {e}"


# ── Tool: Get Tools Per Turn ────────────────────────────────────────────

@mcp.tool
def get_tools_per_turn(run_id: str = None) -> str:
    """Get the Pega tools (API calls) invoked per conversation turn.

    Args:
        run_id: Specific run to check. If omitted, uses the latest run.
    """
    run_filter = f"run_id = '{run_id}'" if run_id else "run_id = (SELECT run_id FROM runs ORDER BY run_date DESC LIMIT 1)"
    sql = f"""
    SELECT turn_number, turn_label, tools_called
    FROM turn_metrics
    WHERE {run_filter}
    ORDER BY turn_number
    """
    try:
        conn = _get_db_connection()
        cursor = conn.execute(sql)
        result = _rows_to_markdown(cursor)
        conn.close()
        return result
    except Exception as e:
        return f"Error: {e}"


# ── Main ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="QA Results MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http", "sse"],
        default="stdio",
        help="MCP transport type (default: stdio for Claude Desktop)",
    )
    args = parser.parse_args()

    port = int(os.environ.get("PORT", 8090))

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    elif args.transport == "http":
        mcp.run(transport="streamable-http", host="0.0.0.0", port=port, path="/")
    elif args.transport == "sse":
        mcp.run(transport="sse", host="0.0.0.0", port=port, path="/mcp")
