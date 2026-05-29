-- SQLite3 schema for QA results
-- Database: qa_results.db
--
-- Run setup:  python3 db_etl.py --init-db
-- Or manually: sqlite3 qa_results.db < sqlite_schema.sql

-- ─── runs: one row per test-suite execution ───────────────────────
CREATE TABLE IF NOT EXISTS runs (
  run_id              TEXT    PRIMARY KEY,   -- "20260223_112405"
  run_date            TEXT    NOT NULL,      -- ISO8601 timestamp
  total_tests         INTEGER,
  passed              INTEGER,
  failed              INTEGER,
  pass_rate           REAL,
  session_duration_s  REAL,
  latency_ratio       REAL,                  -- e.g. 1.16
  golden_file         TEXT,
  report_file         TEXT                   -- archive filename
);

-- ─── test_scores: one row per test per run ────────────────────────
CREATE TABLE IF NOT EXISTS test_scores (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id        TEXT    NOT NULL,
  test_name     TEXT    NOT NULL,            -- "test_knowledge_retention"
  method        TEXT,                        -- "LLM Judge" | "Logic"
  outcome       TEXT,                        -- "passed" | "failed"
  score         TEXT,                        -- "1.00", "0 fails", "1.16x"
  duration_s    REAL,
  failure_reason TEXT,
  FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

-- ─── turn_metrics: one row per conversation turn per run ──────────
CREATE TABLE IF NOT EXISTS turn_metrics (
  id                  INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id              TEXT    NOT NULL,
  turn_number         INTEGER NOT NULL,
  turn_label          TEXT,                  -- "Upload Zelle.pdf"
  actual_ms           INTEGER,
  golden_ms           INTEGER,
  hallucination_score REAL,
  hallucination_reason TEXT,
  tools_called        TEXT,                  -- JSON array: ["tool1", "tool2"]
  tools_source_summary TEXT,                 -- e.g. "tool_calls:3,regex:1"
  step_agents         TEXT,                  -- JSON array: ["agent1", "agent2"]
  FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

-- ─── Indexes for common queries ───────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_test_scores_run_id ON test_scores(run_id);
CREATE INDEX IF NOT EXISTS idx_turn_metrics_run_id ON turn_metrics(run_id);
CREATE INDEX IF NOT EXISTS idx_runs_run_date ON runs(run_date);
