-- Insert or replace a single run summary row.
-- Parameters: run_id, run_date, total_tests, passed, failed,
--             pass_rate, session_duration_s, latency_ratio,
--             golden_file, report_file
INSERT OR REPLACE INTO runs (
    run_id,
    run_date,
    total_tests,
    passed,
    failed,
    pass_rate,
    session_duration_s,
    latency_ratio,
    golden_file,
    report_file
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
