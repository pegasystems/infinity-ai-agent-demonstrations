-- Insert a single test score row.
-- Parameters: run_id, test_name, method, outcome, score,
--             duration_s, failure_reason
INSERT INTO test_scores (
    run_id,
    test_name,
    method,
    outcome,
    score,
    duration_s,
    failure_reason
) VALUES (?, ?, ?, ?, ?, ?, ?);
