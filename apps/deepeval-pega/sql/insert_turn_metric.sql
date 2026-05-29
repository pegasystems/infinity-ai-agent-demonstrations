-- Insert a single turn-level metric row.
-- Parameters: run_id, turn_number, turn_label, actual_ms, golden_ms,
--             hallucination_score, hallucination_reason, tools_called,
--             step_agents
INSERT INTO turn_metrics (
    run_id,
    turn_number,
    turn_label,
    actual_ms,
    golden_ms,
    hallucination_score,
    hallucination_reason,
    tools_called,
    step_agents
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
