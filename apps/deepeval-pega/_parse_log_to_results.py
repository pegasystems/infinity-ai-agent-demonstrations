#!/usr/bin/env python3
"""Parse a pytest log file to extract per-test stdout into a _pytest_results JSON.

Usage:
    python3 _parse_log_to_results.py /tmp/deepeval_comprehensive_run.log

This lets you iterate on report_generator.py and report_prompt.md
WITHOUT re-running the 15-minute test suite.
"""

import re
import json
import sys
from pathlib import Path


def parse_log(log_path: str) -> list:
    with open(log_path) as f:
        log = f.read()

    # Split by test markers: "test_golden_session.py::test_xxx PASSED/FAILED" or
    # "test_golden_session.py::test_xxx " at the start of a test run line.
    # Only match the COLLECTING/RUNNING lines (preceded by newline), not warning
    # paths like "test_golden_session.py::test_xxx\n  /Users/..."
    test_pattern = r"\ntest_golden_session\.py::(\w+)\s"
    matches = list(re.finditer(test_pattern, log))

    results = []
    for i, m in enumerate(matches):
        test_name = m.group(1)
        start = m.end()

        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            teardown = log.find("Running teardown", start)
            end = teardown if teardown > 0 else log.find("=" * 10, start)

        chunk = log[start:end].strip()

        if "PASSED" in chunk:
            outcome = "passed"
        elif "FAILED" in chunk:
            outcome = "failed"
        else:
            outcome = "unknown"

        # Try to find duration (if present in logs via METRIC_TEST_DURATION)
        duration_match = re.search(r"METRIC_TEST_DURATION: ([\d.]+)s", chunk)
        duration = float(duration_match.group(1)) if duration_match else 0.0

        # Clean the PASSED/FAILED marker
        stdout = re.sub(r"\s*PASSED\s*$", "", chunk).strip()
        stdout = re.sub(r"\s*FAILED\s*$", "", stdout).strip()

        results.append({
            "nodeid": f"test_golden_session.py::{test_name}",
            "outcome": outcome,
            "duration": duration,
            "longrepr": None,
            "stdout": stdout if stdout else None,
        })

    # De-duplicate: keep only 8 unique test names (first occurrence wins)
    seen = set()
    deduped = []
    for r in results:
        name = r["nodeid"]
        if name not in seen:
            seen.add(name)
            deduped.append(r)
    return deduped


if __name__ == "__main__":
    log_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/deepeval_comprehensive_run.log"
    results = parse_log(log_path)

    for r in results:
        preview = (r["stdout"] or "")[:100].replace("\n", " ")
        print(f"  {r['nodeid'].split('::')[1]:45s} stdout={len(r['stdout'] or ''):5d} chars | {preview}")

    out = Path(__file__).parent / "_pytest_results_from_log.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nWrote {out} with {len(results)} tests")
