"""Shared pytest configuration for evaluate-surface tests."""
import io
import os
import sys

import pytest

# Accumulates every test call result so the QA report covers ALL tests,
# not just the 3 that go through DeepEval's assert_test().
_pytest_test_results = []

# ── Tee-capture: records print() output even when -s is used ──────────
# pytest's report.sections only has stdout when capture is ON (no -s).
# Many users run with -s to watch progress live.  This tee approach
# writes to both the original stdout AND a per-test StringIO buffer,
# so the report always gets the diagnostic output.
_test_stdout_buffers = {}   # nodeid → StringIO (set in fixture, read in hook)


class _Tee:
    """Duplicate writes to both the real stdout and a StringIO buffer."""

    def __init__(self, original, buf):
        self._original = original
        self._buf = buf

    def write(self, data):
        self._original.write(data)
        self._buf.write(data)
        return len(data)

    def flush(self):
        self._original.flush()

    @property
    def encoding(self):
        return getattr(self._original, "encoding", "utf-8")

    def isatty(self):
        return False

    def __getattr__(self, name):
        return getattr(self._original, name)


@pytest.fixture(autouse=True)
def _capture_test_stdout(request):
    """Tee stdout so diagnostic output is captured even with ``-s``."""
    buf = io.StringIO()
    _test_stdout_buffers[request.node.nodeid] = buf
    original = sys.stdout
    sys.stdout = _Tee(original, buf)
    yield
    sys.stdout = original


def pytest_runtest_logreport(report):
    """Collect the outcome of every test call for the full-suite report.

    Reads diagnostic stdout from our tee buffer (works with or without -s),
    falling back to pytest's built-in report.sections (works without -s only).
    """
    if report.when == "call":  # skip setup / teardown phases
        # Primary: our tee buffer (always populated)
        buf = _test_stdout_buffers.get(report.nodeid)
        stdout_text = buf.getvalue().strip() if buf else None

        # Fallback: pytest's own capture (only populated when -s is NOT used)
        if not stdout_text:
            stdout_parts = []
            for header, content in (report.sections or []):
                if "stdout" in header.lower():
                    stdout_parts.append(content)
            stdout_text = "\n".join(stdout_parts).strip() or None

        # Print metric duration to stdout so log parsers can pick it up
        # This is CRITICAL for report generation from log files.
        print(f"\nMETRIC_TEST_DURATION: {round(report.duration, 3)}s")

        _pytest_test_results.append({
            "nodeid": report.nodeid,
            "outcome": report.outcome,   # "passed", "failed", "error"
            "duration": round(report.duration, 3),
            "longrepr": str(report.longrepr) if report.failed else None,
            "stdout": stdout_text,
        })


def pytest_addoption(parser):
    """Add --golden and --transport CLI options.

    Both options also fall back to environment variables so they can be set
    when using ``deepeval test run`` (which does not forward arbitrary pytest
    flags):

        GOLDEN_FILE=golden_sessions/golden_Zelle_Campaign_...json \\
        TRANSPORT=agentx \\
        deepeval test run test_golden_session.py -v -d all
    """
    parser.addoption(
        "--golden",
        action="store",
        default=os.environ.get("GOLDEN_FILE"),
        help=(
            "Path to a golden session JSON file. "
            "Falls back to the GOLDEN_FILE environment variable."
        ),
    )
    parser.addoption(
        "--transport",
        action="store",
        default=os.environ.get("TRANSPORT", "agentx"),
        choices=["agentx", "a2a", "auto"],
        help=(
            "Which API transport to use for replaying golden sessions. "
            "'agentx' (default) uses the Pega Application v2 API — same path "
            "as the UI, supports file uploads. "
            "'a2a' uses the A2A JSON-RPC protocol — tests the external interop "
            "contract but cannot attach files. "
            "'auto' uses AgentX for turns with file attachments and A2A for "
            "everything else."
        ),
    )
    parser.addoption(
        "--project-config",
        action="store",
        default=os.environ.get("PROJECT_CONFIG"),
        help=(
            "Path to a project_config.*.json file that describes the agent, "
            "workflow, and evaluation context.  Falls back to the PROJECT_CONFIG "
            "environment variable, then project_config.json in the test directory."
        ),
    )
    parser.addoption(
        "--tool-detection-mode",
        action="store",
        default=os.environ.get("TOOL_DETECTION_MODE", "hybrid"),
        choices=["hybrid", "strict_structured", "regex_only"],
        help=(
            "Tool detection policy for test_tool_invocations_match_golden. "
            "'hybrid' (default): structured tool_calls preferred; regex fallback "
            "produces a warning, not a hard fail. "
            "'strict_structured': fail if no structured tool data available for "
            "a session captured with detection_mode=structured. "
            "'regex_only': legacy behavior, always uses regex-detected names. "
            "Falls back to the TOOL_DETECTION_MODE environment variable."
        ),
    )
    parser.addoption(
        "--metrics",
        action="store",
        default=os.environ.get("EVAL_METRICS", ""),
        help=(
            "Comma-separated list of metric IDs to run. When provided, tests "
            "for optional Pega-specific metrics (pega_tool_correctness, "
            "business_case_lifecycle) are skipped unless explicitly included. "
            "Falls back to the EVAL_METRICS environment variable."
        ),
    )
    parser.addoption(
        "--case-id",
        action="store",
        default=os.environ.get("PEGA_CASE_ID"),
        help=(
            "Pega case ID for step agent evaluation (e.g. 'UPLUS-FS-WORK P-168004'). "
            "Required when the agent is a step agent operating on an existing case. "
            "Falls back to the PEGA_CASE_ID environment variable."
        ),
    )


def pytest_sessionfinish(session, exitstatus):
    """After the test session finishes, generate a Markdown report from live results.

    Strategy (in priority order):
      1. Grab DeepEval's in-memory TestRun via global_test_run_manager — this is
         populated by assert_test() calls regardless of whether the runner is
         ``python3 -m pytest`` or ``deepeval test run``.  Serialize it to a temp
         file and pass it to report_generator.py via --data.
      2. Fall back to .deepeval/.latest_test_run.json (written only by
         ``deepeval test run``) if in-memory data is unavailable / empty.
      3. Skip report generation quietly if neither source has data.

    This means BOTH run modes produce a valid, up-to-date report without
    requiring the user to switch to ``deepeval test run``.
    """
    import subprocess
    import sys
    import json
    import tempfile
    from pathlib import Path

    _this_dir = Path(__file__).resolve().parent
    report_gen_script = _this_dir / "report_generator.py"

    if not report_gen_script.exists():
        print(f"\n\x1b[1;33m[DeepEval Report Generator Skip]\x1b[0m Script not found at {report_gen_script}")
        return

    data_file = None

    # ── Strategy 1: in-memory DeepEval results (works with both pytest + deepeval test run) ──
    try:
        from deepeval.test_run import global_test_run_manager  # type: ignore
        test_run = global_test_run_manager.get_test_run()
        if test_run is not None:
            # Serialize pydantic model (v1 .json() or v2 .model_dump_json())
            try:
                serialized = test_run.model_dump_json()  # pydantic v2
            except AttributeError:
                serialized = test_run.json()  # pydantic v1

            # Only use in-memory data if it actually has test cases
            parsed = json.loads(serialized)
            if parsed.get("testCases") or parsed.get("test_cases"):
                results_dir = _this_dir / "test_results"
                results_dir.mkdir(exist_ok=True)
                tf = tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False,
                    dir=str(results_dir), prefix="_live_run_"
                )
                tf.write(serialized)
                tf.close()
                data_file = tf.name
                print("\n\x1b[1;36m[DeepEval Report Generator]\x1b[0m Using live in-memory results.")
    except Exception as e:
        print(f"\n\x1b[1;33m[DeepEval Report Generator]\x1b[0m Could not read in-memory results ({e}), trying fallback.")

    # ── Always write the full pytest suite results (all 8 tests) ──
    pytest_results_file = None
    if _pytest_test_results:
        try:
            results_dir = _this_dir / "test_results"
            results_dir.mkdir(exist_ok=True)
            prf = tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False,
                dir=str(results_dir), prefix="_pytest_results_"
            )
            json.dump(_pytest_test_results, prf)
            prf.close()
            pytest_results_file = prf.name
        except Exception as e:
            print(f"\n\x1b[1;33m[DeepEval Report Generator]\x1b[0m Could not write pytest results ({e})")

    # ── Strategy 2: fall back to .latest_test_run.json ──
    if data_file is None:
        fallback = _this_dir / ".deepeval" / ".latest_test_run.json"
        if fallback.exists():
            data_file = str(fallback)
            print("\n\x1b[1;33m[DeepEval Report Generator]\x1b[0m WARNING: using stale .latest_test_run.json — run via 'deepeval test run' for a fresh report.")
        else:
            print("\n\x1b[1;33m[DeepEval Report Generator Skip]\x1b[0m No test data available.")
            return

    print("\n\x1b[1;36m[DeepEval Report Generator]\x1b[0m Triggering final report...")
    try:
        cmd = [sys.executable, str(report_gen_script), "--data", data_file]
        if pytest_results_file:
            cmd += ["--pytest-results", pytest_results_file]
        subprocess.run(cmd, check=False)
    except Exception as e:
        print(f"\n\x1b[1;31m[DeepEval Report Generator Error]\x1b[0m {e}")

    # ── Database insert: push structured results for agent queries ──
    if _pytest_test_results:
        try:
            from insert_results import insert_results
            print("\n\x1b[1;36m[DB Insert]\x1b[0m Pushing results to SQLite...")
            insert_results(_pytest_test_results)
        except ImportError:
            # Fallback to legacy db_etl if insert_results not available
            try:
                from db_etl import push_to_database
                print("\n\x1b[1;36m[DB ETL]\x1b[0m Pushing results to SQLite...")
                push_to_database(_pytest_test_results)
            except ImportError:
                print("\n\x1b[1;33m[DB Insert Skip]\x1b[0m insert_results.py not found.")
        except Exception as e:
            # Non-fatal: don't block the test run for DB failures
            print(f"\n\x1b[1;33m[DB Insert Warning]\x1b[0m {e}")

    # ── Cleanup: keep pytest results JSON, remove only transient live-run files ──
    for tmp in [data_file]:
        if tmp and "_live_run_" in tmp:
            try:
                Path(tmp).unlink(missing_ok=True)
            except Exception:
                pass
    if pytest_results_file:
        print(f"\n\x1b[1;36m[Results]\x1b[0m Pytest results JSON kept at: {pytest_results_file}")
