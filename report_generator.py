import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Import GeminiJudgeLLM from the local directory
# We need to add the current directory to sys.path
_this_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_this_dir))

try:
    from test_surface_agents import GeminiJudgeLLM
except ImportError:
    # Fallback if imports fail in some environments
    from google import genai
    from deepeval.models import DeepEvalBaseLLM

    class GeminiJudgeLLM(DeepEvalBaseLLM):
        def __init__(self, model_name: str = "gemini-2.5-flash"):
            self._model_name = model_name
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable is required")
            self._client = genai.Client(api_key=api_key)
        def load_model(self): return self._client
        def generate(self, prompt: str) -> str:
            return self._client.models.generate_content(model=self._model_name, contents=prompt).text
        async def a_generate(self, prompt: str) -> str:
            return self.generate(prompt)
        def get_model_name(self) -> str: return self._model_name

def generate_report(data_file: str = None, pytest_results_file: str = None):
    """Reads a DeepEval test run JSON and generates a Markdown report.

    Args:
        data_file: Path to the DeepEval test run JSON (LLM-judge tests only).
                   If None, falls back to .deepeval/.latest_test_run.json.
        pytest_results_file: Path to a JSON list of ALL pytest test outcomes
                   (all 8 tests).  When provided, this is prepended to the
                   prompt so the report reflects the full suite, not just the
                   3 DeepEval-registered tests.
    """
    # 1. Locate the test run JSON
    if data_file:
        json_path = Path(data_file)
    else:
        json_path = _this_dir / ".deepeval" / ".latest_test_run.json"

    if not json_path.exists():
        print(f"No test run data found at {json_path}")
        return

    # 2. Locate the report prompt template
    prompt_path = _this_dir / "report_prompt.md"
    if not prompt_path.exists():
        print(f"Prompt template not found at {prompt_path}")
        return

    with open(json_path, 'r') as f:
        test_data = f.read()

    # 2a. Load full pytest suite results if provided (all 8 tests, not just DeepEval's 3)
    pytest_summary = ""
    if pytest_results_file:
        try:
            with open(pytest_results_file, 'r') as f:
                pytest_results = json.load(f)
            passed = [r for r in pytest_results if r["outcome"] == "passed"]
            failed = [r for r in pytest_results if r["outcome"] in ("failed", "error")]
            lines = [
                f"## Full pytest suite results ({len(passed)}/{len(pytest_results)} passed)\n",
                "| Test | Outcome | Duration (s) |",
                "|---|---|---|",
            ]
            for r in pytest_results:
                icon = "✅" if r["outcome"] == "passed" else "❌"
                name = r["nodeid"].split("::")[-1]
                lines.append(f"| {name} | {icon} {r['outcome']} | {r['duration']} |")
            # Per-test diagnostic stdout (tool invocations, latency table,
            # step agents, hallucination per-turn scores, case key, etc.)
            lines.append("\n## Per-test diagnostic output\n")
            for r in pytest_results:
                name = r["nodeid"].split("::")[-1]
                if r.get("stdout"):
                    lines.append(f"### {name}\n```\n{r['stdout']}\n```\n")
                if r.get("longrepr"):
                    lines.append(f"### {name} — FAILURE DETAIL\n```\n{r['longrepr']}\n```\n")
            pytest_summary = "\n".join(lines) + "\n\n"
        except Exception as e:
            pytest_summary = f"(pytest results unavailable: {e})\n\n"

    # Extract the actual prompt from the markdown file
    # (The report_prompt.md contains instructions and a code block with the prompt)
    with open(prompt_path, 'r') as f:
        prompt_content = f.read()
    
    # Look for the prompt block
    if "```text" in prompt_content:
        system_prompt = prompt_content.split("```text")[1].split("```")[0].strip()
    else:
        system_prompt = prompt_content # Fallback

    full_prompt = (
        f"{system_prompt}\n\n"
        + (f"IMPORTANT — Full pytest suite results (use these for overall pass/fail counts and the scorecard):\n\n{pytest_summary}" if pytest_summary else "")
        + f"DeepEval LLM-judge detail (use for metric scores, failure analysis, conversation flow):\n\n{test_data}"
    )

    print("Generating QA report via Gemini...")
    judge = GeminiJudgeLLM()
    report_md = judge.generate(full_prompt)

    # 3. Save the report
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    human_ts = now.strftime("%B %d, %Y at %H:%M:%S")
    report_filename = f"QA_Report_{timestamp}.md"
    report_path = _this_dir / report_filename
    latest_path = _this_dir / "latest_qa_report.md"

    # Prepend a clear header so both files are self-identifying at a glance
    header = (
        f"# QA Report — {human_ts}\n"
        f"> **This is the latest report.** Archive copy: `{report_filename}`\n"
        f"> Open `latest_qa_report.md` for the most current run at any time.\n\n"
    )
    archived_header = (
        f"# QA Report — {human_ts}\n"
        f"> Archive copy. Current report is always in `latest_qa_report.md`.\n\n"
    )

    with open(latest_path, 'w') as f:
        f.write(header + report_md)

    with open(report_path, 'w') as f:
        f.write(archived_header + report_md)

    print(f"Report generated successfully: {report_path}")
    print(f"\n✅ LATEST REPORT: file://{latest_path.absolute()}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate a Markdown QA report from a DeepEval test run JSON.")
    parser.add_argument(
        "--data",
        default=None,
        help="Path to the DeepEval test run JSON.  Defaults to .deepeval/.latest_test_run.json."
    )
    parser.add_argument(
        "--pytest-results",
        default=None,
        dest="pytest_results",
        help="Path to a JSON file containing all pytest test outcomes (produced by conftest.py)."
    )
    parser.add_argument(
        "--replay-log",
        default=None,
        dest="replay_log",
        help=(
            "Path to a pytest console log (e.g. /tmp/deepeval_comprehensive_run.log). "
            "Parses per-test stdout from the log and uses it as --pytest-results. "
            "This lets you re-generate the report WITHOUT re-running the 15-min test suite."
        ),
    )
    args = parser.parse_args()

    pytest_results_file = args.pytest_results

    # --replay-log: parse a saved pytest log into a temp _pytest_results JSON
    if args.replay_log and not pytest_results_file:
        from _parse_log_to_results import parse_log
        import tempfile
        results = parse_log(args.replay_log)
        tf = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False,
            dir=str(_this_dir), prefix="_pytest_results_"
        )
        import json as _json
        _json.dump(results, tf, indent=2)
        tf.close()
        pytest_results_file = tf.name
        print(f"Parsed {len(results)} tests from {args.replay_log}")

    generate_report(data_file=args.data, pytest_results_file=pytest_results_file)
