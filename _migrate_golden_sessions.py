"""Migrate golden session JSON files to add detection_mode annotation.

Classifies each session based on whether any turn has structured tool_calls
data (call_id present in insight.tool_details), then writes back the result.

Usage:
    python3 _migrate_golden_sessions.py [--dir golden_sessions] [--dry-run]

Expected outcomes for existing sessions:
  - Sessions already having detection_mode: skipped (idempotent).
  - Sessions with insight.tool_details containing call_id: detection_mode="structured"
  - All other sessions: detection_mode="regex"
"""

import argparse
import json
import sys
from pathlib import Path


def _classify_session(data: dict) -> str:
    """Return 'structured' if any turn has insight tool_details with a call_id."""
    for turn in data.get("turns", []):
        insight = turn.get("insight", {})
        tool_details = insight.get("tool_details", [])
        if isinstance(tool_details, list):
            for td in tool_details:
                if isinstance(td, dict) and td.get("call_id"):
                    return "structured"
    return "regex"


def _validate_expected_tools(data: dict, detection_mode: str) -> list[str]:
    """Check that expected_tools align with insight.tool_details for structured sessions.

    Returns a list of warning strings (empty = all good).
    """
    if detection_mode != "structured":
        return []
    warnings = []
    for turn in data.get("turns", []):
        expected = set(turn.get("expected_tools", []))
        if not expected:
            continue
        insight = turn.get("insight", {})
        detail_names = {
            td.get("tool_name", "")
            for td in insight.get("tool_details", [])
            if isinstance(td, dict)
        }
        missing = expected - detail_names
        extra = detail_names - expected
        t = turn.get("turn", "?")
        if missing:
            warnings.append(f"  Turn {t}: expected_tools has {missing} not in insight.tool_details")
        if extra:
            warnings.append(f"  Turn {t}: insight.tool_details has {extra} not in expected_tools")
    return warnings


def migrate_file(path: Path, dry_run: bool = False) -> bool:
    """Migrate a single golden session file. Returns True if a change was made."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  ERROR reading {path.name}: {e}")
        return False

    if "detection_mode" in data:
        print(f"  SKIP  {path.name}  (already has detection_mode={data['detection_mode']!r})")
        return False

    mode = _classify_session(data)
    data["detection_mode"] = mode
    data["_migration_note"] = (
        f"detection_mode='{mode}' injected by _migrate_golden_sessions.py"
    )

    warnings = _validate_expected_tools(data, mode)
    change_symbol = "DRY" if dry_run else "WRITE"
    print(f"  {change_symbol}  {path.name}  →  detection_mode={mode!r}")
    for w in warnings:
        print(f"    WARNING: {w}")

    if not dry_run:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", default="golden_sessions", help="Directory containing golden session JSON files")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change without writing files")
    args = parser.parse_args()

    sessions_dir = Path(args.dir)
    if not sessions_dir.is_dir():
        print(f"ERROR: Directory not found: {sessions_dir}", file=sys.stderr)
        sys.exit(1)

    files = sorted(sessions_dir.glob("golden_*.json"))
    if not files:
        print(f"No golden_*.json files found in {sessions_dir}")
        sys.exit(0)

    print(f"{'DRY RUN — ' if args.dry_run else ''}Migrating {len(files)} golden session(s) in {sessions_dir}/\n")

    changed = 0
    for f in files:
        if migrate_file(f, dry_run=args.dry_run):
            changed += 1

    print(f"\nDone. {'Would modify' if args.dry_run else 'Modified'} {changed}/{len(files)} file(s).")


if __name__ == "__main__":
    main()
