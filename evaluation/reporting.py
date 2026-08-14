"""Versioned report storage and A/B comparison for benchmark artifacts.

Every benchmark run should be stored under ``reports/`` so any number quoted
in a resume or design review can be traced to a concrete file. This module
provides:

- ``save_report``: content-addressed, timestamped copies plus a ``latest-``
  pointer for the most recent run of the same name;
- ``compare`` / ``compare_markdown``: numeric deltas between two reports;
- a ``miniclaude-report`` CLI (``python -m evaluation.reporting``).
"""

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any


def _default_reports_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "reports"


def _sanitize_name(name: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in name.strip()
    ).strip("-")
    if not cleaned:
        raise ValueError("report name must not be empty")
    return cleaned


def _render(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n"


def save_report(
    name: str,
    payload: Any,
    reports_dir: str | Path | None = None,
    stamp: str | None = None,
) -> Path:
    """Persist a versioned copy plus a ``latest-<name>`` pointer.

    The file name includes a content hash, so identical runs collapse to the
    same artifact and any change is visible in the file name.
    """
    reports_dir = (
        Path(reports_dir)
        if reports_dir is not None
        else _default_reports_dir()
    )
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = stamp or date.today().isoformat()
    rendered = _render(payload)
    digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()[:10]
    target = reports_dir / f"{_sanitize_name(name)}-{stamp}-{digest}.json"
    target.write_text(rendered, encoding="utf-8")
    latest = reports_dir / f"latest-{_sanitize_name(name)}.json"
    latest.write_text(rendered, encoding="utf-8")
    return target


def load_report(path: str | Path) -> dict[str, Any]:
    """Load a JSON report file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _flatten_numeric(
    value: Any,
    prefix: str = "",
    out: dict[str, float | int] | None = None,
) -> dict[str, float | int]:
    """Flatten dicts into dotted paths, keeping only scalar numbers.

    Lists (e.g. per-case results) are intentionally skipped so aggregate
    comparisons stay readable.
    """
    if out is None:
        out = {}
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            _flatten_numeric(item, path, out)
    elif isinstance(value, bool):
        pass
    elif isinstance(value, (int, float)):
        out[prefix] = value
    return out


def compare(left: Any, right: Any) -> dict[str, Any]:
    """Compare two report payloads (dicts or paths) on common numeric fields."""
    left_payload = load_report(left) if isinstance(left, (str, Path)) else left
    right_payload = (
        load_report(right) if isinstance(right, (str, Path)) else right
    )
    left_flat = _flatten_numeric(left_payload)
    right_flat = _flatten_numeric(right_payload)
    common = sorted(set(left_flat) & set(right_flat))

    changes: dict[str, dict[str, Any]] = {}
    for path in common:
        left_value = left_flat[path]
        right_value = right_flat[path]
        entry: dict[str, Any] = {
            "left": left_value,
            "right": right_value,
            "delta": round(right_value - left_value, 6),
            "percent": (
                round((right_value - left_value) / left_value * 100, 2)
                if left_value
                else None
            ),
        }
        if left_value != right_value:
            changes[path] = entry

    return {
        "left": str(left) if isinstance(left, (str, Path)) else "<inline>",
        "right": str(right) if isinstance(right, (str, Path)) else "<inline>",
        "compared": common,
        "changes": changes,
    }


def compare_markdown(delta: dict[str, Any]) -> str:
    """Render a comparison as a Markdown table, most significant change first."""
    changes = delta.get("changes", {})
    rows = sorted(
        changes.items(),
        key=lambda item: abs(item[1]["percent"])
        if item[1]["percent"] is not None
        else 0.0,
        reverse=True,
    )
    lines = [
        "# Report comparison",
        "",
        f"- left: `{delta.get('left')}`",
        f"- right: `{delta.get('right')}`",
        f"- common numeric fields: {len(delta.get('compared', []))}",
        f"- changed fields: {len(changes)}",
        "",
        "| metric | left | right | delta | % change |",
        "|---|---|---|---|---|",
    ]
    for path, entry in rows:
        percent = (
            f"{entry['percent']:+.2f}%"
            if entry["percent"] is not None
            else "n/a"
        )
        lines.append(
            f"| {path} | {entry['left']} | {entry['right']} "
            f"| {entry['delta']:+.6g} | {percent} |"
        )
    if not rows:
        lines.append("_No numeric differences._")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="miniclaude-report",
        description="Store, load, and compare versioned benchmark reports",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    save_parser = subparsers.add_parser("save", help="store a JSON report")
    save_parser.add_argument("--name", required=True)
    save_parser.add_argument("--input", required=True)
    save_parser.add_argument("--reports-dir", default=None)

    compare_parser = subparsers.add_parser(
        "compare", help="diff two JSON reports"
    )
    compare_parser.add_argument("--left", required=True)
    compare_parser.add_argument("--right", required=True)
    compare_parser.add_argument("--markdown", action="store_true")
    compare_parser.add_argument("--output", default=None)

    args = parser.parse_args()
    if args.command == "save":
        payload = load_report(args.input)
        target = save_report(args.name, payload, args.reports_dir)
        print(target)
        return 0

    delta = compare(args.left, args.right)
    rendered = (
        compare_markdown(delta)
        if args.markdown
        else json.dumps(delta, ensure_ascii=False, indent=2)
    )
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
