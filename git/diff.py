"""Pure unified-diff generation used by reports and tests."""

from difflib import unified_diff


def generate_diff(before, after, path="file"):
    lines = unified_diff(
        before.splitlines(keepends=True),
        after.splitlines(keepends=True),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
    )
    return "".join(lines)
