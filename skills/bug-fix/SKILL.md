---
name: bug-fix
description: Locate, patch, and verify failing code by reading tests first and re-verifying after every change.
when_to_use: failing test, test failure, bug, broken, error, traceback, exception, wrong result, pytest, fix, crash, regression, implement, feature, add function, make all tests pass, syntax error, missing import, dependency, NameError
version: 1.0.0
tools: list_directory, glob_files, read_file, grep_files, execute_command, replace_text, write_file, workspace_diff
---

# Bug Fix Workflow

Follow this order for any bug-fix task:

1. **Read the tests first.** Use `list_directory`, `glob_files`, and `read_file`
   to find the test files that exercise the failing behavior. The tests define
   the expected contract; do not change them.
2. **Reproduce before editing.** Run the failing tests with `execute_command`
   (e.g. `pytest -q tests/...`) so you can see the actual error. Never fix a bug
   you have not observed failing.
3. **Locate the minimal cause.** Use `grep_files` to find the suspicious
   symbol or branch, then `read_file` the surrounding function. Prefer the
   smallest change that makes the contract hold.
4. **Edit with exact replacement.** Use `replace_text` for a unique, exact text
   block. If the block appears more than once, narrow it with surrounding
   context. Avoid rewriting unrelated code.
5. **Verify after every change.** Re-run the failing test first, then the full
   suite. Success means exit code 0 from the test command.
6. **Report honestly.** Summarize root cause, the exact change, and the test
   results. Only claim the bug is fixed when the verification command passed.

Pitfalls:

- Do not edit tests to make them pass.
- Do not silence errors with broad `except` clauses unless the task asks for it.
- If a file path is outside the workspace, report that it cannot be accessed
  instead of trying to bypass the boundary.
