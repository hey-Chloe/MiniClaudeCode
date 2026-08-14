---
name: code-review
description: Review a diff or codebase for correctness, safety, and clarity, and report concrete findings.
when_to_use: review, pull request, diff, code review, audit, merge, changes, refactor, rename, extract, behavior, safe change, delete, rm, secret, permission, security, approval, policy, confirm, edit
version: 1.0.0
tools: git_diff, git_status, grep_files, read_file, execute_command, workspace_diff
---

# Code Review Workflow

1. **Scope the review.** Use `git_diff` to see the changed lines, or
   `grep_files`/`read_file` to inspect the files under review.
2. **Read surrounding context.** Review findings need context: read the
   function and its callers before judging a change.
3. **Check in this order:**
   - Correctness: off-by-one, edge cases, error handling, race-prone state.
   - Security: path handling, shell-like argument use, unchecked input,
     secrets in code.
   - Clarity: naming, dead code, duplication that matters.
4. **Rank findings by severity.** Prefer a short list of concrete, actionable
   issues over an exhaustive list of style nits.
5. **Quote the code.** Each finding must reference file and line, with a
   suggested fix, so the author can act without re-deriving the context.

Pitfalls:

- Do not rewrite code during a review unless the task explicitly asks for it.
- If a change cannot be verified locally, say so instead of guessing.
