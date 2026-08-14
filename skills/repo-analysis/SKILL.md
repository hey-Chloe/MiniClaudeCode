---
name: repo-analysis
description: Explore an unfamiliar repository and explain its structure, responsibilities, and key flows.
when_to_use: understand, explore, explain, architecture, overview, summarize, onboarding, map, structure, locate, find, contains, defines, search, repository, codebase, file
version: 1.0.0
tools: list_directory, glob_files, read_file, grep_files
---

# Repository Analysis Workflow

1. **Map the top level.** Use `list_directory` from the workspace root, then
   follow interesting directories one level at a time.
2. **Find entry points and docs.** Look for README, package manifests
   (`pyproject.toml`, `requirements.txt`), and test layouts. These reveal the
   module boundaries.
3. **Trace one flow end to end.** Pick the most important entry point, then
   follow it with `grep_files` and `read_file` across modules. Understand one
   real flow before generalizing.
4. **Collect evidence.** Note file paths and line numbers for every claim.
   Prefer observable facts (symbol definitions, call sites, tests) over
   naming-based guesses.
5. **Deliver a structured summary.** Present: top-level layout, core modules
   and responsibilities, key flows, tests, and any obvious risks.

Pitfalls:

- Do not dump every file; summarize patterns, not line counts.
- Respect the workspace boundary: report when information cannot be reached
  instead of attempting to escape it.
