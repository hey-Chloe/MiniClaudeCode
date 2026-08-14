# reports/

This directory is the single source of truth for measurable claims about
MiniClaudeCode. Any number quoted in the README, a resume, or a design review
must be traceable to a file here.

## Conventions

- `benchmark-<run-type>-<date>-<hash>.json` — content-addressed snapshot of
  one benchmark run (`validate` or `live`). Identical runs collapse to the
  same file name, so a changed number is visible in the artifact name.
- `latest-<name>.json` — convenience pointer to the most recent run of the
  same name (kept in sync with the versioned copy).
- The coding benchmark runner persists a versioned copy automatically on
  every invocation, including `--validate-only`.
- `skill-routing-*.json` — keyword vs hybrid skill-routing hit rates on the
  26-task benchmark (`python -m evaluation.skill_routing`).

## Comparing two runs

```powershell
python -m evaluation.reporting compare --left reports\A.json --right reports\B.json
python -m evaluation.reporting compare --left reports\A.json --right reports\B.json --markdown
```

Equivalent console script:

```powershell
miniclaude-report compare --left reports\A.json --right reports\B.json
```

## Storing an existing report

```powershell
miniclaude-report save --name benchmark-live --input my-report.json
```
