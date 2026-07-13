# AGENTS.md

## Project
Multilingual jailbreaking in agentic LLM settings — a defensive security
(red-team) research project. Purpose: measure and understand a vulnerability,
NOT to build usable attack tools.

## Who reads this file
- Codex reads this file natively.
- Claude Code reads it via CLAUDE.md (which points here).
Both agents must read this before starting any task.

## Collaboration protocol
- Treat `main` as protected. NEVER commit or push directly to `main`.
- Only the human may approve and merge changes into `main`.
- Each agent works on its own branch:
  - Claude Code: `claude/<short-task-name>`
  - Codex: `codex/<short-task-name>`
- All changes are submitted as **Draft PRs**. A branch + Draft PR is enough
  for the other agent to read and review — do NOT wait for a merge to inspect
  the other side's work.
- **Only the human (小萌) merges.** Agents never merge.

## Cross-review
- Cross-review is selective by default.
- Cross-review is required for changes involving:
  - safety boundaries or threat models;
  - dataset ingestion or sensitive outputs;
  - model-provider calls, prompts, or agent behavior;
  - scoring, evaluation methodology, or reported metrics;
  - dependencies, secrets, CI, or release configuration.
- Documentation-only, formatting, and low-risk mechanical changes may skip
  cross-review when the human approves.

## Tasks
- Tasks are tracked as **GitHub Issues** (not in-repo log files).
- Each Issue should state: goal, non-goals, inputs/outputs, allowed
  directories, acceptance criteria, safety constraints, dependencies.
- Task size guideline: one task = one result + one branch + one PR,
  ~30–120 minutes of work.

## Secrets and cost controls
- NEVER commit API keys, OAuth tokens, credentials, `.env` files, or secrets.
- Redact credentials and sensitive content from logs and error reports.
- Do not start paid or high-volume model runs without explicit human approval.
- Before an experiment, state the provider, model, sample count, and estimated
  number of requests.

## Safety boundaries (hard constraints — never violate)
- Defensive red-team only. Measure/understand; do not create usable exploits.
- Use only approved, clearly-licensed **public benchmark datasets**.
  Candidate datasets, including MultiJail, require human review of their
  license, provenance, and permitted use before ingestion.
- Do NOT develop automated novel-attack-optimization algorithms.
- Data-processing steps (translation, back-translation, encoding or template
  transforms) must be reviewed before use in real experiments to decide
  whether they count as authored attack content; this test run is unaffected.
- NEVER commit directly-reusable high-risk outputs, jailbreak payloads,
  or exploits to the repo.
- Raw model outputs live in a git-ignored local directory. The repo stores
  only: data IDs, hashes, aggregate metrics, and minimal redacted examples.
- Any sensitive result is reviewed by the human before it is made public.
- Follow responsible disclosure.
