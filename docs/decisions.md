# Technical decisions

## 2026-07-15 — Project reset

- The human changed the capstone topic to **AI-Based Image Albedo Estimation and Relighting** under the supervision of Dr Juncheng Liu.
- Published Git history will be preserved while the current repository content and package naming move from the obsolete LLM security project to `albedo-relight` / `relight`.
- The initial smoke pipeline is dependency-light plumbing only. It is not an albedo estimator or a scientific relighting baseline.
- Model, training, dataset, input-domain, relighting, and evaluation choices remain open pending literature and feasibility review.
- The human confirmed in Codex chat on 2026-07-15 that AI use is unrestricted for this course provided complete records are retained; the human owns the final submissions and audit record.

## 2026-09-25 — Agent rules revised

The human revised the agent rules in `AGENTS.md` and `CLAUDE.md`:

- Agent-to-agent cross-review (Codex) is retired.
- A GitHub Issue is recommended for larger tasks but no longer required for every change. Branches may be named `claude/<slug>`.
- Agents still never push to `main`, never force-push, and never rewrite published history. The human merges every PR.
- At the human's request, agents may author commits under the human's name. Every agent-made commit must carry a `Co-Authored-By: Claude ...` trailer so the AI-use record stays complete.
- Agents may launch cluster jobs when the human explicitly asks for them in the session.
- During the learning stage, small sample images whose rights have not been checked may be committed if the commit says so; they must be reviewed before any public release or submission.
- `third_party/` keeps upstream style and Python version and is excluded from ruff.

Audit note: before this revision, Claude Code had already launched IDArb inference and a training smoke test on the cluster (node hpc-jliu02) in a session where the human explicitly asked for them.
