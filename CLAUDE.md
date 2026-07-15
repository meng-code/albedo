@AGENTS.md

## Claude Code specifics
- Work only on `claude/<issue#>-<slug>` branches; open Draft PRs early.
- Run `pytest tests/ -q` and `ruff check .` before marking any PR ready for review.
- Review Codex's Draft PRs per the risk tiers in @AGENTS.md §4; post conclusions as PR comments only — never approve or merge.
- If a task requires anything listed in @AGENTS.md §7 (Safety boundaries), stop and open an Issue instead.
