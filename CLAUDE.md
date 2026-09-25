@AGENTS.md

## Claude Code specifics
- Work only on `claude/<slug>` (or `claude/<issue#>-<slug>`) branches; never push to `main`.
- Run `pytest tests/ -q` and `ruff check .` before every push.
- End every commit message with a `Co-Authored-By: Claude ...` trailer, including commits authored under the human's name.
- If a task requires anything listed in @AGENTS.md §7 (Safety boundaries), stop and open an Issue instead.
