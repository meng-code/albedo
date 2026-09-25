@AGENTS.md

## Claude Code notes

- Talk to the owner in Chinese. Use plain, complete sentences, explain a technical term the first time it
  appears, and avoid metaphors and arrow shorthand. Keep answers to small questions short.
- The owner is still learning this codebase. When changing code, explain what changed and why in terms
  they could explain to their supervisor.
- Before every push, run `pytest tests/ -q` and `ruff check .`.
- Commits use the git identity configured in this clone (the owner's name) and end with Claude Code's
  standard `Co-Authored-By` trailer.
