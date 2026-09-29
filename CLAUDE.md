@AGENTS.md

## Claude Code notes

- Talk to the owner in Chinese. Use plain, complete sentences, explain a technical term the first time it
  appears, and avoid metaphors and arrow shorthand. Keep answers to small questions short.
- The owner is still learning this codebase. When changing code, explain what changed and why in terms
  they could explain to their supervisor.
- Before every push, run `pytest tests/ -q` and `ruff check .`.
- Commits use only the owner's git identity configured in this clone. Do not add a `Co-Authored-By`
  trailer or any other Claude attribution to commit messages or pull requests.
