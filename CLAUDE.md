# CLAUDE.md

Guidance for Claude Code (and any other AI coding agent) working in this repository.

## 1. Project

**Textured mesh generation by single image decomposition.** From a single RGB image, decompose it into
albedo (the intrinsic colour of a surface, with lighting removed), normal, metallic and roughness, rebuild
a mesh of the visible surface from the normals, and colour it with the albedo.

- **Owner:** Peiran Yu (GitHub `meng-code`). **Supervisor:** Dr Juncheng Liu.
- **Current state:**
  - Intrinsic decomposition: IDArb (`third_party/IDArb/`) runs on the GPU server and splits an image into
    albedo, normal, metallic and roughness.
  - Geometry: `scripts/idarb/normal_to_mesh.py` rebuilds a mesh from an IDArb normal map by Poisson normal
    integration, optionally with a Depth Anything V2 depth prior (`scripts/idarb/estimate_depth.py`, run
    on the server) that restores the front-to-back structure normals alone lose. It runs on a laptop.
- **Direction:** single images, built on pretrained models. Multi-view reconstruction is out of scope,
  because novel-view-synthesis methods already cover it.
- Technical decisions are recorded with dates in `docs/decisions.md`.

## 2. Repository layout

```
scripts/idarb/             # input preparation, depth maps, normal-to-mesh, evaluation and figure scripts
tests/                     # pytest; tests/test_normal_integration.py must pass
third_party/IDArb/         # vendored IDArb: the first commit is unmodified upstream, later commits are ours
results/                   # small results and sample data: results/idarb_single/, results/analysis/
docs/decisions.md          # dated technical decisions
NOTICE.md                  # sources and licenses of third-party code, models and data
data/, weights/, outputs/  # git-ignored
```

## 3. Environments

| Environment | Where | Python | Used for | Install |
|---|---|---|---|---|
| `idarb` | GPU server (NVIDIA A40, CUDA 11.8) | 3.8 | IDArb inference and training, depth maps | `scripts/idarb/requirements-server.txt` |
| `mesh` | laptop | 3.10 | normal-to-mesh, evaluation scripts, pytest, ruff | `scripts/idarb/requirements-local.txt` |

## 4. Talking to the owner

- Talk to the owner in Chinese. Use plain, complete sentences, explain a technical term the first time it
  appears, and avoid metaphors and arrow shorthand. Keep answers to small questions short.
- The owner is still learning this codebase. When changing code, explain what changed and why in terms
  they could explain to their supervisor.

## 5. Git workflow

- Work on branches named with a short description in lowercase words joined by hyphens, for example
  `depth-prior-mesh`. Do not add a `claude/` or any other tool prefix. Never commit or push directly to
  `main`, and never force-push `main` or rewrite its history.
- The owner merges into `main` through a pull request. Do not approve, merge or close pull requests.
- Amend or force-push your own unmerged branch only when the owner asks.
- Keep each commit focused, with a message that says what changed and why.
- Commits use only the owner's git identity configured in this clone. Do not add a `Co-Authored-By`
  trailer or any other Claude attribution to commit messages or pull requests.
- Pull request description: **What / Why / How to test**, plus a few **Notes** bullets when a change needs
  explaining.
- **Before every push, run `pytest tests/ -q` and `ruff check .`** (in the `mesh` environment).

## 6. Engineering conventions

- Our code is in `scripts/idarb/` and `tests/`. ruff is configured in `pyproject.toml`.
- Scripts in `scripts/idarb/` also run in the `idarb` environment (Python 3.8), so they must not use newer
  syntax, for example `zip(strict=...)`, `match` statements, or `X | Y` type unions evaluated at run time.
- `third_party/` keeps the upstream style and Python version and is excluded from ruff. Change it only in
  separate, clearly described commits, so the diff against upstream stays readable.
- Chinese learning comments in the code stay as they are. Markdown documents, commit messages and pull
  requests are written in English.
- Experiments: put parameters on the command line, and record the command and the git commit together with
  the results (in the results file or in `docs/decisions.md`). Every number quoted in the report should be
  reproducible by a script in `scripts/idarb/`.
- Third-party code, model weights and datasets: record the source, version and license in `NOTICE.md`,
  and keep upstream license files.
- Large files stay out of git: datasets, model weights, checkpoints and generated meshes. Small sample
  images, preview images and CSV results are fine.

## 7. GPU server

The server is shared. Run jobs there only when the owner asks. Check `nvidia-smi` before starting, keep
runs as small as the task needs, and stop background processes when they are no longer needed.

## 8. Safety

Never:

- Push to `main` or rewrite the history of `main`.
- Delete branches, pull requests or files that the owner has not asked to remove.
- Run destructive commands outside the workspace (`rm -rf` on other paths, `sudo`, disk tools), or
  `git clean -xdf` without listing what it would delete first.
- Commit secrets: tokens, SSH keys, `.env` files or other credentials.
- Send repository contents, data or credentials to third-party services beyond what the task needs.
- Download more than about 2 GB, or add a heavy new dependency, without telling the owner first.

When the scope or risk of a task is unclear, ask the owner.
