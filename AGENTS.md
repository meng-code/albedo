# AGENTS.md

Guidance for AI coding agents working in this repository. `CLAUDE.md` adds notes specific to Claude Code.

## 1. Project

**Textured mesh generation by single image decomposition.** From a single RGB image, decompose it into
albedo (the intrinsic colour of a surface, with lighting removed), normal, metallic and roughness, rebuild
a mesh of the visible surface from the normals, and colour it with the albedo. The repository and the
Python package keep their original names, `albedo-relight` and `relight`.

- **Owner:** Peiran Yu (GitHub `meng-code`). **Supervisor:** Dr Juncheng Liu.
- **Current state:**
  - Intrinsic decomposition: IDArb (`third_party/IDArb/`) is the working baseline. It runs on the GPU
    cluster and splits an image into albedo, normal, metallic and roughness.
  - Geometry: `scripts/idarb/normal_to_mesh.py` reconstructs a mesh from an IDArb normal map. It runs on a
    laptop. An optional depth prior (Depth Anything V2 depth maps from `scripts/idarb/estimate_depth.py`,
    produced on the server) restores front-to-back structure that normals alone lose.
  - Relighting: dropped from the project with the supervisor's agreement (October 2026). `src/relight/`
    is a placeholder package left from the original scope; its smoke pipeline treats the input as albedo
    and multiplies it by a constant light, so that the package and tests exist.
- **Direction:** work on single images and build on pretrained models. Multi-view reconstruction is out of
  scope, because novel-view-synthesis methods already cover it.
- Technical decisions are recorded with dates in `docs/decisions.md`.

## 2. Repository layout

```
src/relight/               # our package (placeholder left from the original relighting scope)
tests/                     # pytest; tests/test_smoke.py must pass
scripts/idarb/             # IDArb helpers: input preparation, depth maps, normal-to-mesh, figures
third_party/IDArb/         # vendored IDArb: the first commit is unmodified upstream, later commits are ours
configs/                   # experiment configs for our own code
results/                   # small results and sample data, e.g. results/idarb_single/, results/analysis/
docs/decisions.md          # dated technical decisions
docs/diary.md              # progress log; the owner writes the entries
NOTICE.md                  # sources and licenses of third-party code, models and data
data/, weights/, outputs/  # git-ignored
```

## 3. Environments

| Environment | Where | Python | Used for | Install |
|---|---|---|---|---|
| `relight` | anywhere | 3.11+ | our package, tests, ruff | `pip install -e ".[dev]"` |
| `idarb` | GPU cluster (NVIDIA A40, CUDA 11.8) | 3.8 | IDArb inference and training | `scripts/idarb/requirements-server.txt` |
| `mesh` | laptop | 3.10 | `scripts/idarb/normal_to_mesh.py` | `scripts/idarb/requirements-local.txt` |

## 4. Git workflow

- Agents work on branches named with a short description in lowercase words joined by hyphens, for example
  `depth-prior-mesh`. Do not add a `claude/` or any other tool prefix. Never commit or push directly to
  `main`, and never force-push `main` or rewrite its history.
- The owner merges into `main` through a pull request. Agents do not approve, merge or close pull requests.
- An agent may amend or force-push its own unmerged branch only when the owner asks.
- Keep each commit focused, with a message that says what changed and why.
- Pull request description: **What / Why / How to test**, plus a few **Notes** bullets when a change needs
  explaining.
- GitHub Issues are optional; use them for tasks that span several sessions.

## 5. Engineering conventions

- Our code: Python 3.11+, ruff for style, pytest for tests. Run `pytest tests/ -q` and `ruff check .`
  before pushing.
- `third_party/` keeps the upstream style and Python version and is excluded from ruff. Change it only in
  separate, clearly described commits, so the diff against upstream stays readable.
- Scripts that also run in the `idarb` environment (Python 3.8), such as those in `scripts/idarb/`, must
  not use newer syntax, for example `zip(strict=...)`, `match` statements, or `X | Y` type unions
  evaluated at run time.
- Experiments: keep settings in config files, fix random seeds, and save metrics together with the config
  name and git commit so they can be reproduced.
- Third-party code, model weights and datasets: record the source, version and license in `NOTICE.md`,
  and keep upstream license files.
- Large files stay out of git: datasets, model weights, checkpoints and generated meshes. Small sample
  images and preview images are fine.

## 6. GPU cluster

The cluster is shared. Agents may run jobs there when the owner asks. Check `nvidia-smi` before starting,
keep runs as small as the task needs, and stop background processes when they are no longer needed.

## 7. Safety

Never:

- Push to `main` or rewrite the history of `main`.
- Delete branches, pull requests or files that the owner has not asked to remove.
- Run destructive commands outside the workspace (`rm -rf` on other paths, `sudo`, disk tools), or
  `git clean -xdf` without listing what it would delete first.
- Commit secrets: tokens, SSH keys, `.env` files or other credentials.
- Send repository contents, data or credentials to third-party services beyond what the task needs.
- Download more than about 2 GB, or add a heavy new dependency, without telling the owner first.

When the scope or risk of a task is unclear, ask the owner.
