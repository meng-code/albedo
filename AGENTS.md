# AGENTS.md

Single source of truth for AI agents (Claude Code) working in this repository.
Chat-level assistants handle direction discussion and decisions with the human; agents in this repo handle implementation and execution.
Rules last revised on 2026-09-25 at the human's request (see `docs/decisions.md`).

---

## 1. Project

**Name:** albedo-relight (working title)

**Context:** Individual capstone project for 159.333 Programming Project, Massey University, Semester 2 2026.
Supervisor: Dr Juncheng Liu (3D graphics / ML). Student & repo owner: the human.
This semester's 2D pipeline may provide groundwork for future research on relightable 3D scenes (3DGS/NeRF) with the same supervisor.

**Goal:** Given a single RGB image, estimate its albedo (intrinsic image decomposition) using an AI-based method, then generate realistic relit images under user-controllable lighting; evaluate quantitatively and ship an interactive demo.

**Confirmed scope for this semester:**
- Review recent literature on intrinsic image decomposition and image relighting.
- Implement or reproduce a state-of-the-art albedo estimation model.
- Build a relighting module around predicted albedo and controllable lighting parameters.
- Evaluate albedo estimation and relit images quantitatively and qualitatively.
- Explore one feasible improvement if the core pipeline is complete.

**Open research decisions:** input domain; model and training strategy; datasets; relighting formulation; evaluation metrics; and whether a user study is feasible. Record decisions in `docs/decisions.md`; model or dataset additions require an Issue and human sign-off.

**Non-goals (this semester):** 3D scene relighting; real-time mobile deployment; and committing to the possible 3DGS/NeRF extension before the 2D study is evaluated.

**Course deadlines the repo must serve (NZST):**
- Proposal: due Mon 27 Jul 2026 (10%)
- Progress report: mid-semester (10%, date TBC on Stream)
- Software repository: assessed continuously (10%) — commit history, Issues, PRs are graded evidence
- Video presentation ~10 min (20%) and final report (50%): end of semester

## 2. Roles

- **Human (owner):** all decisions, all merges to `main`, makes all course submissions. Launches cluster jobs, or explicitly asks an agent to launch them.
- **Claude Code (this repo):** implements on branches, writes tests, and may run jobs on the cluster when the human explicitly asks in the session.
- Agents never approve, merge, or close PRs.
- Commits: when the human asks, agents may author commits under the human's name, but **every agent-made commit must end with a `Co-Authored-By: Claude ...` trailer** so the AI-use record stays complete (see §6).

## 3. Repository layout

```
pyproject.toml          # single Python package definition
src/relight/            # package code (pipeline, shading, eval, demo)
tests/                  # pytest; smoke tests must always pass
scripts/                # dataset/weight download, cluster job templates
configs/                # experiment configs (yaml); seeds fixed
docs/diary.md           # weekly diary (course requirement; human-owned)
docs/decisions.md       # dated log of technical decisions
results/                # small metric JSONs, preview images, and small sample data
third_party/            # vendored third-party code (e.g. IDArb); keeps upstream license and style
data/, weights/         # gitignored; populated via scripts/
```

- No large binaries in git: datasets and model weights are gitignored and fetched by scripts. Small sample images (roughly up to a few MB each) are fine.
- `docs/diary.md` is written by the human; agents may propose entries via PR but never rewrite existing ones.

## 4. Collaboration protocol

- `main` is protected. **Only the human merges.** Agents push to branches, never to `main`.
- Branch naming: `claude/<slug>`, or `claude/<issue#>-<slug>` when there is an Issue. Agents work only on their own branches.
- Changes reach `main` through a PR that the human opens (or asks the agent to open) and merges.
- GitHub Issues are recommended for larger tasks but not required for every change.
- PR description template: **What / Why / How to test / Learning notes** — Learning notes are 3–5 bullets explaining the change so the student can understand and defend every line in supervision meetings and the oral presentation.

**Risk tiers** (agent-to-agent cross-review has been retired):
- **Low** (docs, tests, comments, configs, samples): the agent runs the checks in §5 before pushing.
- **Medium** (any change under `src/`): as Low, and the PR states the smoke-test result.
- **High** (dependency changes, anything under `scripts/` that downloads/deletes/moves data, cluster job templates, eval-metric code): the human explicitly approves before merge.

## 5. Engineering conventions

- Python 3.11+, PyTorch. Style: ruff. Tests: pytest; `tests/test_smoke.py` must pass on every PR.
- Exception: `third_party/` keeps the upstream Python version and style and is excluded from ruff. Scripts that must also run in a third-party environment (e.g. `scripts/idarb/`, Python 3.8) avoid syntax newer than that environment supports.
- Reproducibility: every experiment reads a config from `configs/` with a fixed seed; metrics written to `results/` as JSON with the config name and git hash.
- Third-party code or weights: keep the license file, record source + license in `NOTICE.md`, and mark adapted blocks with a source comment. This feeds the report's attribution section — never strip provenance.

## 6. Course compliance (binding on all agents)

- This is graded university work. AI use is **unrestricted for this course provided full records are kept** (confirmed by the human with the teaching team — the human records the date/source of that confirmation in `docs/decisions.md`).
- The binding requirement is therefore **record-keeping**: the audit trail (PRs, Issues, commit history, exported chat logs) doubles as the **AI-use appendix** required by Massey's Generative AI in Assessment Policy. Never squash, rewrite, or delete history that documents who did what.
- Submission documents (proposal, progress report, final report, presentation) may be AI-assisted, but the human reviews, edits, and owns every final version, and must be able to explain all of it in supervision meetings and the oral presentation.
- Training and evaluation datasets must be public research datasets used within their license terms; record each in `NOTICE.md`.
- During the learning stage, small sample images whose rights have not been checked may be committed for testing, as long as the commit says so. Review them before any public release or submission.

## 7. Safety boundaries

Never, under any circumstances:
- Push to `main`, force-push, or rewrite published history.
- Approve/merge/close PRs, or delete branches other than your own.
- Run destructive commands (`rm -rf` outside the workspace, `git clean -xdf` without listing first, `sudo`, disk/partition tools).
- Commit secrets (tokens, SSH keys, `.env`, cluster credentials) — these stay gitignored.
- Launch jobs on the Massey cluster (shared NVIDIA A40 nodes) unless the human explicitly asks for that job in the session.
- Upload repository contents, data, or credentials to any third-party service.
- Auto-download files > 2 GB or add a new dataset/model dependency without an Issue and human sign-off (High tier).

When uncertain about scope, risk, or course rules: **stop and open an Issue for the human.**
