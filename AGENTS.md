# AGENTS.md

Single source of truth for all agents (Claude Code, Codex) working in this repository.
Chat-level assistants (Claude chat, Codex chat) handle direction discussion and decisions with the human; agents in this repo handle implementation, execution, and cross-review.

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

- **Human (owner):** all decisions, all merges to `main`, launches all cluster jobs, makes all course submissions.
- **Claude Code / Codex (this repo):** implement via branches + Draft PRs, write tests, cross-review each other, output review conclusions only.
- Agents never approve, merge, or close PRs, and never act as the human on GitHub.

## 3. Repository layout

```
pyproject.toml          # single Python package definition
src/relight/            # package code (pipeline, shading, eval, demo)
tests/                  # pytest; smoke tests must always pass
scripts/                # dataset/weight download, cluster job templates
configs/                # experiment configs (yaml); seeds fixed
docs/diary.md           # weekly diary (course requirement; human-owned)
docs/decisions.md       # dated log of technical decisions
results/                # small metric JSONs + preview images only
data/, weights/         # gitignored; populated via scripts/
```

- No large binaries in git: datasets, model weights, images > 1 MB are gitignored and fetched by scripts.
- `docs/diary.md` is written by the human; agents may propose entries via PR but never rewrite existing ones.

## 4. Collaboration protocol

- `main` is protected. **Only the human merges.**
- Branch naming: `claude/<issue#>-<slug>` and `codex/<issue#>-<slug>`. Agents work only on their own branches.
- Every change goes through a **Draft PR** so the other agent can review before merge. Every PR links an Issue.
- Tasks live in **GitHub Issues**. No per-agent log files.
- PR description template: **What / Why / How to test / Learning notes** — Learning notes are 3–5 bullets explaining the change so the student can understand and defend every line in supervision meetings and the oral presentation.

**Cross-review risk tiers:**
- **Low** (docs, tests, comments, configs): one review pass, conclusions as PR comment.
- **Medium** (any change under `src/`): reviewer must run the smoke tests and state the result.
- **High** (dependency changes, anything under `scripts/` that downloads/deletes/moves data, cluster job templates, eval-metric code): review + explicit human approval in the PR thread before merge.

## 5. Engineering conventions

- Python 3.11+, PyTorch. Style: ruff. Tests: pytest; `tests/test_smoke.py` must pass on every PR.
- Reproducibility: every experiment reads a config from `configs/` with a fixed seed; metrics written to `results/` as JSON with the config name and git hash.
- Third-party code or weights: keep the license file, record source + license in `NOTICE.md`, and mark adapted blocks with a source comment. This feeds the report's attribution section — never strip provenance.

## 6. Course compliance (binding on all agents)

- This is graded university work. AI use is **unrestricted for this course provided full records are kept** (confirmed by the human with the teaching team — the human records the date/source of that confirmation in `docs/decisions.md`).
- The binding requirement is therefore **record-keeping**: the audit trail (PRs, Issues, commit history, exported chat logs) doubles as the **AI-use appendix** required by Massey's Generative AI in Assessment Policy. Never squash, rewrite, or delete history that documents who did what.
- Submission documents (proposal, progress report, final report, presentation) may be AI-assisted, but the human reviews, edits, and owns every final version, and must be able to explain all of it in supervision meetings and the oral presentation.
- All datasets must be public research datasets used within their license terms; record each in `NOTICE.md`. No scraping, no personal photos without clear rights.

## 7. Safety boundaries

Never, under any circumstances:
- Push to `main`, force-push, or rewrite published history.
- Approve/merge/close PRs, or delete branches other than your own.
- Run destructive commands (`rm -rf` outside the workspace, `git clean -xdf` without listing first, `sudo`, disk/partition tools).
- Commit secrets (tokens, SSH keys, `.env`, cluster credentials) — these stay gitignored.
- Submit or launch jobs on the Massey cluster (shared NVIDIA A40 nodes). Agents may write and update job scripts under `scripts/`, but only the human connects to the cluster and runs them.
- Upload repository contents, data, or credentials to any third-party service.
- Auto-download files > 2 GB or add a new dataset/model dependency without an Issue and human sign-off (High tier).

When uncertain about scope, risk, or course rules: **stop and open an Issue for the human.**
