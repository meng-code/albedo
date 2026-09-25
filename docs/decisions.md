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

## 2026-09-25 — IDArb as the first albedo model to reproduce

- With the supervisor's agreement, the first model to get running is **IDArb** (Li et al., arXiv:2412.12083), which decomposes an image into albedo, normal, metallic and roughness. Scope agreed with the supervisor: run the official code, get results on the bundled examples and on our own images, and understand each step; matching the paper's metrics is not required.
- The official code is vendored unmodified at `third_party/IDArb/` (upstream commit `df26b90`); local changes are separate commits.
- **Where things run:** IDArb decomposition needs a GPU and runs on the cluster. Everything downstream (for example normal-to-mesh) runs on the student's own computer. Going forward the work uses single images only; the training data is not needed locally (supervisor, 2026-09-25).
- **Training smoke test** (supervisor request): train IDArb from Stable Diffusion 2.1 weights, not IDArb's released weights, on a few public objects to confirm the training code works. Result: 3 Arb-Objaverse objects, 300 steps on one A40 in 7m22s; validation normal cosine similarity rose from -0.19 to 0.79. This shows the model can fit the training objects, not that it generalises. Inferring at 512 px with weights trained at 256 px lowered normal cosine similarity from 0.93 to 0.77.
- **Upstream bug fixed:** `mv_dataset.py` `load_mr_abo` used `out_imgs` before creating it, so training on ABO or Arb-Objaverse crashed.
- **Normal-to-mesh** (supervisor suggestion: Poisson reconstruction): implemented both Poisson normal integration and screened Poisson surface reconstruction in `scripts/idarb/normal_to_mesh.py`. The supervisor observed that single-view meshes look flattened. Experiments traced most of this to depth discontinuities at occlusion boundaries, which normals cannot encode (true gaps of 49/99/149 px were recovered as about 5 px). Per the supervisor, meshes are now centred at the origin and can be stretched along Z (`--z_scale`).
- **Multi-view reconstruction** is not pursued; the supervisor considers novel-view-synthesis methods sufficient for that.
