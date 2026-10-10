# Technical decisions

## 2026-07-15 — Project reset

- The human changed the capstone topic to **AI-Based Image Albedo Estimation and Relighting** under the supervision of Dr Juncheng Liu.
- Published Git history will be preserved while the current repository content and package naming move from the obsolete LLM security project to `albedo-relight` / `relight`.
- The initial smoke pipeline is dependency-light plumbing only. It is not an albedo estimator or a scientific relighting baseline.
- Model, training, dataset, input-domain, relighting, and evaluation choices remain open pending literature and feasibility review.
- The human confirmed in Codex chat on 2026-07-15 that AI use is unrestricted for this course provided complete records are retained; the human owns the final submissions and audit record.

## 2026-09-25 — Repository set up for long-term use

The repository will be used beyond the original course, so the owner rewrote `AGENTS.md` and `CLAUDE.md` as general project guidance:

- Course-specific rules (deadlines, semester scope, AI-use record-keeping) were removed.
- Codex cross-review is retired. GitHub Issues are optional. Agents work on `claude/<short-description>` branches, never push to `main`, and the owner merges through pull requests.
- Agents may run jobs on the shared GPU cluster when the owner asks, and may commit under the owner's git identity.
- Obsolete branches `claude/6-agents-default-strategy` and `claude/7-validation-rejection-tests` were deleted; their commits remain reachable through pull requests #9 and #10.
- `third_party/` keeps upstream style and Python version and is excluded from ruff.
- **Update 2026-09-30:** at the owner's request, branch names no longer carry the `claude/` prefix; they are a short hyphenated description such as `depth-prior-mesh`, and `AGENTS.md` was updated. Existing branches were renamed on GitHub: `claude/depth-prior-mesh` to `depth-prior-mesh`, and the merged `claude/idarb-integration` to `idarb-integration`. Pull request #11 still shows the old branch name; its commits are unchanged and remain in `main`.

## 2026-09-25 — IDArb as the first albedo model to reproduce

- With the supervisor's agreement, the first model to get running is **IDArb** (Li et al., arXiv:2412.12083), which decomposes an image into albedo, normal, metallic and roughness. Scope agreed with the supervisor: run the official code, get results on the bundled examples and on our own images, and understand each step; matching the paper's metrics is not required.
- The official code is vendored unmodified at `third_party/IDArb/` (upstream commit `df26b90`); local changes are separate commits.
- **Where things run:** IDArb decomposition needs a GPU and runs on the cluster. Everything downstream (for example normal-to-mesh) runs on the owner's laptop. Going forward the work uses single images only; the training data is not needed locally (supervisor, 2026-09-25).
- **Training smoke test** (supervisor request): train IDArb from Stable Diffusion 2.1 weights, not IDArb's released weights, on a few public objects to confirm the training code works. Result: 3 Arb-Objaverse objects, 300 steps on one A40 in 7m22s; validation normal cosine similarity rose from -0.19 to 0.79. This shows the model can fit the training objects, not that it generalises. Inferring at 512 px with weights trained at 256 px lowered normal cosine similarity from 0.93 to 0.77.
- **Upstream bug fixed:** `mv_dataset.py` `load_mr_abo` used `out_imgs` before creating it, so training on ABO or Arb-Objaverse crashed.
- **Normal-to-mesh** (supervisor suggestion: Poisson reconstruction): implemented both Poisson normal integration and screened Poisson surface reconstruction in `scripts/idarb/normal_to_mesh.py`. The supervisor observed that single-view meshes look flattened. Experiments traced most of this to depth discontinuities at occlusion boundaries, which normals cannot encode (true gaps of 49/99/149 px were recovered as about 5 px). Per the supervisor, meshes are now centred at the origin and can be stretched along Z (`--z_scale`).
- **Multi-view reconstruction** is not pursued; the supervisor considers novel-view-synthesis methods sufficient for that.

## 2026-09-30 — Depth prior for normal-to-mesh

- **Problem:** meshes from normal integration look flattened. The supervisor noted that even the visible surface is squashed. The cause is depth discontinuities at occlusion boundaries, which normals cannot encode, plus small normal biases that integrate into a global tilt.
- **Change:** add a monocular depth prior. `scripts/idarb/estimate_depth.py` runs Depth Anything V2 Small on the server; `scripts/idarb/normal_to_mesh.py --depth_dir` solves `min Σ(Δh − g)² + w·Σ(h − s·d)²`, drops gradient constraints across depth jumps detected in the depth map, merges fragments smaller than 300 px back into the neighbouring part they share the most cut edges with, and removes the "wall" triangles that span a cut. Without `--depth_dir` the behaviour is unchanged.
- **Results:** on a synthetic two-sphere scene the recovered occlusion gap is 99.4 px (true 99.4; normals only 5.4). On samples, chair1 depth/width goes from 0.28 to 0.82 and labubu's tilt disappears; objects without occlusions change little. Surface-to-normal deviation grows (chair1 4.7° to about 11°); the weight barely changes the depth ratio between 0.01 and 0.2, so the default is 0.01.
- **Fragment fix:** a first version cut thin parts into many floating pieces (chair1: 67 parts, 64 under 300 px), reported from the Mac run. Merging each small fragment into only one neighbour removes them (chair1: 3 parts) without re-gluing parts that should stay apart (superman's cape stays separate from the body); merging a fragment into all neighbours did glue them.
- **Model choice:** Small (Apache-2.0, 99 MB) instead of Large (CC-BY-NC-4.0, 1.3 GB); on five samples their depth maps correlate at 0.94 to 0.997 and give nearly identical meshes.
- **Workflow:** depth maps are produced on the server and pushed with the decomposition results, so mesh reconstruction on the laptop still needs no PyTorch.
- **Correction on screened Poisson (method B):** on single images it does not produce a closed mesh or a back side, contrary to earlier comments. On panda, chair1 and bird the raw output is the front surface plus a flat sheet extending from the object's outline to the bounding box; after density trimming only the front remains, with a median distance of about 0.45 px to the method-A surface. The preview's 180° view had looked like a back side because the preview shades both faces of a surface alike. Comments and the README were corrected.

## 2026-10-10 — Tried: Poisson reconstruction on depth-map points (not adopted)

- **Supervisor's suggestion:** build a point cloud directly from the Depth Anything depth map (one point
  per pixel, z = s·d), attach IDArb's normal to every point, and run screened Poisson surface
  reconstruction on that cloud, so that the depth map gives the front-to-back structure and the normals
  give the surface detail.
- **Result:** the normals have almost no effect. Mean angle between the mesh's normals and IDArb's
  normals (smaller is better; the joint solve is the current default):

  | Object | normals only | joint solve (normals + depth) | depth map alone | Poisson on depth points |
  |---|---|---|---|---|
  | chair1 | 4.8° | 11.0° | 17.0° | 16.9° |
  | superman | 7.6° | 13.9° | 17.9° | 18.0° |
  | panda | 3.3° | 7.2° | 11.9° | 11.4° |

  The depth-to-width ratio is the same for all depth-based variants (chair1: 0.82). Octree depths 6 to 9
  give 20.2° to 16.9° on chair1, so a coarser octree does not help either.
- **Why:** screened Poisson pins the surface to the input points. With a point in every octree cell
  (depth 9 is about one pixel), the surface passes through the depth-map points, including their noise and
  blurred occlusion edges, and the normals only shape the surface inside each cell. The joint solve in
  `normal_to_mesh.py` weights the depth term at 0.01, so its surface can leave the depth-map points to
  follow the normals; in effect the Poisson variant is the joint solve with a very large depth weight.
  Open3D 0.18 does not expose PoissonRecon's point weight, which would be the parameter to lower.
- **Decision:** keep the joint solve. Script: `scripts/idarb/poisson_on_depth_points.py`.
