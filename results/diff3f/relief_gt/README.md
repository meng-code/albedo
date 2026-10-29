# Tier-B relief validation inputs (hpc02, 10-27 to 10-30)

Made with `scripts/idarb/make_relief_gt.py` at 27b87ab (worktree ~/semtex/wt_relief), driven by
`run_relief_steps123.sh` (steps 1-3) and `run_relief_step4.py` (step 4, Diff3F features).

For each of the 40 validation objects, `<uid>/` holds:

- `<uid>.png`: the render of the full mesh (step 1), the input image of the real pipeline.
- `<uid>_buffers.npz`: the render buffers used to carry ground truth through the render (step 1).
- `relief.ply`: the relief mesh built by IDArb normals, Depth Anything V2 depth and `normal_to_mesh.py` (steps 2-3).
- `<uid>_relief_gt.npz`: ground truth from each relief vertex to the full mesh (every relief vertex has one).

`vertex_lists/<uid>_relief.npy` are the 6000 sampled relief vertices passed to `diff3f_features.py
--vertex_list`. The features themselves are in `../features_dataset/<uid>_relief.npz` and
`<uid>_relief_targets.npy` (commits 91198a3, ac34ff2). `logs/` has every step's log and the summaries
(`summary.txt` for steps 1-3, 542 s; `summary_step4.txt`, 18.9 h, 40/40 check ok).
The intermediate IDArb/depth outputs (`_pipeline/`, 498 MB) stay on hpc02 only.
