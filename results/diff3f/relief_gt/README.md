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

## eval/: tier-B measurement (10-30)

`eval_relief_tierb.py` computes train_adapter.py's validation metric (its evaluate() code) on
three sets, with the identity and with adapter B (pilot_full_B.npz): `train_val40` (the training
run's 40 val pairs, a check: it reproduces the run's val_before exactly), `tierB_relief` (the 40
reliefs) and `tierA_cuts_same_objects` (the 79 tier-A val cuts of the same 40 objects). Columns
without `pooled_` are means over pairs, as train_adapter reports them; `pooled_` over all points.
Errors are fractions of the full diagonal. Command in `command.txt`, output in `run.log`.
