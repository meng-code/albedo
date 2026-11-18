# Part-naming test (SigLIP2 joint space), run on the GPU server 2026-10-23

Stage 1 (diff3fvar env), views and part-id masks, written outside the repository:

```
# Stage 1 of the SigLIP2 part-naming test: Diff3F's painted views and part-id masks.
# Runs in tmux so it survives the remote desktop closing.
cd ~/semtex/wt_f3d
P=~/miniconda3/envs/diff3fvar/bin/python
O=~/semtex/partviews
PL=~/semtex/part_labels/results/diff3f/part_labels_tmp
S=$(date +%s); $P scripts/idarb/diff3f_features.py --mesh outputs/mesh_depth/labubu/relief.ply --prompt "plush toy" --front_only --no_cpu_offload --save_views $O/labubu_partial --part_labels $PL/labubu_partial_parts.npz --out $O/labubu_partial/feat.npz > $O/log_labubu_partial.txt 2>&1; echo "labubu_partial exit $? $(( $(date +%s) - S )) s" >> $O/summary.txt
S=$(date +%s); $P scripts/idarb/diff3f_features.py --mesh third_party/Diff3F/meshes/camel.obj --prompt "camel" --no_cpu_offload --save_views $O/camel_full --part_labels $PL/camel_full_parts.npz --out $O/camel_full/feat.npz > $O/log_camel_full.txt 2>&1; echo "camel_full exit $? $(( $(date +%s) - S )) s" >> $O/summary.txt
S=$(date +%s); $P scripts/idarb/diff3f_features.py --mesh outputs/mesh_depth/cat/relief.ply --prompt "cat" --front_only --no_cpu_offload --save_views $O/cat_partial --part_labels $PL/cat_partial_parts.npz --out $O/cat_partial/feat.npz > $O/log_cat_partial.txt 2>&1; echo "cat_partial exit $? $(( $(date +%s) - S )) s" >> $O/summary.txt
S=$(date +%s); $P scripts/idarb/diff3f_features.py --mesh third_party/Diff3F/meshes/man.obj --prompt "man" --no_cpu_offload --save_views $O/man_full --part_labels $PL/man_full_parts.npz --out $O/man_full/feat.npz > $O/log_man_full.txt 2>&1; echo "man_full exit $? $(( $(date +%s) - S )) s" >> $O/summary.txt
S=$(date +%s); $P scripts/idarb/diff3f_features.py --mesh outputs/mesh_depth/superman/relief.ply --prompt "man" --front_only --no_cpu_offload --save_views $O/superman_partial --part_labels $PL/superman_partial_parts.npz --out $O/superman_partial/feat.npz > $O/log_superman_partial.txt 2>&1; echo "superman_partial exit $? $(( $(date +%s) - S )) s" >> $O/summary.txt
S=$(date +%s); $P scripts/idarb/diff3f_features.py --mesh third_party/Diff3F/meshes/lion.off --prompt "lion" --no_cpu_offload --save_views $O/lion_full --part_labels $PL/lion_full_parts.npz --out $O/lion_full/feat.npz > $O/log_lion_full.txt 2>&1; echo "lion_full exit $? $(( $(date +%s) - S )) s" >> $O/summary.txt
S=$(date +%s); $P scripts/idarb/diff3f_features.py --mesh outputs/mesh_depth/cow/relief.ply --prompt "cow" --front_only --no_cpu_offload --save_views $O/cow_partial --part_labels $PL/cow_partial_parts.npz --out $O/cow_partial/feat.npz > $O/log_cow_partial.txt 2>&1; echo "cow_partial exit $? $(( $(date +%s) - S )) s" >> $O/summary.txt
S=$(date +%s); $P scripts/idarb/diff3f_features.py --mesh third_party/Diff3F/meshes/cow.obj --prompt "cow" --no_cpu_offload --save_views $O/cow_full --part_labels $PL/cow_full_parts.npz --out $O/cow_full/feat.npz > $O/log_cow_full.txt 2>&1; echo "cow_full exit $? $(( $(date +%s) - S )) s" >> $O/summary.txt
echo STAGE1 DONE >> $O/summary.txt
```

Stage 2 (siglip2oc env), this folder:

```
python scripts/idarb/siglip2_part_naming.py --views ~/semtex/partviews --labels ~/semtex/part_labels/results/diff3f/part_labels_tmp --out_dir results/diff3f/part_naming
```

Part labels: branch part-labels-tmp (70d5d7a), results/diff3f/part_labels_tmp/.
