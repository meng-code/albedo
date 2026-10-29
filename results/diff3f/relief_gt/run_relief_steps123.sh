#!/bin/bash
# Validation tier B, steps 1-3 for the 40 val objects (runs in tmux; Mac's request of 10-27).
# Code: the worktree ~/semtex/wt_relief, detached at diff3f-registration 27b87ab.
#   (1) make_relief_gt.py render  (dataset env)  view = the v0 partial's cutting camera
#   (2) IDArb exactly as for the real photos (README step 1): prepare_images.py, then
#       main.py --input_type single (idarb env); estimate_depth.py (Depth Anything V2 Small, idarb
#       env); normal_to_mesh.py batch mode with the depth prior and default settings, as for
#       outputs/mesh_depth (relief only, dataset env)
#   (3) make_relief_gt.py attach  (dataset env)
# Usage: run_relief_steps123.sh [number of objects, default all 40]
# Per-step seconds go to summary.txt; the GPU steps run as one process each over all images.
set -u
H=~/semtex/relief_gt
CODE=~/semtex/wt_relief
DS=~/miniconda3/envs/dataset/bin/python
ID=~/miniconda3/envs/idarb/bin/python
W=$H/_pipeline   # inputs/ (prepared RGBA), decomposition/ (IDArb), depth/, mesh_depth/
mkdir -p $W/raw $W/inputs $W/decomposition $W/depth $W/mesh_depth
log() { echo "$(date +%H:%M:%S) $*" | tee -a $H/summary.txt; }
N=${1:-40}
UIDS=$($DS -c "import json; j=json.load(open('$HOME/semtex/zoom/pilot_objects.json')); print(' '.join([x['uid'] for x in j if x['split']=='val'][:$N]))")
log "start: $(echo $UIDS | wc -w) val objects, code $(git -C $CODE rev-parse --short HEAD)"

t0=$(date +%s)
for u in $UIDS; do
  [ -f $H/$u/${u}_buffers.npz ] && continue
  s=$(date +%s)
  (cd $CODE && $DS scripts/idarb/make_relief_gt.py render --mesh ~/semtex/dataset/full/$u.ply \
      --gt ~/semtex/dataset/partial/${u}_v0_gt.npz --out_dir $H/$u) > $H/log_render_$u.txt 2>&1 \
    && log "render $u $(( $(date +%s) - s )) s $(tail -1 $H/log_render_$u.txt)" \
    || log "render $u FAILED (log_render_$u.txt)"
  [ -f $H/$u/$u.png ] && cp $H/$u/$u.png $W/raw/$u.png
done
log "step render done $(( $(date +%s) - t0 )) s"

# IDArb input preparation: the renders already carry a valid alpha, so prepare_images.py keeps it
s=$(date +%s)
(cd $CODE/third_party/IDArb && $ID ../../scripts/idarb/prepare_images.py --input $W/raw --output $W/inputs) \
  > $H/log_prepare.txt 2>&1 || log "prepare FAILED (log_prepare.txt)"
log "step prepare $(( $(date +%s) - s )) s, $(ls $W/inputs | wc -l) inputs"

s=$(date +%s)
(cd $CODE/third_party/IDArb && $ID main.py --data_dir $W/inputs --output_dir $W/decomposition --input_type single) \
  > $H/log_idarb.txt 2>&1 || log "IDArb FAILED (log_idarb.txt)"
log "step IDArb $(( $(date +%s) - s )) s, $(ls $W/decomposition | wc -l) decompositions"

s=$(date +%s)
(cd $CODE && $ID scripts/idarb/estimate_depth.py --input_dir $W/inputs --out_dir $W/depth) \
  > $H/log_depth.txt 2>&1 || log "depth FAILED (log_depth.txt)"
log "step depth $(( $(date +%s) - s )) s, $(ls $W/depth | wc -l) depth maps"

for u in $UIDS; do
  [ -f $W/decomposition/${u}_00.png ] || { log "mesh $u SKIPPED: no IDArb result"; continue; }
  s=$(date +%s)
  (cd $CODE && $DS scripts/idarb/normal_to_mesh.py --result $W/decomposition/${u}_00.png \
      --input $W/inputs/$u.png --depth $W/depth/$u.png --methods relief --out_dir $W/mesh_depth) > $H/log_mesh_$u.txt 2>&1 \
    && [ -f $W/mesh_depth/$u/relief.ply ] \
    && { cp $W/mesh_depth/$u/relief.ply $H/$u/relief.ply; log "mesh $u $(( $(date +%s) - s )) s"; } \
    || { log "mesh $u FAILED (log_mesh_$u.txt)"; continue; }
  s=$(date +%s)
  (cd $CODE && $DS scripts/idarb/make_relief_gt.py attach --relief $H/$u/relief.ply \
      --buffers $H/$u/${u}_buffers.npz --out $H/$u/${u}_relief_gt.npz) > $H/log_attach_$u.txt 2>&1 \
    && log "attach $u $(( $(date +%s) - s )) s $(tail -1 $H/log_attach_$u.txt)" \
    || log "attach $u FAILED (log_attach_$u.txt)"
done
log "STEPS 1-3 DONE"
