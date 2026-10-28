#!/bin/bash
# Full pilot training of the feature adapter (runs in tmux; Mac's request of 10-28, started only
# after the owner's agreement in the server window). Waits until all 275 partials exist (hpc02's
# and hpc01's in features_dataset, the laptop's 33 in ~/semtex/features_win), builds the symlink
# view ~/semtex/features_train (140 fulls + 275 partials + 275 targets, no dangling link), then runs
# B (--w_soft 1 --w_nce 0), C (B plus --w_resid 1) and A (defaults) one after another, 20 epochs,
# all train pairs, --val_max_pairs 40. Every 30 s the trainer's GPU memory, the card's memory, RAM
# available and the trainer's RSS go to samples_full_<run>.txt; a run is stopped after 3 h or when
# RAM available drops under 1.2 GB, and the next run starts only with at least 6 GB available.
# Code: the worktree ~/semtex/wt_train, detached at c94c442 (diff3f-registration).
set -u
CODE=~/semtex/wt_train
FD=~/semtex/wt_fdata/results/diff3f/features_dataset
FW=~/semtex/features_win/results/diff3f/features_dataset
FT=~/semtex/features_train
PY=~/miniconda3/envs/diff3f/bin/python
H=~/semtex/adapter
OUT=$CODE/outputs/adapter
mkdir -p $OUT
log() { echo "$(date '+%m-%d %H:%M') $*" >> $H/summary_full.txt; }
stems() { tail -n +2 $FD/pilot_queue.txt | cut -f1; }
n_present() { stems | while read s; do { [ -f $FD/${s}_targets.npy ] || [ -f $FW/${s}_targets.npy ]; } && echo; done | wc -l; }

log "waiting for 275 partials (now $(n_present)), code $(git -C $CODE rev-parse --short HEAD)"
until [ $(n_present) -ge 275 ]; do sleep 300; done
sleep 180  # an rsync from hpc01 renames the npz before its targets; let the last pair settle

rm -rf $FT; mkdir -p $FT
for f in $FD/*_full.npz; do ln -s $f $FT/; done
for s in $(stems); do
  if [ -f $FD/${s}_targets.npy ]; then src=$FD; else src=$FW; fi
  ln -s $src/$s.npz $FT/$s.npz; ln -s $src/${s}_targets.npy $FT/${s}_targets.npy
done
nf=$(ls $FT/*_full.npz | wc -l); np=$(ls $FT/*_v[01].npz | wc -l); nt=$(ls $FT/*_targets.npy | wc -l)
dang=$(find -L $FT -type l | wc -l)
miss=$($PY -c "
import json, os
q = {l.split('\t')[0] for l in list(open('$FD/pilot_queue.txt'))[1:]}
m = [p for p in json.load(open(os.path.expanduser('~/semtex/dataset/manifest.json')))['pairs']
     if f\"{p['uid']}_v{p['view']}\" in q]
print(len(m) - 275 +sum(not all(os.path.exists(os.path.join('$FT', f)) for f in
          (f\"{p['uid']}_v{p['view']}.npz\", f\"{p['uid']}_v{p['view']}_targets.npy\", f\"{p['uid']}_full.npz\"))
          for p in m))")
log "features_train: $nf fulls, $np partials, $nt targets, $dang dangling links, $miss manifest pairs missing a file"
if [ $nf -ne 140 ] || [ $np -ne 275 ] || [ $nt -ne 275 ] || [ $dang -ne 0 ] || [ $miss -ne 0 ]; then
  log "FULL TRAINING NOT STARTED: features_train check failed"; exit 1
fi

run() {  # run <name> [extra args]
  name=$1; shift
  avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  if [ $avail -lt 6000 ]; then log "run $name not started: RAM available $avail MiB (< 6000)"; return; fi
  cd $CODE
  $PY scripts/idarb/train_adapter.py --root ~/semtex/dataset --feature_dir $FT --epochs 20 \
      --val_max_pairs 40 --device cuda --out $OUT/pilot_full_$name.npz "$@" > $OUT/pilot_full_$name.log 2>&1 &
  pid=$!
  t0=$(date +%s); why=""
  while kill -0 $pid 2>/dev/null; do
    gpu=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -F', ' -v p=$pid '$1==p{print $2}')
    card=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
    rss=$(ps -o rss= -p $pid | awk '{print int($1/1024)}')
    echo "$(date +%H:%M:%S) trainer_gpu_mib ${gpu:-0} card_mib $card ram_avail_mib $avail trainer_rss_mib ${rss:-0}" >> $H/samples_full_$name.txt
    if [ $(( $(date +%s) - t0 )) -gt 10800 ]; then why="over 3 h"; fi
    if [ "${avail:-99999}" -lt 1200 ]; then why="RAM available ${avail} MiB"; fi
    if [ -n "$why" ]; then kill $pid; sleep 10; kill -9 $pid 2>/dev/null; break; fi
    sleep 30
  done
  wait $pid; rc=$?
  peak=$(awk '{print $3}' $H/samples_full_$name.txt | sort -n | tail -1)
  minram=$(awk '{print $7}' $H/samples_full_$name.txt | sort -n | head -1)
  log "run $name exit $rc $(( $(date +%s) - t0 )) s ${why:+STOPPED ($why) }trainer GPU peak $peak MiB, RAM available min $minram MiB"
  log "run $name: $(tail -1 $OUT/pilot_full_$name.log)"
}

run B --w_soft 1 --w_nce 0
#un C --w_soft 1 --w_nce 0 --w_resid 1
#un A
log "FULL TRAINING DONE"
