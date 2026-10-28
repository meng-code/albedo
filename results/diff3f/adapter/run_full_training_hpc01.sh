#!/bin/bash
# Full pilot adapter training on hpc01 (Mac's request of 10-29): runs C and A in parallel, each in its
# own tmux session via `run_full_hpc01.sh <name> [extra args]`. Same command, data and stop rules as
# hpc02's run_full_training.sh run(): every 30 s the trainer's GPU memory, the card's memory, RAM
# available and the trainer's RSS go to samples_full_<name>.txt; the run is stopped after 3 h or when
# RAM available drops under 1.2 GB. Code: ~/semtex/wt_train, detached at c94c442.
set -u
CODE=~/semtex/wt_train
FT=~/semtex/features_train
PY=~/miniconda3/envs/diff3f/bin/python
H=~/semtex/adapter
OUT=$CODE/outputs/adapter
mkdir -p $OUT
log() { echo "$(date '+%m-%d %H:%M') $*" >> $H/summary_full.txt; }

name=$1; shift
avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
if [ $avail -lt 6000 ]; then log "run $name not started: RAM available $avail MiB (< 6000)"; exit 1; fi
log "run $name start, code $(git -C $CODE rev-parse --short HEAD), args: $*"
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
