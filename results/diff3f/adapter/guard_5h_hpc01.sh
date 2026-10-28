#!/bin/bash
# Takes over the stop rules of a running run_full_hpc01.sh (Mac's request of 10-29 06:50: time limit
# 3 h -> 5 h, RAM rule unchanged, trainer not restarted). The wrapper's limit is a literal in its
# already-parsed loop, and SIGSTOP does not hold (tmux resumes a stopped pane process). Instead the
# wrapper's samples path becomes a FIFO without a reader: its next `echo >> samples_full_<name>.txt`
# blocks in open(), before its time check. This guard writes the same 30 s samples to the real file
# (kept as samples_full_<name>.real.txt meanwhile) and stops the trainer after 5 h of its run time or
# when RAM available drops under 1.2 GB. When the trainer has exited, the real file goes back to its
# path and the FIFO is read once, so the wrapper finishes its loop and writes its summary lines.
# Usage: guard_5h.sh <name> <trainer_pid>
set -u
name=$1; pid=$2
H=~/semtex/adapter
S=$H/samples_full_$name.txt; R=$H/samples_full_$name.real.txt; F=$H/samples_full_$name.fifo
log() { echo "$(date '+%m-%d %H:%M') $*" >> $H/summary_full.txt; }
mkfifo $F.new && mv $S $R && mv $F.new $S
log "run $name: wrapper blocked on a FIFO at $S; guard_5h.sh watches trainer $pid (limit 5 h, RAM 1.2 GB)"
why=""
while kill -0 $pid 2>/dev/null; do
  gpu=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -F', ' -v p=$pid '$1==p{print $2}')
  card=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  rss=$(ps -o rss= -p $pid | awk '{print int($1/1024)}')
  echo "$(date +%H:%M:%S) trainer_gpu_mib ${gpu:-0} card_mib $card ram_avail_mib $avail trainer_rss_mib ${rss:-0}" >> $R
  el=$(ps -o etimes= -p $pid | tr -d ' ')
  if [ "${el:-0}" -gt 18000 ]; then why="over 5 h"; fi
  if [ "${avail:-99999}" -lt 1200 ]; then why="RAM available ${avail} MiB"; fi
  if [ -n "$why" ]; then log "run $name STOPPED by guard ($why)"; kill $pid; sleep 10; kill -9 $pid 2>/dev/null; break; fi
  sleep 30
done
mv $S $F && mv $R $S
timeout 120 cat $F >> $S   # lets the blocked wrapper write its last sample line and go on
rm -f $F
log "run $name: trainer $pid exited, wrapper released (its 'over 3 h' note, if any, is moot: the trainer had already exited)"
