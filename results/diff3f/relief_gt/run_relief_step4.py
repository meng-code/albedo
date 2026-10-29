"""Validation tier B, steps 2-3 and 4 after the pilot partials (runs in tmux; Mac's request of
10-27). Steps 2-3 (run_relief_steps123.sh: IDArb, depth, normal_to_mesh, attach) cannot share the
GPU with a partial job: IDArb ran out of memory on its third image next to it (attempt 1, 10-27).
So once the partials are done this runs run_relief_steps123.sh (the renders of step 1 exist and
are kept) and then step 4: Diff3F features of the 40 val reliefs.

The pilot partials' recipe: --front_only --zoom 2 --lift gpu --num_views 100 --seed 0, prompt =
category, at a vertex list: the relief vertices whose orig_idx (make_relief_gt.py attach) is among
the full npz's sampled idx, a seeded random subset of 6000 (seed 0, sorted) when more; relief
vertices without ground truth (orig_idx -1) are never listed. Sidecar <uid>_relief_targets.npy:
for every sample, the row of <uid>_full.npz that is its true vertex. Code: ~/semtex/wt_run2
(3a127e8, --vertex_list). Output: features_dataset/<uid>_relief.npz. A relief with fewer than 2000
vertices with ground truth counts as failed and is not extracted. After the last relief, the
features pusher (push_features_dataset.sh) is started again once the earlier one has ended, so the
relief features go to features-dataset-tmp with SHA256SUMS like the rest.
"""

import json
import os
import subprocess
import time

import numpy as np

HOME = os.path.expanduser("~")
CODE = f"{HOME}/semtex/wt_run2"
OUT = f"{HOME}/semtex/wt_fdata/results/diff3f/features_dataset"
H = f"{HOME}/semtex/relief_gt"
LISTS = f"{H}/vertex_lists"
PILOT = f"{HOME}/semtex/zoom/pilot"
PY = f"{HOME}/miniconda3/envs/diff3f/bin/python"
os.makedirs(LISTS, exist_ok=True)


def read(path):
    return open(path).read() if os.path.exists(path) else ""


def summary(line):
    with open(f"{H}/summary_step4.txt", "a") as f:
        f.write(time.strftime("%m-%d %H:%M ") + line + "\n")


while "PARTIALS DONE" not in read(f"{PILOT}/summary_partials.txt"):
    time.sleep(60)
t0 = time.time()
rc = subprocess.run(["bash", f"{H}/run_relief_steps123.sh"]).returncode
summary(f"steps 1-3 exit {rc} {time.time() - t0:.0f} s (summary.txt)")
objs = [o for o in json.load(open(f"{HOME}/semtex/zoom/pilot_objects.json")) if o["split"] == "val"]
summary(f"start: {len(objs)} val reliefs")
t_start = time.time()
for o in objs:
    uid = o["uid"]
    out = f"{OUT}/{uid}_relief.npz"
    gt_npz = f"{H}/{uid}/{uid}_relief_gt.npz"
    if os.path.exists(out):
        continue
    if not os.path.exists(gt_npz):
        summary(f"{uid} SKIPPED: no relief ground truth (steps 1-3 failed)")
        continue
    full = np.load(f"{OUT}/{uid}_full.npz")["idx"].astype(np.int64)
    orig = np.load(gt_npz)["orig_idx"].astype(np.int64)
    n_gt = int((orig >= 0).sum())
    if n_gt < 2000:
        summary(f"{uid} FAILED relief: {n_gt} vertices with ground truth (< 2000)")
        continue
    row_of = -np.ones(int(max(full.max(), orig.max())) + 1, np.int64)
    row_of[full] = np.arange(len(full))
    has = np.zeros(len(orig), bool)
    has[orig >= 0] = row_of[orig[orig >= 0]] >= 0  # -1 must not index row_of from the end
    cand = np.nonzero(has)[0]
    if len(cand) > 6000:
        cand = np.sort(np.random.default_rng(0).choice(cand, 6000, replace=False))
    targets = row_of[orig[cand]]
    lst = f"{LISTS}/{uid}_relief.npy"
    np.save(lst, cand)
    cmd = [PY, "scripts/idarb/diff3f_features.py", "--mesh", f"{H}/{uid}/relief.ply",
           "--prompt", o["prompt"], "--front_only", "--no_cpu_offload", "--zoom", "2", "--lift",
           "gpu", "--num_views", "100", "--seed", "0", "--vertex_list", lst, "--out", out]
    t0 = time.time()
    with open(f"{H}/log_features_{uid}.txt", "w") as f:
        rc = subprocess.run(cmd, cwd=CODE, stdout=f, stderr=subprocess.STDOUT).returncode
    ok, meta = False, {}
    if rc == 0:
        z = np.load(out, allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        ok = bool(np.array_equal(z["idx"], cand)) and bool(np.array_equal(full[targets], orig[cand]))
        np.save(f"{OUT}/{uid}_relief_targets.npy", targets)
    summary(f"{uid} exit {rc} {time.time() - t0:.0f} s samples {len(cand)} of {len(orig)} "
            f"(with gt {n_gt}) check {'ok' if ok else 'FAILED'} views {meta.get('views_used')} "
            f"peak {meta.get('peak_gpu_mib')} MiB")
summary(f"RELIEF FEATURES DONE {(time.time() - t_start) / 3600:.1f} h")

# the earlier pusher ends by itself once everything it saw is pushed; then start it again
while subprocess.run(["tmux", "has", "-t", "pilotpush"], capture_output=True).returncode == 0:
    time.sleep(60)
subprocess.run(["tmux", "new", "-d", "-s", "pilotpush", f"{HOME}/semtex/zoom/push_features_dataset.sh"])
summary("pusher started again for the relief features")
