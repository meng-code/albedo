# Diff3F features of pilot train partials, made on the Windows laptop (RTX 4080 Laptop GPU)

The laptop's share of the phase-B pilot: the last 70 train rows of pilot_queue.txt (from
features-dataset-tmp at a05aa8c), listed in `results/diff3f/features_dataset/share.txt`, walked from the last row upward.
The server's runner skips exactly these stems. Files here: 27 partials so far.

- `<stem>.npz`: Diff3F features of the partial, code diff3f-registration 3a127e8 (the server's
  commit), the server's command: `python scripts/idarb/diff3f_features.py --mesh
  <dataset>/partial/<stem>.ply --prompt "<prompt column>" --front_only --no_cpu_offload --zoom 2
  --lift gpu --num_views 100 --seed 0 --vertex_list vertex_lists/<stem>.npy --out <stem>.npz`,
  with the vertex lists pushed on features-dataset-tmp. The command and commit are also in each
  file's meta.
- `<stem>_targets.npy`: for every partial sample, the row of `<uid>_full.npz` (features-dataset-tmp)
  holding its true vertex: the position of gt orig_idx[vertex list] in the full file's idx
  (reproduces the server's targets for all 34 val partials checked).
- `runlog.tsv`: stem, prompt, finish time, wall seconds, peak GPU MiB, OOM retry (1 = rerun
  without --no_cpu_offload).
- Check against the server: on val partial 0a5f52a298864b20a290fae141c2791b_v0, laptop vs server
  per-vertex cosine mean 0.99998 (min 0.99987); p2f acc@1 % 0.1934 vs 0.1931.

SHA256SUMS lists every npz and targets file (sha256sum -c SHA256SUMS in results/diff3f/features_dataset/). Custody copy:
D:\albedo-archive\features-dataset-win on the laptop.
