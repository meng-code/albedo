# Diff3F features of the partial-to-full dataset (phase B pilot, GPU server)

Origin: `~/semtex/wt_fdata/results/diff3f/features_dataset/` on the GPU server (hpc-jliu02),
from 2026-10-26. Dataset: ~/semtex/dataset (make_partial_dataset.py; archive-server-dataset-tmp).
Pilot objects: the 40 val objects and the first 100 train objects of objects.csv.

- `<uid>_full.npz`: Diff3F features of the full mesh, --zoom 2 --lift gpu --num_views 100
  --seed 0, cameras all around, 15000 sampled vertices, prompt = category with underscores as
  spaces (code: diff3f-registration 1caeb01).
- `<uid>_v<k>.npz`: the partial, --front_only, same settings, at a vertex list: the partial
  vertices whose true full vertex is among the full npz's idx, a seeded random subset of 6000
  when more (code 3a127e8, --vertex_list).
- `<uid>_v<k>_targets.npy`: for every partial sample, the row of `<uid>_full.npz` holding its
  true vertex.

SHA256SUMS lists every file (sha256sum -c SHA256SUMS in results/diff3f/features_dataset/). Pushed
in commits under 1 GB as the files accumulate.

Added 10-27: `vertex_lists/<uid>_v<k>.npy` for all 275 pilot partials (the --vertex_list of each
partial; the 45 the runner had written by then are byte-identical, the rest computed ahead with
the same code, ~/semtex/zoom/make_queue_and_lists.py) and `pilot_queue.txt` (the runner's order,
split, prompt, done or todo at the time of writing).
