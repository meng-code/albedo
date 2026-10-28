# Early pilot training of the feature adapter (GPU server hpc02, 2026-10-28)

Code: diff3f-registration c94c442, scripts/idarb/train_adapter.py, diff3f env (torch 2.1.0+cu118),
run by ~/semtex/adapter/run_early_training.sh next to the partial extraction. Started when 80
train partials existed in results/diff3f/features_dataset (A: 80 train pairs, B: 82); 40 val
pairs (--val_max_pairs 40, seeded choice of the 79).

- A: `--root ~/semtex/dataset --feature_dir results/diff3f/features_dataset --epochs 20
  --val_max_pairs 40 --device cuda --out pilot_early_A.npz`
- B: the same plus `--w_soft 1 --w_nce 0`, run because A's summary said beats_identity true.

`pilot_early_<run>.npz`: weights of the best epoch (f2p acc_1pct on val), `.log`: the trainer's
output, `samples_<run>.txt`: trainer GPU memory, card memory, RAM available and trainer RSS every
30 s, `summary_early_training.txt`: the waiter's log. SHA256SUMS lists every file here.
