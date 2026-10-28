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

# Full pilot training (2026-10-29)

All 275 pilot partials (hpc02, hpc01 and the laptop's 33 from features-dataset-win-tmp a91ee41),
read through a symlink view ~/semtex/features_train (140 fulls + 275 partials + 275 targets,
checked: no dangling link, no manifest pair missing a file). 196 train pairs, the same 40 val
pairs as the early runs (val_before identical). Code c94c442, 20 epochs, --val_max_pairs 40.

- B (hpc02): `--feature_dir ~/semtex/features_train --epochs 20 --val_max_pairs 40 --device cuda
  --out pilot_full_B.npz --w_soft 1 --w_nce 0`, next to the tier-B relief extraction.
- C (hpc01): the same as B plus `--w_resid 1`; A (hpc01): defaults. Both ran in parallel on hpc01
  by the owner's decision, from a copy of features_train pulled from hpc02 (same 690 files, val_before
  identical), with code c94c442 and the same env; run_full_training_hpc01.sh is hpc01's copy of
  run(). At 06:46 the time limit was raised from 3 h to 5 h: guard_5h_hpc01.sh took over the 30 s
  sampling and the stop rules without touching the trainers (summary_full_training_hpc01.txt).

Results (f2p acc_1pct on val, val_before 0.3066): B 0.3718 (epoch 20), C 0.381 (epoch 5),
A 0.3677 (epoch 15).

`run_full_training_hpc02.sh` is the script as it ended: its lines for C and A were turned into
comments while B ran, so hpc02 ran only B (see summary_full_training_hpc02.txt).
