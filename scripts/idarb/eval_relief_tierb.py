"""
Tier-B measurement: the validation metric of train_adapter.py
(val_before / val) on the 40 tier-B reliefs, i.e. <uid>_relief.npz from renders of the full mesh
through the real pipeline, against <uid>_full.npz, ground truth from <uid>_relief_gt.npz; with the
unadapted features (identity) and with saved adapter weights. For comparison the same metric on
the tier-A validation cuts (<uid>_v<k>) of the same objects, and, as a check of this code, on the
exact 40 val pairs the training run used (it must reproduce the run's val_before).

The per-pair evaluation is train_adapter.evaluate's code: p2f on a seeded subsample of 4096
partial samples, f2p over the full samples within 1 % of some partial sample's true position,
errors as fractions of the full diagonal, acc_2pct_partial at 2 % of the partial's diagonal.
Reported both as train_adapter reports it (mean over pairs) and pooled over all points, plus
the mean and median error.

Usage (server, idarb environment; run on hpc02 on 2026-10-30, 149 s):
    python scripts/idarb/eval_relief_tierb.py --root ~/semtex/dataset \
        --feature_dir ~/semtex/features_train --relief_features <features_dataset> \
        --relief_gt ~/semtex/relief_gt --weights B=<pilot_full_B.npz> \
        --out results/diff3f/relief_gt/tierb_metrics.csv
"""

import argparse
import csv
import json
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", help="checkout whose scripts/idarb holds train_adapter.py and "
                    "adapter_data.py (default: this script's own directory)")
    ap.add_argument("--root", required=True)
    ap.add_argument("--feature_dir", required=True, help="tier-A features (fulls, partials)")
    ap.add_argument("--relief_features", required=True)
    ap.add_argument("--relief_gt", required=True)
    ap.add_argument("--weights", action="append", default=[], help="NAME=adapter.npz")
    ap.add_argument("--val_samples", type=int, default=4096)
    ap.add_argument("--val_max_pairs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(args.code, "scripts", "idarb") if args.code
                    else os.path.dirname(os.path.abspath(__file__)))
    import torch

    from adapter_data import dataset_pairs, pair_arrays
    from train_adapter import Adapter, load_full_vertices

    dev = args.device
    val = dataset_pairs(args.root, args.feature_dir, "val")
    uids = sorted({m["uid"] for m in val})
    relief = []
    for u in uids:
        p = os.path.join(args.relief_features, u + "_relief.npz")
        if os.path.exists(p):
            relief.append(dict(uid=u, stem=u + "_relief", partial_npz=p,
                               full_npz=os.path.join(args.feature_dir, u + "_full.npz"),
                               targets_npy=os.path.join(args.relief_features,
                                                        u + "_relief_targets.npy"),
                               gt_npz=os.path.join(args.relief_gt, u, u + "_relief_gt.npz"),
                               full_ply=next(m["full_ply"] for m in val if m["uid"] == u)))
    relief_uids = {m["uid"] for m in relief}
    # the training run's val subset: same rng calls as train_adapter.main
    rng = np.random.default_rng(args.seed)
    train_val = val
    if args.val_max_pairs and len(val) > args.val_max_pairs:
        train_val = [val[i] for i in rng.permutation(len(val))[:args.val_max_pairs]]
    sets = [("train_val40", train_val, rng),
            ("tierB_relief", relief, np.random.default_rng(args.seed)),
            ("tierA_cuts_same_objects", [m for m in val if m["uid"] in relief_uids],
             np.random.default_rng(args.seed))]

    nets = [("identity", None)]
    for w in args.weights:
        name, path = w.split("=", 1)
        z = np.load(path)
        meta = json.loads(str(z["meta"]))
        if meta["args"].get("both"):
            sys.exit("--both adapters are not handled here")
        net = Adapter(int(meta["dim"]), int(meta["args"]["hidden"]), dev)
        for mod, kw, kb in ((net.ln, "ln_w", "ln_b"), (net.l1, "w1", "b1"), (net.l2, "w2", "b2")):
            mod.weight.data = torch.from_numpy(z[kw]).to(dev)
            mod.bias.data = torch.from_numpy(z[kb]).to(dev)
        nets.append((name, net))
        print(json.dumps(dict(weights=name, path=path, epoch=meta["epoch"],
                              identity=bool(meta.get("identity")))), flush=True)

    full_vertices = {}
    rows = []
    for set_name, pairs, srng in sets:
        data = []
        for m in pairs:
            if m["uid"] not in full_vertices:
                full_vertices[m["uid"]] = load_full_vertices(m["full_ply"])
            a = pair_arrays(m["partial_npz"], m["full_npz"], m["gt_npz"],
                            full_vertices[m["uid"]], targets_npy=m.get("targets_npy"))
            diag = float(np.linalg.norm(a["pos_all"].max(axis=0) - a["pos_all"].min(axis=0)))
            n = len(a["fp"])
            sel = srng.choice(n, min(args.val_samples, n), replace=False) if n else np.zeros(0, int)
            data.append(dict(
                fp=torch.from_numpy(a["fp"][sel]), pos=torch.from_numpy(a["pos"][sel]),
                fp_all=torch.from_numpy(a["fp"]).half(), pos_p_all=torch.from_numpy(a["pos"]),
                ff_all=torch.from_numpy(a["ff_all"]).half(),
                pos_all=torch.from_numpy(a["pos_all"]), diag=diag,
                diag_p=float(np.linalg.norm(a["pos"].max(axis=0) - a["pos"].min(axis=0)))
                if len(a["pos"]) else diag, keep=a["keep_fraction"]))
        for net_name, net in nets:
            for key in ("p2f", "f2p"):
                per = {k: [] for k in ("acc1", "acc5", "acc2p", "mean_err")}
                pooled = []
                with torch.no_grad():
                    for v in data:
                        if len(v["fp"]) == 0:
                            continue
                        ff = v["ff_all"].to(dev).float()
                        pos_all = v["pos_all"].to(dev)
                        if key == "p2f":
                            fp = v["fp"].to(dev)
                            if net is not None:
                                fp = net(fp)
                            nn = (fp @ ff.T).argmax(dim=1)
                            err = torch.linalg.norm(pos_all[nn] - v["pos"].to(dev), dim=1)
                            err = err / v["diag"]
                        else:
                            fp_all = v["fp_all"].to(dev).float()
                            if net is not None:
                                fp_all = net(fp_all)
                            pos_p_all = v["pos_p_all"].to(dev)
                            dmin, _ = torch.cdist(pos_all, pos_p_all).min(dim=1)
                            has = dmin < 0.01 * v["diag"]
                            if not bool(has.any()):
                                continue
                            nn = (ff[has] @ fp_all.T).argmax(dim=1)
                            err = torch.linalg.norm(pos_p_all[nn] - pos_all[has], dim=1) / v["diag"]
                        err = err.cpu().double().numpy()
                        e2p = err * v["diag"] / v["diag_p"]
                        per["acc1"].append(float((err < 0.01).mean()))
                        per["acc5"].append(float((err < 0.05).mean()))
                        per["acc2p"].append(float((e2p < 0.02).mean()))
                        per["mean_err"].append(float(err.mean()))
                        pooled.append(np.stack([err, e2p], axis=1))
                P = np.concatenate(pooled) if pooled else np.zeros((0, 2))
                row = dict(set=set_name, adapter=net_name, direction=key, pairs=len(per["acc1"]),
                           points=len(P),
                           keep_fraction_mean=round(float(np.mean([v["keep"] for v in data])), 4),
                           acc_1pct=round(float(np.mean(per["acc1"])), 4),
                           acc_5pct=round(float(np.mean(per["acc5"])), 4),
                           acc_2pct_partial=round(float(np.mean(per["acc2p"])), 4),
                           mean_err=round(float(np.mean(per["mean_err"])), 4),
                           pooled_acc_1pct=round(float((P[:, 0] < 0.01).mean()), 4),
                           pooled_acc_5pct=round(float((P[:, 0] < 0.05).mean()), 4),
                           pooled_acc_2pct_partial=round(float((P[:, 1] < 0.02).mean()), 4),
                           pooled_mean_err=round(float(P[:, 0].mean()), 4),
                           pooled_median_err=round(float(np.median(P[:, 0])), 4))
                rows.append(row)
                print(json.dumps(row), flush=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
