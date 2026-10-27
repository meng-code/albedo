"""
Train the feature adapter on the partial-to-full dataset (needs torch; runs on the server).

The adapter is a residual two-layer perceptron on the 2048-d Diff3F features of the PARTIAL
mesh (the relief side), trained so that a partial vertex's adapted feature matches the Diff3F
feature of its true vertex on the FULL mesh: a cosine regression to that feature plus an
InfoNCE term against the other sampled vertices of the same full mesh (the hard negatives
being the symmetric and nearby parts). The full side is left as it is, so the adapter maps
the relief's features into the full mesh's feature space, which is where the partial-shape
shift of front-only views lives. --both adapts both sides with one shared network instead.

Evaluation after every epoch on the validation split (on the GPU, 4096 partial samples per
pair): nearest-neighbour accuracy of the true full vertex within 1 % / 5 % of the full diagonal
(the GT measure of refine_correspondence.py), before and after the adapter. Pairs are read from
disk as they are used (the dataset's features do not fit in RAM), 4 pairs mixed per step; full
samples within 1 % of the true vertex, or whose feature is a near copy of the positive's
(cosine above --near_cos) while lying far away (--sym_dist: the symmetric counterpart), are
not negatives; near copies close by stay as hard negatives. Besides the partial -> full InfoNCE
there is one in the evaluation's direction (--w_nce_rev, full -> partial within the batch), an
optional soft-label cross-entropy over the candidates (--w_soft, Gaussian in the distance to the
true vertex, --sigma) and an optional residual penalty (--w_resid). The validation also reports
accuracy at 2 % of the PARTIAL's diagonal (the real task's threshold) and the share of points
correct before the adapter and wrong after it. If no epoch beats the unadapted features the
saved weights are the identity. Weights are saved as an npz that adapter_data.mlp_forward and
apply_adapter.py read without torch.

Usage (server, an environment with torch):
    python scripts/idarb/train_adapter.py --root ~/semtex/dataset \
        --feature_dir results/diff3f/features_dataset --out results/diff3f/adapter/adapter.npz
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adapter_data import dataset_pairs, pair_arrays  # noqa: E402


def load_full_vertices(ply):
    """The full mesh's vertices in the file's own numbering (the one orig_idx and the feature
    files' idx refer to), with no merging or reordering."""
    import trimesh

    return np.asarray(trimesh.load(ply, process=False, force="mesh").vertices, dtype=np.float64)


class Adapter:
    def __init__(self, dim, hidden, device):
        import torch

        self.ln = torch.nn.LayerNorm(dim).to(device)
        self.l1 = torch.nn.Linear(dim, hidden).to(device)
        self.l2 = torch.nn.Linear(hidden, dim).to(device)
        torch.nn.init.zeros_(self.l2.weight)  # starts as the identity map
        torch.nn.init.zeros_(self.l2.bias)
        self.torch = torch

    def parameters(self):
        return (list(self.ln.parameters()) + list(self.l1.parameters())
                + list(self.l2.parameters()))

    def __call__(self, x):
        h = self.l2(self.torch.nn.functional.gelu(self.l1(self.ln(x)), approximate="tanh"))
        return self.torch.nn.functional.normalize(x + h, dim=1)

    def weights(self):
        g = lambda t: t.detach().cpu().numpy().astype(np.float32)  # noqa: E731
        return dict(ln_w=g(self.ln.weight), ln_b=g(self.ln.bias), w1=g(self.l1.weight),
                    b1=g(self.l1.bias), w2=g(self.l2.weight), b2=g(self.l2.bias))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="dataset root (manifest.json, split.json)")
    ap.add_argument("--feature_dir", required=True)
    ap.add_argument("--out", required=True, help="weights npz")
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--w_regress", type=float, default=1.0)
    ap.add_argument("--w_nce", type=float, default=1.0)
    ap.add_argument("--negatives", type=int, default=2048, help="full samples per InfoNCE term")
    ap.add_argument("--batch_vertices", type=int, default=1024,
                    help="partial samples per pair per step")
    ap.add_argument("--pairs_per_step", type=int, default=4, help="pairs mixed in one step")
    ap.add_argument("--near_cos", type=float, default=0.95,
                    help="full samples whose feature is this close to the positive's AND lie "
                    "farther than --sym_dist from it are not negatives (the symmetric parts "
                    "Diff3F cannot tell apart); near copies close by stay hard negatives")
    ap.add_argument("--sym_dist", type=float, default=0.15,
                    help="fraction of the full diagonal beyond which a near copy counts as a "
                    "symmetric counterpart")
    ap.add_argument("--w_nce_rev", type=float, default=1.0,
                    help="InfoNCE in the evaluation's direction: a full sample against the "
                    "partial samples of the batch")
    ap.add_argument("--w_soft", type=float, default=0.0,
                    help="cross-entropy to a soft label over the full samples, Gaussian in the "
                    "distance to the true vertex (--sigma), instead of the one-hot InfoNCE")
    ap.add_argument("--sigma", type=float, default=0.01,
                    help="width of the soft label, fraction of the full diagonal")
    ap.add_argument("--w_resid", type=float, default=0.0,
                    help="penalty on the squared norm of the residual the adapter adds")
    ap.add_argument("--val_samples", type=int, default=4096,
                    help="partial samples per validation pair")
    ap.add_argument("--val_max_pairs", type=int, default=0,
                    help="validation pairs kept in RAM, a seeded choice across the manifest "
                    "(0: all; about 120 MB of RAM per pair at 4096 samples)")
    ap.add_argument("--both", action="store_true", help="adapt the full side too (shared net, "
                    "the regression target detached)")
    ap.add_argument("--max_pairs", type=int, default=0, help="limit for quick runs (0: all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = args.device
    train = dataset_pairs(args.root, args.feature_dir, "train")
    val = dataset_pairs(args.root, args.feature_dir, "val")
    if args.max_pairs:  # a quick run: a shuffle first, the manifest is grouped by category
        train = [train[i] for i in rng.permutation(len(train))[:args.max_pairs]]
        val = [val[i] for i in rng.permutation(len(val))[:max(args.max_pairs // 4, 1)]]
    if args.val_max_pairs and len(val) > args.val_max_pairs:
        val = [val[i] for i in rng.permutation(len(val))[:args.val_max_pairs]]
    print(json.dumps(dict(train_pairs=len(train), val_pairs=len(val))), flush=True)
    if not train:
        sys.exit("no training pairs with features")
    if not val:
        sys.exit("no validation pairs with features")

    full_vertices = {}

    def arrays(m):
        """One pair's arrays, read from disk when used (the whole dataset would not fit in
        RAM: features are float32 at up to 15000 x 2048 per mesh)."""
        if m["uid"] not in full_vertices:
            full_vertices[m["uid"]] = load_full_vertices(m["full_ply"])
        a = pair_arrays(m["partial_npz"], m["full_npz"], m["gt_npz"], full_vertices[m["uid"]],
                        targets_npy=m.get("targets_npy"))
        a["diag"] = float(np.linalg.norm(a["pos_all"].max(axis=0) - a["pos_all"].min(axis=0)))
        return a

    # the validation pairs, subsampled once and kept in RAM (the big feature arrays in
    # float16); evaluate() moves one pair at a time to the device, so the validation set
    # costs no device memory next to a running extraction (it did cost ~120 MB per pair)
    t0 = time.time()
    val_data = []
    for m in val:
        a = arrays(m)
        n = len(a["fp"])
        sel = rng.choice(n, min(args.val_samples, n), replace=False) if n else np.zeros(0, int)
        # the partial side keeps every sample's feature (float16) for the full -> partial
        # direction, and the subsample for the partial -> full direction
        val_data.append(dict(
            fp=torch.from_numpy(a["fp"][sel]), pos=torch.from_numpy(a["pos"][sel]),
            fp_all=torch.from_numpy(a["fp"]).half(),
            pos_p_all=torch.from_numpy(a["pos"]),
            ff_all=torch.from_numpy(a["ff_all"]).half(),
            pos_all=torch.from_numpy(a["pos_all"]), diag=a["diag"],
            diag_p=float(np.linalg.norm(a["pos"].max(axis=0) - a["pos"].min(axis=0)))
            if len(a["pos"]) else a["diag"],
            keep=a["keep_fraction"]))
    print(json.dumps(dict(val_loaded_seconds=round(time.time() - t0, 1),
                          val_keep_fraction_mean=round(float(np.mean(
                              [v["keep"] for v in val_data])), 4) if val_data else None)),
          flush=True)
    dim = int(np.load(train[0]["full_npz"])["feat"].shape[1])
    net = Adapter(dim, args.hidden, dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    correct_before = {}

    @torch.no_grad()
    def evaluate(adapt):
        """Nearest-neighbour GT accuracy in both directions, means over the validation pairs:
        p2f, every (subsampled) partial sample to its nearest full sample, error = distance of
        that sample to the partial sample's TRUE full position; f2p, every full sample that has
        a partial counterpart to its nearest partial sample (the direction of
        refine_correspondence.evaluate), error = distance between the two true positions.
        Errors as fractions of the full diagonal."""
        out = {}
        for key in ("p2f", "f2p"):
            acc1, acc5, acc2p, lost = [], [], [], []
            for vi, v in enumerate(val_data):
                if len(v["fp"]) == 0:
                    continue
                # one pair on the device at a time
                ff = v["ff_all"].to(dev).float()
                pos_all = v["pos_all"].to(dev)
                if adapt and args.both:
                    ff = net(ff)
                if key == "p2f":
                    fp = v["fp"].to(dev)
                    if adapt:
                        fp = net(fp)
                    nn = (fp @ ff.T).argmax(dim=1)
                    err = torch.linalg.norm(pos_all[nn] - v["pos"].to(dev), dim=1) / v["diag"]
                else:
                    fp_all = v["fp_all"].to(dev).float()
                    if adapt:
                        fp_all = net(fp_all)
                    pos_p_all = v["pos_p_all"].to(dev)
                    # full samples with a counterpart: those within 1 % of some partial
                    # sample's true position
                    d_full = torch.cdist(pos_all, pos_p_all)
                    dmin, _ = d_full.min(dim=1)
                    has = dmin < 0.01 * v["diag"]
                    if not bool(has.any()):
                        continue
                    nn = (ff[has] @ fp_all.T).argmax(dim=1)
                    err = torch.linalg.norm(pos_p_all[nn] - pos_all[has], dim=1)
                    err = err / v["diag"]
                err = err.cpu()
                acc1.append(float((err < 0.01).float().mean()))
                acc5.append(float((err < 0.05).float().mean()))
                # the real task's threshold: 2 % of the PARTIAL's diagonal
                ok2 = err * v["diag"] / v["diag_p"] < 0.02
                acc2p.append(float(ok2.float().mean()))
                if not adapt:
                    correct_before[(key, vi)] = ok2
                elif (key, vi) in correct_before:
                    before = correct_before[(key, vi)]
                    lost.append(float((before & ~ok2).float().sum() / max(float(before.sum()), 1)))
            out[key] = dict(acc_1pct=round(float(np.mean(acc1)), 4) if acc1 else None,
                            acc_5pct=round(float(np.mean(acc5)), 4) if acc5 else None,
                            acc_2pct_partial=round(float(np.mean(acc2p)), 4) if acc2p else None,
                            lost_share=round(float(np.mean(lost)), 4) if lost else None)
        out["acc_1pct"] = out["f2p"]["acc_1pct"]  # the headline, as in the GT tables
        return out

    val_before = evaluate(False)
    print(json.dumps(dict(epoch=0, val_before=val_before)), flush=True)
    best = None
    steps_per_epoch = int(np.ceil(len(train) / args.pairs_per_step))
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(train))
        losses, masked_band = [], []
        for step in range(steps_per_epoch):
            group = order[step * args.pairs_per_step:(step + 1) * args.pairs_per_step]
            loss = None
            for i in group:
                a = arrays(train[i])
                n = len(a["fp"])
                if n == 0:
                    continue
                sel = rng.choice(n, min(args.batch_vertices, n), replace=False)
                m = len(a["ff_all"])
                neg = rng.choice(m, min(args.negatives, m), replace=False)
                fp = torch.from_numpy(a["fp"][sel]).to(dev)
                ff = torch.from_numpy(a["ff"][sel]).to(dev)
                ff_all = torch.from_numpy(a["ff_all"][neg]).to(dev)
                pos_t = torch.from_numpy(a["pos"][sel]).float().to(dev)
                pos_all = torch.from_numpy(a["pos_all"][neg]).float().to(dev)
                g = net(fp)
                if args.both:  # the targets come from the same net but do not move with it
                    with torch.no_grad():
                        ff, ff_all = net(ff), net(ff_all)
                term = args.w_regress * (1.0 - (g * ff).sum(dim=1)).mean()
                if args.w_resid > 0:
                    term = term + args.w_resid * ((g - fp) ** 2).sum(dim=1).mean()
                dist_t = torch.cdist(pos_t, pos_all)  # (b, neg) true vertex to full samples
                # not negatives: full samples within 1 % of the true vertex, and far-away
                # near copies of the positive's feature (the symmetric counterpart); near
                # copies close by (2-10 %) stay as the hard negatives of localisation
                near = dist_t < 0.01 * a["diag"]
                sym = ((ff @ ff_all.T) > args.near_cos) & (dist_t > args.sym_dist * a["diag"])
                masked = near | sym
                pos_logit = (g * ff).sum(dim=1, keepdim=True) / args.temperature
                logits = g @ ff_all.T / args.temperature  # (b, neg)
                if args.w_nce > 0:
                    lg = torch.where(masked, torch.full_like(logits, -1e4), logits)
                    full = torch.cat([pos_logit, lg], dim=1)
                    term = term + args.w_nce * (-torch.log_softmax(full, dim=1)[:, 0]).mean()
                if args.w_soft > 0:
                    # a soft label over (positive, negatives): Gaussian in the distance to the
                    # true vertex, so the loss rewards candidates by how close they are
                    d = torch.cat([torch.zeros_like(pos_logit), dist_t / a["diag"]], dim=1)
                    q = torch.exp(-(d ** 2) / (2 * args.sigma ** 2))
                    q = torch.where(torch.cat([torch.zeros_like(pos_logit, dtype=torch.bool),
                                               sym], dim=1), torch.zeros_like(q), q)
                    q = q / q.sum(dim=1, keepdim=True)
                    full = torch.cat([pos_logit, logits], dim=1)
                    term = term + args.w_soft * (-(q * torch.log_softmax(full, dim=1))
                                                 .sum(dim=1)).mean()
                if args.w_nce_rev > 0:
                    # the evaluation's direction: every full target against the partial
                    # samples of the batch (its own partial sample is the positive; partial
                    # samples whose true vertex is within 1 % are not negatives)
                    rev = ff @ g.T / args.temperature  # (b, b)
                    near_p = torch.cdist(pos_t, pos_t) < 0.01 * a["diag"]
                    eye = torch.eye(len(g), dtype=torch.bool, device=dev)
                    rev = torch.where(near_p & ~eye, torch.full_like(rev, -1e4), rev)
                    term = term + args.w_nce_rev * (
                        -torch.log_softmax(rev, dim=1).diagonal()).mean()
                masked_band.append(float((masked & (dist_t > 0.02 * a["diag"])
                                          & (dist_t < 0.10 * a["diag"])).float().mean()))
                loss = term if loss is None else loss + term
            if loss is None:
                continue
            loss = loss / len(group)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
        val_acc = evaluate(True)
        print(json.dumps(dict(epoch=epoch, loss=round(float(np.mean(losses)), 4), val=val_acc,
                              masked_in_2_10pct_band=round(float(np.mean(masked_band)), 5),
                              seconds=round(time.time() - t0, 1))), flush=True)
        score = val_acc["acc_1pct"] if val_acc["acc_1pct"] is not None else -float(np.mean(losses))
        if best is None or score > best[0]:
            best = (score, epoch)
            os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
            np.savez(args.out, **net.weights(),
                     meta=json.dumps(dict(args=vars(args), epoch=epoch, val=val_acc,
                                          val_before=val_before, dim=dim)))
    beats = (best[0] is not None and val_before["acc_1pct"] is not None
             and best[0] > val_before["acc_1pct"])
    if not beats:  # the identity map is the selected model, so nothing downstream gets worse
        ident = Adapter(dim, args.hidden, dev)
        np.savez(args.out, **ident.weights(),
                 meta=json.dumps(dict(args=vars(args), epoch=0, val=val_before,
                                      val_before=val_before, dim=dim, identity=True)))
    print(json.dumps(dict(best_val_acc_1pct=best[0], best_epoch=best[1], saved=args.out,
                          beats_identity=bool(beats),
                          warning=None if beats else "no epoch beat the unadapted features; "
                          "identity weights saved")))


if __name__ == "__main__":
    main()
