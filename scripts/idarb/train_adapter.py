"""
Train the feature adapter on the partial-to-full dataset (needs torch; runs on the server).

The adapter is a residual two-layer perceptron on the 2048-d Diff3F features of the PARTIAL
mesh (the relief side), trained so that a partial vertex's adapted feature matches the Diff3F
feature of its true vertex on the FULL mesh: a cosine regression to that feature plus an
InfoNCE term against the other sampled vertices of the same full mesh (the hard negatives
being the symmetric and nearby parts). The full side is left as it is, so the adapter maps
the relief's features into the full mesh's feature space, which is where the partial-shape
shift of front-only views lives. --both adapts both sides with one shared network instead.

Evaluation after every epoch on the validation split: nearest-neighbour accuracy of the true
full vertex within 1 % / 5 % of the full diagonal (the GT measure of refine_correspondence.py),
before and after the adapter. Weights are saved as an npz that adapter_data.mlp_forward and
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
from adapter_data import dataset_pairs, mlp_forward, pair_arrays  # noqa: E402


def load_full_vertices(ply):
    from correspondence_maps import load_mesh

    return np.asarray(load_mesh(ply).vertices, dtype=np.float64)


def nn_accuracy(fp, ff_all, pos, pos_all, diag):
    """Nearest neighbour of every partial feature among the full samples: the share whose
    position is within 1 % / 5 % of the diagonal of the true position."""
    nn = (fp @ ff_all.T).argmax(axis=1)
    err = np.linalg.norm(pos_all[nn] - pos, axis=1) / diag
    return float((err < 0.01).mean()), float((err < 0.05).mean())


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
    ap.add_argument("--batch_vertices", type=int, default=4096, help="partial samples per step")
    ap.add_argument("--both", action="store_true", help="adapt the full side too (shared net)")
    ap.add_argument("--max_pairs", type=int, default=0, help="limit for quick runs (0: all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    train = dataset_pairs(args.root, args.feature_dir, "train")
    val = dataset_pairs(args.root, args.feature_dir, "val")
    if args.max_pairs:
        train, val = train[:args.max_pairs], val[:max(args.max_pairs // 4, 1)]
    print(json.dumps(dict(train_pairs=len(train), val_pairs=len(val))), flush=True)
    if not train:
        sys.exit("no training pairs with features")

    def load(pairs):
        out = []
        for m in pairs:
            full_v = load_full_vertices(m["full_ply"])
            a = pair_arrays(m["partial_npz"], m["full_npz"], m["gt_npz"], full_v)
            a["diag"] = float(np.linalg.norm(a["pos_all"].max(axis=0) - a["pos_all"].min(axis=0)))
            a["uid"] = m["uid"]
            out.append(a)
        return out

    t0 = time.time()
    train_data, val_data = load(train), load(val)
    print(json.dumps(dict(loaded_seconds=round(time.time() - t0, 1),
                          keep_fraction_mean=round(float(np.mean(
                              [a["keep_fraction"] for a in train_data])), 4))), flush=True)
    dim = train_data[0]["fp"].shape[1]
    net = Adapter(dim, args.hidden, args.device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    def evaluate(data, weights=None):
        acc1, acc5 = [], []
        for a in data:
            fp = mlp_forward(a["fp"], weights) if weights is not None else a["fp"]
            ff = mlp_forward(a["ff_all"], weights) if (weights is not None and args.both) \
                else a["ff_all"]
            r = nn_accuracy(fp, ff, a["pos"], a["pos_all"], a["diag"])
            acc1.append(r[0])
            acc5.append(r[1])
        return dict(acc_1pct=round(float(np.mean(acc1)), 4),
                    acc_5pct=round(float(np.mean(acc5)), 4))

    print(json.dumps(dict(epoch=0, val_before=evaluate(val_data))), flush=True)
    best = None
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(train_data))
        losses = []
        for i in order:
            a = train_data[i]
            n = len(a["fp"])
            if n == 0:
                continue
            sel = rng.choice(n, min(args.batch_vertices, n), replace=False)
            fp = torch.from_numpy(a["fp"][sel]).to(args.device)
            ff = torch.from_numpy(a["ff"][sel]).to(args.device)
            m = len(a["ff_all"])
            neg = rng.choice(m, min(args.negatives, m), replace=False)
            ff_all = torch.from_numpy(a["ff_all"][neg]).to(args.device)
            pos_t = torch.from_numpy(a["pos"][sel]).to(args.device)
            pos_all = torch.from_numpy(a["pos_all"][neg]).to(args.device)
            g = net(fp)
            if args.both:
                ff, ff_all = net(ff), net(ff_all)
            loss = args.w_regress * (1.0 - (g * ff).sum(dim=1)).mean()
            if args.w_nce > 0:
                logits = g @ ff_all.T / args.temperature  # (b, neg)
                # the positive is the true vertex itself; full samples within 1 % of it are not
                # negatives (they would punish a correct neighbour)
                near = torch.cdist(pos_t, pos_all) < 0.01 * a["diag"]
                pos_logit = (g * ff).sum(dim=1, keepdim=True) / args.temperature
                logits = torch.where(near, torch.full_like(logits, -1e4), logits)
                full = torch.cat([pos_logit, logits], dim=1)
                loss = loss + args.w_nce * (-torch.log_softmax(full, dim=1)[:, 0]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
        w = net.weights()
        val_acc = evaluate(val_data, w)
        print(json.dumps(dict(epoch=epoch, loss=round(float(np.mean(losses)), 4), val=val_acc,
                              seconds=round(time.time() - t0, 1))), flush=True)
        if best is None or val_acc["acc_1pct"] > best[0]:
            best = (val_acc["acc_1pct"], epoch)
            os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
            np.savez(args.out, **w, meta=json.dumps(dict(args=vars(args), epoch=epoch,
                                                        val=val_acc, dim=dim)))
    print(json.dumps(dict(best_val_acc_1pct=best[0], best_epoch=best[1], saved=args.out)))


if __name__ == "__main__":
    main()
