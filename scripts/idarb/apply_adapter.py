"""
Apply a trained adapter (train_adapter.py weights npz) to feature files, without torch.

Every <name>.npz of --feature_dir is copied to --out_dir with feat replaced by the adapted,
renormalised features (float16); by default only the partial files (*_partial.npz and the
dataset's *_v<k>.npz) are adapted and the full files are copied as they are, because the
adapter maps the relief side into the full side's feature space; --both adapts every file.
The meta of each file records the adapter and the commit.

Usage (repository root, mesh environment):
    python scripts/idarb/apply_adapter.py --adapter results/diff3f/adapter/adapter.npz \
        --feature_dir outputs/diff3f/features_zoom2 --out_dir outputs/diff3f/features_adapted
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adapter_data import mlp_forward  # noqa: E402


def is_partial(name):
    stem = os.path.splitext(os.path.basename(name))[0]
    return stem.endswith("_partial") or ("_v" in stem and stem.rsplit("_v", 1)[1].isdigit())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--feature_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--both", action="store_true", help="adapt the full files too")
    args = ap.parse_args()
    w = dict(np.load(args.adapter, allow_pickle=False))
    weights = {k: w[k] for k in ("ln_w", "ln_b", "w1", "b1", "w2", "b2")}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                         text=True).strip()
    except Exception:  # noqa: BLE001
        commit = None
    os.makedirs(args.out_dir, exist_ok=True)
    done = []
    for path in sorted(glob.glob(os.path.join(args.feature_dir, "*.npz"))):
        out = os.path.join(args.out_dir, os.path.basename(path))
        if not (args.both or is_partial(path)):
            shutil.copyfile(path, out)
            done.append((os.path.basename(path), "copied"))
            continue
        z = dict(np.load(path, allow_pickle=False))
        feat = mlp_forward(z["feat"].astype(np.float32), weights).astype(np.float16)
        meta = {}
        if "meta" in z:
            try:
                meta = json.loads(str(z["meta"]))
            except ValueError:
                meta = {"original_meta": str(z["meta"])}
        meta["adapter"] = dict(path=args.adapter, commit=commit, source=path)
        z["feat"] = feat
        z["meta"] = json.dumps(meta)
        np.savez_compressed(out, **z)
        done.append((os.path.basename(path), "adapted"))
    print(json.dumps(dict(files=len(done), adapted=sum(1 for _, s in done if s == "adapted"),
                          out_dir=args.out_dir)))


if __name__ == "__main__":
    main()
