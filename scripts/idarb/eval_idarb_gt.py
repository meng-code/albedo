"""
Compare IDArb's predicted normals with ground-truth normals on Arb-Objaverse objects.

This is the evaluation behind the report's numbers on normal accuracy (first run on 2026-09-25):
8 objects x 2 views (view 00 and 06), single-view inference with the released IDArb weights.
For each image it reports
  - angle_deg: mean angle between predicted and ground-truth normals inside the object mask
    (mask shrunk by 2 px, as in normal_to_mesh.py);
  - sil_nz_gt / sil_nz_pred: median n_z on the outline ring (the 3 outermost pixels of the mask),
    where n_z is the normal component towards the camera;
  - depth_gt_px / depth_pred_px: depth range obtained by integrating the ground-truth or predicted
    normals with normal_to_mesh.integrate_normals (same mask, default parameters);
  - depth_ratio: depth_pred_px / depth_gt_px.
The summary line gives the mean angle, the mean over images of the outline medians, and the median
and range of depth_ratio, which is how the report states them.

The 8 objects come from IDArb's own training list (datalist/train.json.gz, "000-000" Arb-Objaverse
pack): random.seed(0), 20 objects sampled, the 3 objects used for the training smoke test removed,
the first 8 kept. They are listed in OBJECTS below. Because they are training objects, the numbers
are optimistic. fetch_arbobjaverse_subset.py downloads objects from the same pack.

Needs the GPU server and the idarb environment. Run from the repository root:
    python scripts/idarb/eval_idarb_gt.py --data_dir ~/IDArb/data/Arbobjaverse

Writes results/analysis/idarb_normals_vs_gt.csv.
"""

import argparse
import csv
import json
import os
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
IDARB = os.path.join(HERE, "..", "..", "third_party", "IDArb")
sys.path.insert(0, HERE)
sys.path.insert(0, IDARB)
from idarbdiffusion.data.custom_dataset import CustomDataset  # noqa: E402
from idarbdiffusion.data.mv_dataset import MVDataset  # noqa: E402

import normal_to_mesh as ntm  # noqa: E402

OBJECTS = [
    "000-000/2e097b7943484d5d886a186eebbcc680",
    "000-000/ebeab69af57442048e64cecee41465a7",
    "000-000/3f6379b182de4aed9eb20b3641281b82",
    "000-000/00e1541b9c564305991941ea67596264",
    "000-000/04d9983ba04242fe99182d4506ea0714",
    "000-000/647872b89d084f7cb50329a6f7e35721",
    "000-000/5a3b7ba7b639427ea970ccca90553aab",
    "000-000/37f2819e82cf4cf18421a5021674cb40",
]
VIEWS = ["00", "06"]
FIELDS = [
    "object",
    "view",
    "angle_deg",
    "sil_nz_gt",
    "sil_nz_pred",
    "depth_gt_px",
    "depth_pred_px",
    "depth_ratio",
]


def load_idarb_pipeline(weights):
    """Run the top of IDArb's main.py (imports and load_pipeline) without its argument parsing."""
    with open(os.path.join(IDARB, "main.py"), encoding="utf-8") as f:
        src = f.read().split("parser = argparse")[0]
    ns = {"__name__": "idarb_main"}
    exec(compile(src, "main.py", "exec"), ns)
    pipe = ns["load_pipeline"](weights)
    pipe.unet.enable_xformers_memory_efficient_attention()
    pipe.to("cuda")  # IDArb's pipeline returns None from .to(), so keep the original object
    return pipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="folder holding the 000-000/<uid>/ objects")
    ap.add_argument("--weights", default="lizb6626/IDArb")
    ap.add_argument("--out", default="results/analysis/idarb_normals_vs_gt.csv")
    args = ap.parse_args()

    pipe = load_idarb_pipeline(args.weights)
    cds = CustomDataset.__new__(CustomDataset)  # only its load_image method is used
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump([["arbobjaverse", o] for o in OBJECTS], f)
    mv = MVDataset(
        img_wh=(512, 512),
        object_list=f.name,
        data_root={"arbobjaverse": "", "abo": "", "gobjaverse": ""},
    )
    os.remove(f.name)
    # single view, same camera term as main.py uses for one image without --cam
    pose_value = (np.log(1.5) - np.log(1.2)) / (np.log(2.2) - np.log(1.2)) * np.pi
    pose = torch.tensor([[[0, 0, pose_value, 0]]], dtype=torch.float16).cuda()
    white = np.ones(3, np.float32)

    rows = []
    for o in OBJECTS:
        for v in VIEWS:
            base = os.path.join(args.data_dir, o)
            color = os.path.join(base, f"color_0_{v}.png")
            img, alpha = cds.load_image(color, white, None, return_type="pt")
            x = img.permute(2, 0, 1)[None].half().cuda()
            g = torch.Generator("cuda").manual_seed(0)
            with torch.autocast("cuda"):
                out = pipe(
                    x,
                    torch.tensor([0, 1, 2]),
                    num_views=1,
                    cam_pose=pose,
                    height=512,
                    width=512,
                    guidance_scale=1.0,
                    output_type="pt",
                    eta=1.0,
                    generator=g,
                ).images
            pred_rgb = out[1].permute(1, 2, 0).float().cpu().numpy()
            _, a2 = mv.load_image(color, white, None)
            gt_rgb = mv.load_normal_arbo(os.path.join(base, f"normal_{v}.exr"), white, a2)
            mask = alpha[..., 0] > 0.5
            if mask.sum() < 2000:
                print(f"skipped {o}/{v}: object too small")
                continue

            n_gt, n_pred = ntm.decode_normal(gt_rgb), ntm.decode_normal(pred_rgb)
            m = ntm.erode(mask, 2)
            ring = mask & ~ntm.erode(mask, 3)
            angle = ntm.angular_error(n_gt, n_pred, m)[0]
            h_gt, h_pred = ntm.integrate_normals(n_gt, m), ntm.integrate_normals(n_pred, m)
            d_gt = float(np.nanmax(h_gt) - np.nanmin(h_gt))
            d_pred = float(np.nanmax(h_pred) - np.nanmin(h_pred))
            row = dict(
                object=o.split("/")[-1],
                view=v,
                angle_deg=round(angle, 2),
                sil_nz_gt=round(float(np.median(n_gt[ring, 2])), 3),
                sil_nz_pred=round(float(np.median(n_pred[ring, 2])), 3),
                depth_gt_px=round(d_gt, 1),
                depth_pred_px=round(d_pred, 1),
                depth_ratio=round(d_pred / d_gt, 3),
            )
            rows.append(row)
            print(", ".join(f"{k}={row[k]}" for k in FIELDS))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    ang = np.array([r["angle_deg"] for r in rows])
    sg = np.array([r["sil_nz_gt"] for r in rows])
    sp = np.array([r["sil_nz_pred"] for r in rows])
    ratio = np.array([r["depth_ratio"] for r in rows])
    print(
        f"{len(rows)} images: mean angle {ang.mean():.1f} deg; outline n_z (mean of per-image "
        f"medians) GT {sg.mean():.2f}, predicted {sp.mean():.2f}; depth ratio median "
        f"{np.median(ratio):.2f} (range {ratio.min():.2f} to {ratio.max():.2f})"
    )
    print("saved", args.out)


if __name__ == "__main__":
    main()
