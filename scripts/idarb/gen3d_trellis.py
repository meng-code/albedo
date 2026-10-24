"""
Generate a full textured 3D mesh from one photo with TRELLIS.2 (Microsoft, microsoft/TRELLIS.2-4B),
the second generator of the "generate the full mesh first, then match" route (gen3d_hunyuan.py is
the first).

TRELLIS.2 encodes the photo with DINOv3, samples a sparse voxel structure with one flow model, then
a shape latent and a texture (PBR) latent on those voxels with two more flow models, and decodes
them into a mesh with a texture. It has four resolution settings; this script uses only one
(--pipeline_type, default 512, the smallest), because the official floor is 24 GB of GPU memory
for the larger ones and our card is a 24 GB slice.

Changes to how the official example loads the pipeline, all made from outside its code:
  - only the models the chosen setting uses are loaded (the 512 setting skips the two 1024 flow
    models, 5 GB);
  - every model is moved to the GPU as soon as it is loaded and the pipeline runs with
    low_vram off: the models are built in float32 before they are cast down, and building all of
    them on the CPU first would need about 14 GB of the server's 16 GB of RAM;
  - the background-removal model (briaai/RMBG-2.0, gated, CC BY-NC 4.0) is replaced by a stub:
    our inputs are RGBA with the background removed, and TRELLIS.2 skips background removal for
    images with an alpha channel.
DINOv3 (facebook/dinov3-vitl16-pretrain-lvd1689m) is gated too and needs a Hugging Face login with
access granted.

Per input <name>.png, under --out_root/<name>/: textured.glb (o_voxel's GLB export), textured_vc.ply
with vertex colours sampled from the base-colour texture, and trellis.json (seconds, peak GPU
memory, counts).

Runs on the GPU server in the trellis2 environment (Python 3.10, torch 2.6.0+cu124). Usage:
    python scripts/idarb/gen3d_trellis.py --inputs ~/IDArb/mydata/my_single/cat.png
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODELS = {
    "512": ["sparse_structure_flow_model", "sparse_structure_decoder", "shape_slat_flow_model_512",
            "shape_slat_decoder", "tex_slat_flow_model_512", "tex_slat_decoder"],
}


def git_commit(path):
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


class NoRembg:
    """Stands in for TRELLIS.2's background-removal model; the inputs already have alpha."""

    def __init__(self, *a, **k):
        pass

    def to(self, *a, **k):
        return self

    def cpu(self):
        return self

    def __call__(self, image):
        raise RuntimeError("input without an alpha channel; remove the background first")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", required=True, help="RGBA images, background removed")
    ap.add_argument("--out_root", default=os.path.expanduser("~/semtex/gen3d/trellis2"))
    ap.add_argument("--repo", default=os.path.expanduser("~/semtex/TRELLIS.2"))
    ap.add_argument("--model", default="microsoft/TRELLIS.2-4B")
    ap.add_argument("--pipeline_type", default="512", choices=sorted(MODELS))
    ap.add_argument("--seed", type=int, default=42, help="the official default")
    ap.add_argument("--decimation_target", type=int, default=200000, help="faces in the GLB")
    ap.add_argument("--texture_size", type=int, default=2048)
    args = ap.parse_args()
    inputs = [os.path.abspath(p) for p in args.inputs]

    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    # no PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (the official example sets it): on the
    # server's vGPU slice every allocation then fails with "CUDA driver error: operation not
    # supported"
    sys.path.insert(0, args.repo)
    import o_voxel
    import torch
    import trellis2.models as tmodels
    import trimesh
    from PIL import Image
    from trellis2.pipelines import Trellis2ImageTo3DPipeline, rembg

    from gen3d_hunyuan import texture_vertex_colors

    rembg.BiRefNet = NoRembg
    Trellis2ImageTo3DPipeline.model_names_to_load = MODELS[args.pipeline_type]
    load_model = tmodels.from_pretrained

    def load_to_gpu(*a, **k):
        model = load_model(*a, **k).cuda()
        torch.cuda.empty_cache()
        return model

    tmodels.from_pretrained = load_to_gpu
    t0 = time.time()
    pipe = Trellis2ImageTo3DPipeline.from_pretrained(args.model)
    tmodels.from_pretrained = load_model
    pipe.low_vram = False
    pipe.cuda()
    print(f"pipeline ready after {time.time() - t0:.0f} s, "
          f"{torch.cuda.memory_allocated() / 2 ** 30:.1f} GB on the GPU", flush=True)

    for path in inputs:
        name = os.path.splitext(os.path.basename(path))[0]
        out = os.path.join(args.out_root, name)
        os.makedirs(out, exist_ok=True)
        image = Image.open(path).convert("RGBA")
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        mesh = pipe.run(image, seed=args.seed, pipeline_type=args.pipeline_type)[0]
        mesh.simplify(16777216)  # nvdiffrast limit, as in the official example
        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs, coords=mesh.coords,
            attr_layout=mesh.layout, voxel_size=mesh.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], decimation_target=args.decimation_target,
            texture_size=args.texture_size, remesh=True, remesh_band=1, remesh_project=0,
            verbose=False)
        glb.export(os.path.join(out, "textured.glb"))
        seconds = time.time() - t0
        tm = trimesh.load(os.path.join(out, "textured.glb"), force="mesh")
        trimesh.Trimesh(vertices=tm.vertices, faces=tm.faces,
                        vertex_colors=texture_vertex_colors(tm),
                        process=False).export(os.path.join(out, "textured_vc.ply"))
        info = dict(input=path, model=args.model, pipeline_type=args.pipeline_type,
                    seed=args.seed, seconds=round(seconds, 1),
                    peak_gpu_mib=round(torch.cuda.max_memory_allocated() / 2 ** 20),
                    n_vertices=int(len(tm.vertices)), n_faces=int(len(tm.faces)),
                    repo_commit=git_commit(args.repo), git_commit=git_commit(ROOT),
                    command=" ".join(sys.argv))
        with open(os.path.join(out, "trellis.json"), "w") as f:
            json.dump(info, f, indent=1)
        print(json.dumps(info), flush=True)
        del mesh, glb
        torch.cuda.empty_cache()


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
