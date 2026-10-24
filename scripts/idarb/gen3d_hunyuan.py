"""
Generate a full textured 3D mesh from one photo with Hunyuan3D-2.1 (Tencent), for the
"generate the full mesh first, then match" route: the generated mesh becomes the source of a
full-to-full correspondence instead of the one-sided relief.

Hunyuan3D-2.1 works in two stages with two separate models. The shape model (a diffusion
transformer over a 3D latent, decoded by marching cubes) turns the photo into an untextured mesh;
the paint model renders that mesh's normal and position maps from six cameras, lets a multiview
diffusion model paint albedo and metallic-roughness images that match the photo, and bakes them
into a texture. The paint model needs about 21 GB of GPU memory and the shape model about 10 GB,
so on our 24 GB card they run as two passes, never together: --stage shape for all inputs, then
--stage paint for all inputs.

Per input <name>.png (RGBA, background already removed), under --out_root/<name>/:
  shape stage  shape.glb (the shape model's mesh) and shape.json (seconds, peak GPU memory)
  paint stage  textured.obj with its texture maps (the paint model remeshes to about 40 k faces
               first, so this is not the same mesh as shape.glb), textured.glb converted from it
               with trimesh, textured_vc.ply with vertex colours sampled from the albedo texture
               for our pipeline, and paint.json
The official code converts OBJ to GLB with Blender's bpy, which no longer installs on Python 3.10
(bpy 4.0 is gone from PyPI and later versions need 3.11); a stub module stands in for the import
and the conversion is done here with trimesh instead.

Runs on the GPU server in the hunyuan3d environment (Python 3.10, torch 2.5.1+cu124). Set
LD_LIBRARY_PATH to the environment's lib folder so open3d finds libGL, and HY3DGEN_MODELS to a
folder whose tencent/Hunyuan3D-2.1 points at the Hugging Face snapshot, or the shape model is
downloaded a second time into ~/.cache/hy3dgen. Usage:
    python scripts/idarb/gen3d_hunyuan.py --stage shape --inputs ~/IDArb/mydata/my_single/cat.png
    python scripts/idarb/gen3d_hunyuan.py --stage paint --inputs ~/IDArb/mydata/my_single/cat.png
"""

import argparse
import json
import os
import subprocess
import sys
import time
import types

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def git_commit(path):
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def texture_vertex_colors(mesh):
    """RGB per vertex, sampled from the mesh's base-colour texture at the vertex UVs."""
    import trimesh

    vis = mesh.visual
    mat = getattr(vis, "material", None)
    img = getattr(mat, "baseColorTexture", None)
    if img is None:
        img = getattr(mat, "image", None)
    uv = getattr(vis, "uv", None)
    if img is None or uv is None:
        return np.full((len(mesh.vertices), 3), 180, np.uint8)
    return trimesh.visual.uv_to_color(uv, img)[:, :3]


def run_shape(args, inputs):
    import torch
    from PIL import Image

    sys.path.insert(0, os.path.join(args.repo, "hy3dshape"))
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

    # The server has 16 GB of RAM. Loaded as upstream does, the 7.4 GB checkpoint is read into
    # memory and the 3.3 B-parameter model is first built in float32 (13 GB), which gets the
    # process killed. Memory-mapping the checkpoint and building the modules in float16 (the
    # dtype the pipeline runs in anyway) keeps loading at about 7 GB.
    torch_load = torch.load

    def mmap_load(*a, **k):
        k.setdefault("mmap", True)
        return torch_load(*a, **k)

    torch.load = mmap_load
    torch.set_default_dtype(torch.float16)
    try:
        pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(args.model)
    finally:
        torch.set_default_dtype(torch.float32)
        torch.load = torch_load
    for path in inputs:
        name = os.path.splitext(os.path.basename(path))[0]
        out = os.path.join(args.out_root, name)
        os.makedirs(out, exist_ok=True)
        image = Image.open(path).convert("RGBA")
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        mesh = pipe(image=image, num_inference_steps=args.steps,
                    octree_resolution=args.octree_resolution,
                    generator=torch.manual_seed(args.seed))[0]
        seconds = time.time() - t0
        mesh.export(os.path.join(out, "shape.glb"))
        info = dict(stage="shape", input=path, model=args.model, seed=args.seed, steps=args.steps,
                    octree_resolution=args.octree_resolution, seconds=round(seconds, 1),
                    peak_gpu_mib=round(torch.cuda.max_memory_allocated() / 2 ** 20),
                    n_vertices=int(len(mesh.vertices)), n_faces=int(len(mesh.faces)),
                    repo_commit=git_commit(args.repo), git_commit=git_commit(ROOT),
                    command=" ".join(sys.argv))
        with open(os.path.join(out, "shape.json"), "w") as f:
            json.dump(info, f, indent=1)
        print(json.dumps(info), flush=True)


def run_paint(args, inputs):
    import torch
    import trimesh

    sys.modules.setdefault("bpy", types.ModuleType("bpy"))  # only used for OBJ -> GLB
    sys.path.insert(0, os.path.join(args.repo, "hy3dpaint"))
    sys.path.insert(0, args.repo)
    # the upscaler's basicsr imports torchvision.transforms.functional_tensor, which torchvision
    # 0.20 no longer has; the repository's own fix (used by its demo.py) provides a stand-in
    from torchvision_fix import apply_fix

    apply_fix()
    from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline

    conf = Hunyuan3DPaintConfig(args.max_num_view, args.resolution)
    conf.realesrgan_ckpt_path = os.path.join(args.repo, "hy3dpaint/ckpt/RealESRGAN_x4plus.pth")
    conf.multiview_cfg_path = os.path.join(args.repo, "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml")
    conf.custom_pipeline = os.path.join(args.repo, "hy3dpaint/hunyuanpaintpbr")
    pipe = Hunyuan3DPaintPipeline(conf)
    for path in inputs:
        name = os.path.splitext(os.path.basename(path))[0]
        out = os.path.join(args.out_root, name)
        torch.manual_seed(args.seed)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        obj = pipe(mesh_path=os.path.join(out, "shape.glb"), image_path=path,
                   output_mesh_path=os.path.join(out, "textured.obj"), save_glb=False)
        seconds = time.time() - t0
        mesh = trimesh.load(obj, force="mesh")
        mesh.export(os.path.join(out, "textured.glb"))
        colors = texture_vertex_colors(mesh)
        trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, vertex_colors=colors,
                        process=False).export(os.path.join(out, "textured_vc.ply"))
        info = dict(stage="paint", input=path, max_num_view=args.max_num_view,
                    resolution=args.resolution, seed=args.seed, seconds=round(seconds, 1),
                    peak_gpu_mib=round(torch.cuda.max_memory_allocated() / 2 ** 20),
                    n_vertices=int(len(mesh.vertices)), n_faces=int(len(mesh.faces)),
                    repo_commit=git_commit(args.repo), git_commit=git_commit(ROOT),
                    command=" ".join(sys.argv))
        with open(os.path.join(out, "paint.json"), "w") as f:
            json.dump(info, f, indent=1)
        print(json.dumps(info), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["shape", "paint"])
    ap.add_argument("--inputs", nargs="+", required=True, help="RGBA images, background removed")
    ap.add_argument("--out_root", default=os.path.expanduser("~/semtex/gen3d/hunyuan3d"))
    ap.add_argument("--repo", default=os.path.expanduser("~/semtex/Hunyuan3D-2.1"))
    ap.add_argument("--model", default="tencent/Hunyuan3D-2.1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=50, help="shape diffusion steps (default 50)")
    ap.add_argument("--octree_resolution", type=int, default=384)
    ap.add_argument("--max_num_view", type=int, default=6, help="paint views, 6 to 9")
    ap.add_argument("--resolution", type=int, default=512, help="paint view size, 512 or 768")
    args = ap.parse_args()
    inputs = [os.path.abspath(p) for p in args.inputs]
    os.chdir(args.repo)  # the paint code reads some of its files relative to the repository
    (run_shape if args.stage == "shape" else run_paint)(args, inputs)


if __name__ == "__main__":
    main()
