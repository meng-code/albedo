"""
Renders of a generated mesh (gen3d_hunyuan.py, gen3d_trellis.py) with its vertex colours, next to
the input photo: from the input camera's side and from behind, as a quick visual check of the
generated full mesh, plus an estimate of where the input camera is in the mesh's frame.

The two generators use different frames (checked with the six axis views next to the photos):
  - Hunyuan3D-2.1 aligns the mesh with the input photo: the camera is on +z looking towards -z,
    +x to the right and +y up, like the relief meshes of normal_to_mesh.py.
  - TRELLIS.2 puts the object in its own canonical pose: the object's front faces +z and +y is
    up, whatever the photo's viewpoint, so a side-view cat comes out facing +z and the photo's
    camera is somewhere around the y axis (on -x for the cat, on +x for the okapi).
So the input camera is estimated per mesh (--fit_view, the default): the mesh's silhouette is
rendered from cameras every --azim_step degrees around the y axis at elevations --elevs, cropped
to its bounding box and resized to the photo's alpha mask cropped the same way, and the camera
with the highest intersection over union (IoU) is taken. A photo can look the same from two
mirrored directions (front and back of a left-right symmetric object); when the best camera and
the best camera in the opposite half differ by less than --colour_margin in IoU, the one whose
render is closer in colour to the photo is taken. That colour test is weak (shading and the
small size of a face), so for Hunyuan3D, whose meshes are roughly aligned with the photo, the
search is restricted to the front half (--azim_range -90 90), which removes the ambiguity.
Renders: the fitted input view and the opposite view (azimuth + 180). --no_fit_view renders
azimuth 0 and 180 instead.

Cameras are PyTorch3D look-at cameras at Diff3F's distance (0.65 times the bounding-box diagonal
from the centre); azimuth 0, elevation 0 is the camera on +z.

Writes --out (png: photo | input view | back) and <out without .png>_view.json (azimuth,
elevation and IoU of the chosen camera, the best IoUs of both halves, and the colour errors when
they were compared).

Runs on the GPU server in the diff3f environment (PyTorch3D, trimesh). Usage:
    python scripts/idarb/gen3d_render.py --mesh ~/semtex/gen3d/hunyuan3d/cat/textured_vc.ply \
        --image ~/IDArb/mydata/my_single/cat.png --out ~/semtex/gen3d/hunyuan3d/cat/renders.png
"""

import argparse
import json
import os

import numpy as np


def crop_mask(mask, size=128):
    """A binary mask cropped to its bounding box, padded to a square and resized to size x size."""
    from PIL import Image

    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return np.zeros((size, size), bool)
    m = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = m.shape
    side = max(h, w)
    sq = np.zeros((side, side), bool)
    sq[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = m
    return np.asarray(Image.fromarray(sq.astype(np.uint8) * 255).resize((size, size))) > 127


def crop_rgb(rgb, mask, size=128):
    """An RGB image (white outside mask) cropped and resized like crop_mask."""
    from PIL import Image

    ys, xs = np.nonzero(mask)
    img = np.where(mask[..., None], rgb, 255).astype(np.uint8)[ys.min():ys.max() + 1,
                                                                 xs.min():xs.max() + 1]
    h, w = img.shape[:2]
    side = max(h, w)
    sq = np.full((side, side, 3), 255, np.uint8)
    sq[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = img
    return np.asarray(Image.fromarray(sq).resize((size, size)), np.float32)


def colour_error(render_rgba, photo_rgb, photo_mask):
    """Mean absolute RGB difference (0-255) between a render and the photo, both cropped to their
    own masks, over the pixels inside either mask."""
    rmask = render_rgba[..., 3] > 127
    a = crop_rgb(render_rgba[..., :3], rmask)
    b = crop_rgb(photo_rgb, photo_mask)
    union = crop_mask(rmask) | crop_mask(photo_mask)
    return float(np.abs(a - b)[union].mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mesh", required=True, help="mesh with vertex colours (ply)")
    ap.add_argument("--image", required=True, help="the input photo (RGBA)")
    ap.add_argument("--out", required=True, help="png: photo | input view | back")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--no_fit_view", action="store_true", help="render azimuth 0 and 180")
    ap.add_argument("--azim_step", type=float, default=10.0)
    ap.add_argument("--azim_range", type=float, nargs=2, default=(0.0, 360.0),
                    help="azimuths searched, degrees (Hunyuan3D meshes: -90 90)")
    ap.add_argument("--elevs", type=float, nargs="+", default=(-10.0, 0.0, 10.0, 20.0, 30.0))
    ap.add_argument("--colour_margin", type=float, default=0.05,
                    help="decide front against back by colour when their IoUs differ by less")
    args = ap.parse_args()

    import torch
    import trimesh
    from PIL import Image
    from pytorch3d.renderer import (
        HardPhongShader,
        MeshRasterizer,
        MeshRenderer,
        PerspectiveCameras,
        PointLights,
        RasterizationSettings,
        TexturesVertex,
        look_at_view_transform,
    )
    from pytorch3d.structures import Meshes

    device = torch.device("cuda:0")
    m = trimesh.load(args.mesh, process=False, force="mesh")
    v = torch.as_tensor(np.asarray(m.vertices), dtype=torch.float32)
    f = torch.as_tensor(np.asarray(m.faces), dtype=torch.int64)
    rgb = torch.as_tensor(np.asarray(m.visual.vertex_colors)[:, :3], dtype=torch.float32) / 255
    mesh = Meshes(verts=[v], faces=[f], textures=TexturesVertex(verts_features=rgb[None]))
    mesh = mesh.to(device)
    lo, hi = v.min(0).values, v.max(0).values
    at, dist = ((lo + hi) / 2)[None], float((hi - lo).norm()) * 0.65

    def cameras(elev, azim):
        R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim, at=at, device=device)
        return PerspectiveCameras(R=R, T=T, device=device)

    def render_rgb(cams, size):
        """RGBA uint8 renders (alpha from the rasterised coverage) for a batch of cameras."""
        lights = PointLights(diffuse_color=((0.4, 0.4, 0.4),), ambient_color=((0.7, 0.7, 0.7),),
                             specular_color=((0.0, 0.0, 0.0),),
                             location=cams.get_camera_center(), device=device)
        raster = MeshRasterizer(cameras=cams, raster_settings=RasterizationSettings(
            image_size=size, blur_radius=0.0, faces_per_pixel=1, bin_size=0))
        out = MeshRenderer(raster, HardPhongShader(device=device, cameras=cams,
                                                   lights=lights))(mesh.extend(len(cams)))
        return (out.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

    photo_rgba = Image.open(args.image).convert("RGBA")
    info = dict(mesh=args.mesh, image=args.image)
    if args.no_fit_view:
        azim, elev = 0.0, 0.0
    else:
        target = crop_mask(np.asarray(photo_rgba)[:, :, 3] > 127)
        azims = np.arange(args.azim_range[0], args.azim_range[1], args.azim_step) % 360.0
        grid = [(e, float(a)) for e in args.elevs for a in azims]
        ious = []
        for i in range(0, len(grid), 36):
            chunk = grid[i:i + 36]
            cams = cameras([e for e, _ in chunk], [a for _, a in chunk])
            raster = MeshRasterizer(cameras=cams, raster_settings=RasterizationSettings(
                image_size=256, blur_radius=0.0, faces_per_pixel=1, bin_size=0))
            masks = (raster(mesh.extend(len(chunk))).pix_to_face[..., 0] >= 0).cpu().numpy()
            for mk in masks:
                c = crop_mask(mk)
                ious.append(float((c & target).sum() / max(1, (c | target).sum())))
        best = int(np.argmax(ious))
        opp = [k for k, (e, a) in enumerate(grid)
               if abs(((a - grid[best][1] + 180) % 360) - 180) > 90]
        info.update(iou_best=round(ious[best], 4))
        best_opp = max(opp, key=lambda k: ious[k]) if opp else None
        if best_opp is not None:
            info.update(iou_best_opposite_half=round(ious[best_opp], 4))
        if best_opp is not None and ious[best] - ious[best_opp] < args.colour_margin:
            # front and back of a left-right symmetric object have nearly the same silhouette;
            # decide by the colours inside the photo's mask (the face is on one side only)
            photo_rgb = np.asarray(photo_rgba.convert("RGB"))
            photo_mask = np.asarray(photo_rgba)[:, :, 3] > 127
            errors = []
            for k in (best, best_opp):
                im = render_rgb(cameras([grid[k][0]], [grid[k][1]]), 256)[0]
                errors.append(colour_error(im, photo_rgb, photo_mask))
            info.update(colour_error_best=round(errors[0], 2),
                        colour_error_opposite=round(errors[1], 2))
            if errors[1] < errors[0]:
                best = best_opp
        elev, azim = grid[best]
        info.update(azimuth=float(azim), elevation=float(elev), iou=round(ious[best], 4))
    images = render_rgb(cameras([elev, -elev], [azim, azim + 180.0]), args.size)
    views = [images[k, :, :, :3] for k in range(2)]
    bg = Image.new("RGBA", photo_rgba.size, (255, 255, 255, 255))
    photo = Image.alpha_composite(bg, photo_rgba).convert("RGB")
    photo.thumbnail((args.size, args.size))
    canvas = Image.new("RGB", (3 * args.size, args.size), (255, 255, 255))
    canvas.paste(photo, ((args.size - photo.width) // 2, (args.size - photo.height) // 2))
    for k, im in enumerate(views):
        canvas.paste(Image.fromarray(im), ((k + 1) * args.size, 0))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    canvas.save(args.out)
    with open(args.out[:-len(".png")] + "_view.json", "w") as fh:
        json.dump(info, fh, indent=1)
    print(json.dumps(info))


if __name__ == "__main__":
    main()
