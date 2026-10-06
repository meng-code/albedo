"""
画失败案例图:每个物体一行,从左到右依次是原图、IDArb 四联图(albedo、normal、metallic、roughness)、
只用法线重建的网格、加深度先验重建的网格。网格用和 normal_to_mesh.py 完全相同的函数和默认参数重建,
从正侧面(绕竖直轴转 90 度)渲染,用来看物体有没有厚度。

同时在终端打印每个物体的几个诊断数字:
  平均 nz          法线 z 分量(指向相机的分量)的平均值,1 表示整片正对相机
  nz < 0.9 占比    法线明显倾斜的像素所占的比例
  s                深度先验的换算比例(像素),由法线坡度和深度图对齐估计
  深度范围         重建出的深度图最远到最近相差多少像素(只用法线 / 加深度先验)
  宽度             物体在图片里的宽度(像素)

用法(在仓库根目录运行,需要 normal_to_mesh.py 的环境):
    python scripts/idarb/plot_failure_cases.py --names cat2 cat3 cat4 \
        --out results/analysis/failure_cats.png
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import normal_to_mesh as ntm  # noqa: E402

ROW_H = 256
YAW = 90


def reconstruct(name, root, erode_px=2):
    """按 normal_to_mesh.process_one 的默认参数重建两种网格,返回渲染图和诊断数字。"""
    albedo, normal_rgb, _, _ = ntm.load_result(
        os.path.join(root, "decomposition", name + "_00.png")
    )
    size = normal_rgb.shape[:2]
    mask = ntm.erode(ntm.load_mask(os.path.join(root, "inputs", name + ".png"), size), erode_px)
    n = ntm.decode_normal(normal_rgb)
    d = ntm.load_depth(os.path.join(root, "depth", name + ".png"), size)

    h_n = ntm.integrate_normals(n, mask)
    h_d, info = ntm.integrate_normals_with_depth(n, d, mask)

    renders = []
    for h, cut in ((h_n, None), (h_d, info)):
        verts, faces, vcol = ntm.heightmap_to_mesh(h, mask, albedo)
        if cut is not None:
            faces = ntm.drop_cut_faces(faces, cut["cut_i"], cut["cut_j"], len(verts))
        verts = verts - verts.mean(axis=0)
        renders.append(ntm.render_fast(verts, faces, vcol, YAW, res=ROW_H))

    nz = n[..., 2][mask]
    cols = np.nonzero(mask.any(axis=0))[0]
    stats = dict(
        nz_mean=float(nz.mean()),
        tilted=float((nz < 0.9).mean()),
        s=info["scale"],
        range_n=float(np.nanmax(h_n) - np.nanmin(h_n)),
        range_d=float(np.nanmax(h_d) - np.nanmin(h_d)),
        width=int(cols.max() - cols.min()),
    )
    return renders, stats


def input_on_white(path):
    """原图按 IDArb 推理时的方式缩放、补边、背景涂白(和四联图逐像素对齐)。"""
    rgb, _ = ntm.load_input_512(path)
    return Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", default=["cat2", "cat3", "cat4"])
    ap.add_argument("--root", default="results/idarb_single")
    ap.add_argument("--out", default="results/analysis/failure_cats.png")
    args = ap.parse_args()

    headers = [
        "input",
        "IDArb: albedo | normal | metallic | roughness",
        "mesh, normals only (side view)",
        "mesh, normals + depth prior (side view)",
    ]
    widths = [ROW_H, 4 * ROW_H, ROW_H, ROW_H]
    gap, top, left = 6, 24, 0
    W = sum(widths) + gap * (len(widths) - 1)
    rows = []
    print("物体     平均nz  nz<0.9   s(像素)  深度范围:只用法线  加深度先验  宽度")
    for name in args.names:
        renders, st = reconstruct(name, args.root)
        print(
            "{:8s} {:6.3f} {:6.1f}% {:8.1f} {:16.1f} {:11.1f} {:5d}".format(
                name,
                st["nz_mean"],
                100 * st["tilted"],
                st["s"],
                st["range_n"],
                st["range_d"],
                st["width"],
            )
        )
        decomp = Image.open(os.path.join(args.root, "decomposition", name + "_00.png")).convert(
            "RGB"
        )
        tiles = [
            input_on_white(os.path.join(args.root, "inputs", name + ".png")).resize((ROW_H, ROW_H)),
            decomp.resize((4 * ROW_H, ROW_H)),
            Image.fromarray(renders[0]),
            Image.fromarray(renders[1]),
        ]
        label = "{}   mean nz {:.3f}   s = {:.1f} px   depth range {:.1f} / {:.1f} px".format(
            name, st["nz_mean"], st["s"], st["range_n"], st["range_d"]
        )
        rows.append((tiles, label))

    row_total = ROW_H + 20
    canvas = Image.new("RGB", (W, top + len(rows) * (row_total + gap)), (255, 255, 255))
    dr = ImageDraw.Draw(canvas)
    x = left
    for i, head in enumerate(headers):
        dr.text((x + 4, 6), head, fill=(0, 0, 0))
        x += widths[i] + gap
    for r, (tiles, label) in enumerate(rows):
        y = top + r * (row_total + gap)
        dr.text((4, y + 2), label, fill=(160, 0, 0))
        x = left
        for i, tile in enumerate(tiles):
            canvas.paste(tile, (x, y + 20))
            x += widths[i] + gap
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    canvas.save(args.out)
    print("已保存", args.out)


if __name__ == "__main__":
    main()
