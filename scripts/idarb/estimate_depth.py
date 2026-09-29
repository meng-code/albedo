"""
用单目深度估计模型 Depth Anything V2 为每张输入图生成一张相对深度图,
供 normal_to_mesh.py 的深度先验使用。

为什么需要:
    只用法线积分时,物体前后遮挡处的深度断崖会丢失,前后部件被"贴"在一起,整体看起来被压扁。
    深度模型看一张图就能粗略判断哪里近、哪里远,正好补上这部分信息。
    详见 normal_to_mesh.py 开头的说明。

在哪里运行:
    需要 PyTorch 和 transformers,在服务器的 idarb 环境里运行(显卡可选,CPU 也能跑,只是慢一些)。
    生成的深度图是普通的 PNG,拿到自己电脑上用 normal_to_mesh.py 时不需要 PyTorch。

输入和输出:
    输入是抠好背景的 RGBA 图(和交给 IDArb 的一样)。脚本先按 IDArb 推理时完全相同的方式处理
    (长边缩放到 512、补边成正方形、背景涂白),保证深度图和 IDArb 的法线图逐像素对齐。
    输出是 16 位灰度 PNG,512×512:前景内的深度归一化到 0 到 65535,越亮越靠近相机;背景为 0。
    注意这是相对深度,单位未知,normal_to_mesh.py 会把它换算到像素单位。

模型:
    默认用 Depth-Anything-V2-Small(Apache-2.0 许可证,约 100 MB)。
    --model 可以换成 Large(depth-anything/Depth-Anything-V2-Large-hf,CC-BY-NC-4.0,只能非商业使用)。
    在 5 个样本上测过,两者的深度图相关系数在 0.94 到 0.997 之间,重建结果几乎一样。

用法(在仓库根目录):
    python scripts/idarb/estimate_depth.py \\
        --input_dir results/idarb_single/inputs --out_dir results/idarb_single/depth
"""

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normal_to_mesh import load_input_512  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="抠好背景的 RGBA 图所在文件夹")
    ap.add_argument("--out_dir", required=True, help="深度图保存到哪里")
    ap.add_argument("--model", default="depth-anything/Depth-Anything-V2-Small-hf")
    args = ap.parse_args()

    names = sorted(f for f in os.listdir(args.input_dir) if f.lower().endswith(".png"))
    if not names:
        print(f"在 {args.input_dir} 里没有找到 PNG 图片")
        return
    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModelForDepthEstimation.from_pretrained(args.model).to(device).eval()
    print(f"模型 {args.model},运行在 {device} 上,共 {len(names)} 张图")

    for i, name in enumerate(names):
        rgb, mask = load_input_512(os.path.join(args.input_dir, name))
        image = Image.fromarray((rgb * 255).round().astype(np.uint8))
        with torch.no_grad():
            inputs = processor(images=image, return_tensors="pt").to(device)
            pred = model(**inputs).predicted_depth  # [1, h, w],越大越靠近相机
            pred = torch.nn.functional.interpolate(
                pred[:, None], size=mask.shape, mode="bicubic", align_corners=False
            )[0, 0]
        d = pred.float().cpu().numpy()
        lo, hi = d[mask].min(), d[mask].max()
        d = np.clip((d - lo) / max(hi - lo, 1e-8), 0.0, 1.0)
        d[~mask] = 0.0
        out = (d * 65535).round().astype(np.uint16)
        Image.fromarray(out).save(os.path.join(args.out_dir, name))
        print(f"  [{i + 1}/{len(names)}] {name}")

    print(f"全部完成,深度图在 {args.out_dir}")


if __name__ == "__main__":
    main()
