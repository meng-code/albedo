"""
把你自己拍的照片处理成 IDArb 能读的格式。

为什么需要这一步:
    IDArb 的数据读取代码(custom_dataset.py 原 42 行)有一句 assert,
    要求输入必须是 RGBA 四通道的 PNG。RGBA 里的 A 是 alpha 通道,记录每个像素是物体还是背景。
    手机拍出来的照片只有 RGB 三个通道,没有 A,所以直接喂给 IDArb 会报错。
    这个脚本用 rembg 这个工具把背景识别出来,写进 alpha 通道,再存成 PNG。
    rembg 内部是一个训练好的神经网络,它判断每个像素属于前景还是背景,判断结果不一定完全准确。

怎么用(在服务器上、third_party/IDArb 目录下、idarb 环境里运行):

    单图模式。把照片放在一个文件夹里,处理后得到一堆 PNG,每张图是一个独立的样本:
        python ../../scripts/idarb/prepare_images.py --input mydata/raw --output mydata/my_single

    多视角模式。同一个物体的多张不同角度照片放在一个文件夹里,
    处理后会多生成一个 transforms.json:
        python ../../scripts/idarb/prepare_images.py \
            --input mydata/raw --output mydata/multi/myobject --multiview

    处理完之后跑推理:
        python main.py --data_dir mydata/my_single --output_dir output/my_single --input_type single
        python main.py --data_dir mydata/multi --output_dir output/my_multi --input_type multi

关于 transforms.json:
    多视角模式下,IDArb 需要一个 transforms.json 文件,
    里面按顺序列出每张照片的文件名,以及拍摄这张照片时
    相机的位置和朝向(一个 4x4 的矩阵)。手机拍照不会记录这个矩阵,要算出它需要 COLMAP 这类工具。
    这个脚本生成的 transforms.json 里,矩阵一律写成 null,表示"不知道相机在哪"。
    IDArb 支持这种情况:只要运行 main.py 时不加 --cam 参数,它就会给所有视角用同一个默认位姿,
    让模型自己从图片内容推断视角之间的关系。官方示例里的 moldGoldCape 就是这样做的。
"""

import argparse
import json
import os

from PIL import Image
from rembg import new_session, remove


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="放原始照片的文件夹")
    parser.add_argument("--output", type=str, required=True, help="处理结果存到哪个文件夹")
    parser.add_argument(
        "--multiview",
        action="store_true",
        help="加上这个参数表示这些照片是同一个物体的多个视角,会额外生成 transforms.json",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="u2net",
        help="rembg 使用的抠图模型。u2net 是通用模型,isnet-general-use 有时边缘更干净",
    )
    args = parser.parse_args()

    # 找出输入文件夹里所有的图片文件
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".tif", ".tiff")
    names = sorted(f for f in os.listdir(args.input) if f.lower().endswith(exts))
    if not names:
        print(f"在 {args.input} 里没有找到图片文件。支持的格式:{', '.join(exts)}")
        return

    os.makedirs(args.output, exist_ok=True)
    print(f"找到 {len(names)} 张图片,使用模型 {args.model}")

    session = new_session(args.model)
    saved = []

    for i, name in enumerate(names):
        src_path = os.path.join(args.input, name)
        raw = Image.open(src_path)

        # 先看这张图是不是已经带好了可用的透明通道。
        # 判断办法:图片本身是 RGBA 或 LA 模式,并且 alpha 通道里既有接近 0 的值也有接近 255 的值,
        # 说明有人已经把背景标出来了。
        # 有些文件虽然是 RGBA,但 alpha 全是 255,那等于没有信息,还要重新抠。
        # 注意扩展名不可靠:这批图里有几个文件叫 .jpg,实际是 PNG,而且带着可用的透明通道。
        reuse_alpha = False
        if raw.mode in ("RGBA", "LA"):
            a = raw.convert("RGBA").split()[3]
            lo, hi = a.getextrema()
            reuse_alpha = lo < 16 and hi > 240

        if reuse_alpha:
            out = raw.convert("RGBA")
            how = "沿用原有透明通道"
        else:
            # 转成 RGB 再抠。有些照片是 CMYK 或者带调色板的格式,统一转一下比较保险
            img = raw.convert("RGB")
            # 用 rembg 抠掉背景。返回的图片是 RGBA 四通道,背景处的 alpha 是 0
            out = remove(img, session=session)
            how = "rembg 抠图"

        # 存成 PNG。文件名去掉原来的扩展名,统一加 .png
        stem = os.path.splitext(name)[0]
        out_path = os.path.join(args.output, stem + ".png")
        out.save(out_path)
        saved.append(stem)

        # 统计一下有多大比例的像素被判定成了物体,用来粗略判断抠图效果
        alpha = out.split()[3]
        ratio = sum(1 for v in alpha.getdata() if v > 127) / (out.width * out.height)
        size = f"{out.width}x{out.height}"
        print(f"  [{i + 1}/{len(names)}] {name:<42} {size:>9}  {how:<18} 物体像素占比 {ratio:.1%}")
        if ratio < 0.02:
            print("        提示:物体占比过低,可能抠图失败了,建议打开这张图看一下")
        elif ratio > 0.95:
            print("        提示:物体占比过高,可能背景没被去掉,建议打开这张图看一下")

    # 多视角模式下额外生成 transforms.json
    if args.multiview:
        cfg = {"frames": [{"file_path": s, "transform_matrix": None} for s in saved]}
        json_path = os.path.join(args.output, "transforms.json")
        with open(json_path, "w") as f:
            json.dump(cfg, f, indent=4)
        print(f"已生成 {json_path},里面列出了 {len(saved)} 个视角,相机矩阵都是 null")
        print("跑多视角推理时不要加 --cam,因为没有真实的相机矩阵")

    print(f"全部完成,结果在 {args.output}")


if __name__ == "__main__":
    main()
