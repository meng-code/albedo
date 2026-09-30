"""
从 IDArb 预测的法线图恢复三维网格(mesh)。

一、这个脚本做什么

  IDArb 对一张图输出四张结果:albedo、normal、metallic、roughness。其中 normal(法线图)记录了物体表面
  每个像素的朝向。法线只描述"表面往哪个方向倾斜",不直接给出"表面离相机多远"(深度)。但是倾斜方向
  等价于深度的变化率(梯度),所以可以通过积分把深度恢复出来,再把深度图连成三角网格。

  脚本提供两种方法,默认两种都做:

  方法 A:泊松法线积分(Poisson normal integration)
    设深度图为 h(x, y)。法线 n = (nx, ny, nz) 和深度梯度的关系是 ∂h/∂x = -nx/nz,∂h/∂y = -ny/nz。
    我们求一个 h,使它的梯度和法线给出的梯度在最小二乘意义下最接近。把这个最小二乘问题写成线性方程组,
    系数矩阵正好是离散的拉普拉斯算子,方程右边是梯度场的散度,也就是一个泊松方程,所以叫泊松积分。
    结果是物体朝向相机那一面的表面,输出为一个开口的网格(像一张浮雕)。

  方法 B:筛选泊松表面重建(Screened Poisson Surface Reconstruction,Kazhdan & Hoppe 2013)
    这是三维重建里常说的"泊松重建"。它的输入是一组带法线的三维点;点覆盖物体各个方向时,
    输出是一个封闭的曲面。
    我们用方法 A 得到的三维点,加上 IDArb 预测的法线,交给 Open3D 里的实现。
    注意:单张图片只能看到物体的正面,背面没有任何点,所以在我们的数据上它不会补出背面,
    输出也不是封闭网格。实测(熊猫、chair1、鸟)得到的是物体正面,外加一大片从物体四周
    一直延伸到包围盒边缘的平面。脚本按点的密度把这片离观测点太远的部分裁掉,剩下的正面和
    方法 A 的表面相差不到半个像素(顶点到方法 A 的点的距离中位数约 0.45 像素),
    只是三角形的划分不同。

  方法 A 的改进:加入深度先验(可选,用 --depth_dir 打开)
    只用法线积分有一个根本问题:物体前后遮挡的地方(比如椅子座面挡在椅子腿前面),深度是断开的,
    但法线只描述表面局部的倾斜,不包含"断开了多远"。积分时这些断崖会被抹成平缓的坡,
    前面的部件被"贴"到后面的部件上,整个物体看起来被压扁;法线里的小偏差还会累积成整体倾斜。
    改进办法是再用一个单目深度估计模型(Depth Anything V2,见 estimate_depth.py)给出一张粗略的
    相对深度图 d,和法线一起求解:
        min  Σ (相邻两像素的深度差 − 法线给出的深度差)²  +  w · Σ (h − s·d)²
    法线负责表面细节,深度图负责部件之间的前后关系和整体朝向。s 是把 d 换算到像素单位的比例,
    用平滑区域里 d 的梯度和法线梯度对齐来估计。深度图认为某两个相邻像素之间有明显断崖时,
    就不再要求这两个像素的深度平滑相连,断崖的高度交给深度图决定。
    深度模型在细小结构(比如椅子腿)上估得不准,会切出很多很小的碎块,变成漂浮的碎片,
    所以小于 --min_part 像素的碎块会被接回到与它交界最多的那个相邻块上。
    生成网格时再去掉正好跨在断崖上的三角形,避免前后部件之间被连出一片竖直的"墙"。

二、坐标约定(非常重要,弄错会导致表面左右或上下翻转)

  IDArb 输出的法线图,三个颜色通道 R、G、B 分别对应法线的 x、y、z 分量,
  编码方式是 颜色 = (分量 + 1) / 2。
  它使用的坐标系是 "lub":x 轴指向图像的左边,y 轴指向图像的上方,z 轴指向相机(从屏幕里朝外)。
  依据有两条:训练代码 mv_dataset.py 里的 blender2midas 函数注释写的是 "midas: lub";
  另外用训练数据里一个方箱验证过,朝左的面红色通道高,朝右的面红色通道低。
  本脚本内部统一换成 "x 朝右、y 朝上、z 朝相机" 的坐标系,也就是把 x 分量取反。

  深度方向:h 越大表示离相机越近。相机假设为正交投影(IDArb 的训练数据也基本是这样处理的),
  所以像素坐标和深度可以用同一个单位(像素)。

三、用法

  以下命令在仓库根目录运行。示例数据在 results/idarb_single/ 里:
  decomposition/ 是 IDArb 的分解结果(四联图),inputs/ 是对应的抠图输入(RGBA)。

  处理一个物体:
    python scripts/idarb/normal_to_mesh.py \
        --result results/idarb_single/decomposition/panda_00.png \
        --input results/idarb_single/inputs/panda.png

  批量处理一个结果文件夹(输入图按同名去 --input_dir 里找):
    python scripts/idarb/normal_to_mesh.py \
        --result_dir results/idarb_single/decomposition --input_dir results/idarb_single/inputs

  加入深度先验(深度图由服务器上的 estimate_depth.py 生成,放在 results/idarb_single/depth/):
    python scripts/idarb/normal_to_mesh.py \
        --result_dir results/idarb_single/decomposition --input_dir results/idarb_single/inputs \
        --depth_dir results/idarb_single/depth --out_dir outputs/mesh_depth

  正确性测试(不需要任何输入文件):理想半球,以及有遮挡断崖的两个球
    python scripts/idarb/normal_to_mesh.py --selftest

  输出(默认在 outputs/mesh/<物体名>/ 下,这个文件夹不进 git):
    relief.ply / relief.obj   方法 A 的网格,带 albedo 颜色
    poisson.ply               方法 B 的网格(只有正面,和 relief 几乎相同)
    depth.png                 恢复出的深度图(越亮越靠近相机)
    preview.png               从几个角度渲染的预览图,用来检查三维形状
"""

import argparse
import os
import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from PIL import Image, ImageDraw
from scipy.sparse.csgraph import connected_components

# ------------------------------------------------------------------------------
# 读取数据
# ------------------------------------------------------------------------------


def load_result(result_path):
    """读取 main.py 保存的四联图,拆成 albedo、normal、metallic、roughness 四张,数值范围 0 到 1。"""
    img = np.asarray(Image.open(result_path).convert("RGB")).astype(np.float32) / 255.0
    w = img.shape[1] // 4
    albedo, normal, metallic, roughness = [img[:, i * w : (i + 1) * w] for i in range(4)]
    return albedo, normal, metallic, roughness


def load_input_512(input_path):
    """
    用和推理时完全相同的方式处理输入的 RGBA 图,
    返回 (白底 RGB 图 [512, 512, 3],前景掩膜 [512, 512])。
    推理时 custom_dataset.py 的 load_image 会把图片缩放到长边 512、再补边成正方形,
    所以这里也必须这样处理,才能和 IDArb 输出的像素一一对应。

    下面的步骤逐行照抄 custom_dataset.py 原 27-61 行,但只用 PIL 和 numpy,
    不导入 PyTorch 和 idarbdiffusion,
    这样在没有显卡、没装 PyTorch 的电脑上也能运行。掩膜已经和原代码逐张核对过,完全一致。
    """
    img = Image.open(input_path)
    max_length = 512
    h, w = img.size  # PIL 的 size 是 (宽, 高)。变量名沿用原代码,虽然名字是反的,但结果正确
    if h > w:
        new_h = max_length
        new_w = int(w / h * max_length)
    else:
        new_w = max_length
        new_h = int(h / w * max_length)
    img = img.resize((new_h, new_w), Image.BICUBIC)
    img = np.array(img).astype(np.float32) / 255.0
    if img.shape[-1] != 4:
        raise ValueError(f"输入必须是 RGBA 四通道的 PNG:{input_path}")
    h, w = img.shape[:2]
    pad_h, pad_w = (max_length - h) // 2, (max_length - w) // 2
    img = np.pad(
        img,
        ((pad_h, max_length - h - pad_h), (pad_w, max_length - w - pad_w), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    mask = img[:, :, 3] > 0.5
    alpha = mask[..., None].astype(np.float32)
    rgb = img[:, :, :3] * alpha + (1.0 - alpha)  # 背景涂白,和 custom_dataset.py 原 61 行一样
    return rgb, mask


def load_mask(input_path, size):
    """得到和 IDArb 输出像素一一对应的前景掩膜。"""
    _, mask = load_input_512(input_path)
    if mask.shape != size:
        raise ValueError(f"掩膜尺寸 {mask.shape} 和法线图尺寸 {size} 不一致")
    return mask


def load_depth(depth_path, size):
    """
    读取 estimate_depth.py 保存的深度图(16 位灰度 PNG),返回 0 到 1 的相对深度,越大越靠近相机。
    这是单目深度模型给出的相对值,单位未知,使用时再换算到像素单位。
    """
    d = np.asarray(Image.open(depth_path)).astype(np.float64) / 65535.0
    if d.shape != size:
        raise ValueError(f"深度图尺寸 {d.shape} 和法线图尺寸 {size} 不一致")
    return d


def decode_normal(normal_rgb):
    """
    把法线图的颜色解码成单位向量,并从 IDArb 的 lub 坐标系换成 "x 朝右、y 朝上、z 朝相机"。
    返回形状 [H, W, 3] 的数组。
    """
    n = normal_rgb * 2.0 - 1.0
    n = n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-8)
    n = n.copy()
    n[..., 0] = -n[..., 0]  # lub 的 x 朝左,取反后朝右
    return n


def erode(mask, iterations=1):
    """把掩膜向内收缩几个像素。物体边缘的法线在缩放时混进了白色背景,不可靠,积分时去掉。"""
    m = mask.copy()
    for _ in range(iterations):
        m = m & np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1)
    return m


# ------------------------------------------------------------------------------
# 方法 A:泊松法线积分
# ------------------------------------------------------------------------------


def integrate_normals(n, mask, min_nz=0.1, max_grad=8.0, reg=1e-6):
    """
    输入:n 是 [H, W, 3] 的单位法线(x 朝右、y 朝上、z 朝相机),mask 是 [H, W] 的布尔掩膜。
    输出:h 是 [H, W] 的深度图(掩膜外为 nan),单位是像素。

    做法:
      1. 由法线算出每个像素处的深度梯度。图像的行号是向下增加的,而 y 轴朝上,所以
         向右走一格深度变化 p = -nx/nz,向下走一格深度变化 q = +ny/nz。
      2. 对掩膜内每一对左右相邻的像素,列一个方程 h[右] - h[左] = 两个像素 p 的平均值;
         对每一对上下相邻的像素,列一个方程 h[下] - h[上] = 两个像素 q 的平均值。
         方程个数比未知数多,用最小二乘求解。
      3. 最小二乘的正规方程是 (AᵀA) h = Aᵀb。AᵀA 就是图上的离散拉普拉斯算子,Aᵀb 是梯度场的散度,
         所以这一步就是在解泊松方程。深度整体加上任何常数都满足方程,所以再加一个很小的正则项 reg·h,
         让解唯一。
    """
    H, W = mask.shape
    nz = np.clip(n[..., 2], min_nz, None)  # 物体边缘处 nz 接近 0,梯度会趋于无穷,这里限制一下
    p = np.clip(-n[..., 0] / nz, -max_grad, max_grad)
    q = np.clip(n[..., 1] / nz, -max_grad, max_grad)

    idx = -np.ones((H, W), dtype=np.int64)
    idx[mask] = np.arange(mask.sum())
    N = int(mask.sum())

    rows, cols, vals, b = [], [], [], []
    eq = 0
    # 左右相邻的像素对
    m_h = mask[:, :-1] & mask[:, 1:]
    r, c = np.nonzero(m_h)
    k = len(r)
    rows += [np.arange(eq, eq + k)] * 2
    cols += [idx[r, c + 1], idx[r, c]]
    vals += [np.ones(k), -np.ones(k)]
    b.append(0.5 * (p[r, c] + p[r, c + 1]))
    eq += k
    # 上下相邻的像素对
    m_v = mask[:-1, :] & mask[1:, :]
    r, c = np.nonzero(m_v)
    k = len(r)
    rows += [np.arange(eq, eq + k)] * 2
    cols += [idx[r + 1, c], idx[r, c]]
    vals += [np.ones(k), -np.ones(k)]
    b.append(0.5 * (q[r, c] + q[r + 1, c]))
    eq += k

    A = sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(eq, N)
    )
    b = np.concatenate(b)
    L = (A.T @ A + reg * sp.identity(N)).tocsc()
    x = spla.spsolve(L, A.T @ b)

    h = np.full((H, W), np.nan, dtype=np.float64)
    h[mask] = x
    h[mask] -= np.nanmin(h)  # 让最远的点深度为 0
    return h


def neighbor_pairs(mask):
    """
    列出掩膜内所有左右相邻和上下相邻的像素对。i 是左边或上边的像素,j 是右边或下边的像素。
    返回 (ri, ci, rj, cj, horizontal):两组像素的行号、列号,以及每一对是否是左右相邻。
    """
    r1, c1 = np.nonzero(mask[:, :-1] & mask[:, 1:])
    r2, c2 = np.nonzero(mask[:-1, :] & mask[1:, :])
    ri = np.concatenate([r1, r2])
    ci = np.concatenate([c1, c2])
    rj = np.concatenate([r1, r2 + 1])
    cj = np.concatenate([c1 + 1, c2])
    horizontal = np.concatenate([np.ones(len(r1), bool), np.zeros(len(r2), bool)])
    return ri, ci, rj, cj, horizontal


def merge_small_parts(jump, vi, vj, n_vertices, min_part):
    """
    断崖会把前景切成若干块。深度模型在细小结构(比如椅子腿)上估得不准,会切出很多很小的碎块,
    这些碎块的位置只能靠不准的深度值决定,就成了漂浮的碎片。
    这里把小于 min_part 像素的块接回去:每个小块只和"与它之间断崖边最多"的那一个相邻块重新连起来。
    只连一个,是为了避免一个小块同时连向两个本应分开的大块(比如超人的披风和身体),把它们粘在一起。
    反复进行,直到没有能接回去的小块为止。
    jump 是每个相邻像素对是否为断崖,vi、vj 是每对两端像素的编号。返回更新后的 jump。
    """
    jump = jump.copy()
    for _ in range(100):
        keep = ~jump
        graph = sp.coo_matrix(
            (np.ones(int(keep.sum())), (vi[keep], vj[keep])), shape=(n_vertices, n_vertices)
        )
        _, label = connected_components(graph, directed=False)
        size = np.bincount(label)
        li, lj = label[vi], label[vj]
        cand = np.nonzero(jump & (li != lj) & ((size[li] < min_part) | (size[lj] < min_part)))[0]
        if len(cand) == 0:
            break
        i_is_small = size[li[cand]] <= size[lj[cand]]
        small = np.where(i_is_small, li[cand], lj[cand])
        other = np.where(i_is_small, lj[cand], li[cand])
        base = int(label.max()) + 1
        pair = small * base + other
        uniq, count = np.unique(pair, return_counts=True)
        # 按小块编号排序,同一个小块内按交界边数从多到少排,取每个小块的第一个
        order = np.lexsort((-count, uniq // base))
        first = np.unique(uniq[order] // base, return_index=True)[1]
        chosen = uniq[order][first]
        jump[cand[np.isin(pair, chosen)]] = False
    return jump


def integrate_normals_with_depth(
    n, d, mask, weight=0.01, jump_px=3.0, min_part=300, min_nz=0.1, max_grad=8.0
):
    """
    法线和深度先验联合求解深度图(方法 A 的改进版)。
    输入:n 是单位法线(x 朝右、y 朝上、z 朝相机),d 是 0 到 1 的相对深度(越大越近),mask 是掩膜。
    输出:h 是深度图(单位是像素,掩膜外为 nan),以及一个记录中间量的字典。

    求解的问题是
        min  Σ_{保留的相邻像素对} (h_j − h_i − g_ij)²  +  weight · Σ_i (h_i − s·d_i)²
    其中 g_ij 是法线给出的深度差,和 integrate_normals 里的定义相同。

    步骤:
      1. 估计比例 s:在平滑区域(d 的相邻差值不属于最大的 10%)里,让 s·Δd 和法线给出的 Δh 最接近,
         最小二乘解是 s = Σ Δd·g / Σ Δd²。
      2. 找断崖:一对相邻像素,如果深度图说它们相差超过 jump_px 像素(|s·Δd| > jump_px),
         而且法线给出的差值和它明显对不上(|s·Δd − g| > jump_px),就认为中间是遮挡造成的断崖,
         去掉这一对的方程,不再要求它们平滑相连。
         切出来的块如果小于 min_part 像素,用 merge_small_parts 接回去。
      3. 解线性方程组 (AᵀA + weight·I) h = Aᵀg + weight·s·d。和 integrate_normals 一样是泊松方程,
         只是多了一项把 h 拉向 s·d。这一项同时决定了每一块被断崖隔开的区域的前后位置。
    """
    H, W = mask.shape
    nz = np.clip(n[..., 2], min_nz, None)
    p = np.clip(-n[..., 0] / nz, -max_grad, max_grad)  # 向右一格的深度变化
    q = np.clip(n[..., 1] / nz, -max_grad, max_grad)  # 向下一格的深度变化

    idx = -np.ones((H, W), dtype=np.int64)
    idx[mask] = np.arange(mask.sum())
    N = int(mask.sum())

    ri, ci, rj, cj, horizontal = neighbor_pairs(mask)
    g = np.where(horizontal, 0.5 * (p[ri, ci] + p[rj, cj]), 0.5 * (q[ri, ci] + q[rj, cj]))
    vi, vj = idx[ri, ci], idx[rj, cj]
    dd = d[rj, cj] - d[ri, ci]

    # 1. 比例 s
    smooth = np.abs(dd) < np.quantile(np.abs(dd), 0.9)
    s = float((dd[smooth] * g[smooth]).sum() / max((dd[smooth] ** 2).sum(), 1e-12))
    # 2. 断崖
    jump = (np.abs(s * dd) > jump_px) & (np.abs(s * dd - g) > jump_px)
    jump = merge_small_parts(jump, vi, vj, N, min_part)
    keep = ~jump
    k = int(keep.sum())
    # 3. 求解
    rows = np.concatenate([np.arange(k), np.arange(k)])
    cols = np.concatenate([vj[keep], vi[keep]])
    vals = np.concatenate([np.ones(k), -np.ones(k)])
    A = sp.csr_matrix((vals, (rows, cols)), shape=(k, N))
    rhs = A.T @ g[keep] + weight * s * d[mask]
    L = (A.T @ A + weight * sp.identity(N)).tocsc()
    x = spla.spsolve(L, rhs)

    h = np.full((H, W), np.nan, dtype=np.float64)
    h[mask] = x
    h[mask] -= np.nanmin(h)
    graph = sp.coo_matrix((np.ones(k), (vi[keep], vj[keep])), shape=(N, N))
    n_parts = connected_components(graph, directed=False)[0]
    info = dict(
        scale=s, jump_fraction=float(jump.mean()), parts=n_parts, cut_i=vi[jump], cut_j=vj[jump]
    )
    return h, info


def drop_cut_faces(faces, cut_i, cut_j, n_vertices):
    """
    去掉跨在断崖上的三角形。断崖两侧的像素在图上相邻,heightmap_to_mesh 会把它们连成三角形,
    于是前后两个部件之间被拉出一片竖直的"墙"。三角形只要有一条边正好是被切断的相邻像素对,就删掉。
    顶点编号和 integrate_normals_with_depth 里的像素编号一致,都按掩膜内像素的行优先顺序排列。
    """
    if len(cut_i) == 0:
        return faces
    cut = np.unique(np.minimum(cut_i, cut_j) * n_vertices + np.maximum(cut_i, cut_j))
    bad = np.zeros(len(faces), dtype=bool)
    for a, b in ((0, 1), (1, 2), (0, 2)):
        edge = np.minimum(faces[:, a], faces[:, b]) * n_vertices + np.maximum(
            faces[:, a], faces[:, b]
        )
        bad |= np.isin(edge, cut)
    return faces[~bad]


def normals_from_depth(h):
    """由深度图反过来算法线,用来检查积分结果和输入法线是否一致。"""
    dx = np.full_like(h, np.nan)
    dy = np.full_like(h, np.nan)
    dx[:, 1:-1] = (h[:, 2:] - h[:, :-2]) / 2.0  # 向右的变化率
    dy[1:-1, :] = (h[2:, :] - h[:-2, :]) / 2.0  # 向下的变化率
    n = np.stack([-dx, dy, np.ones_like(h)], axis=-1)  # 和 integrate_normals 里 p、q 的定义对应
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def angular_error(n1, n2, mask):
    """两组法线在掩膜内的平均夹角(单位:度)。"""
    cos = np.clip((n1 * n2).sum(-1), -1, 1)
    ang = np.degrees(np.arccos(cos))
    ang = ang[mask & np.isfinite(ang)]
    return float(ang.mean()), float(np.median(ang))


def heightmap_to_mesh(h, mask, colors):
    """
    把深度图连成三角网格。每个前景像素是一个顶点,位置是 (列号, -行号, 深度),
    再整体缩放到大约 [-1, 1]。
    对每个 2×2 的像素块:四个像素都在前景里就连两个三角形,只有三个在前景里就连一个三角形。
    三角形的顶点顺序保证从相机方向看是逆时针,这样网格的法线朝向相机。
    """
    H, W = mask.shape
    idx = -np.ones((H, W), dtype=np.int64)
    idx[mask] = np.arange(mask.sum())
    rr, cc = np.nonzero(mask)
    scale = 2.0 / max(H, W)
    verts = np.stack([(cc - W / 2) * scale, (H / 2 - rr) * scale, h[rr, cc] * scale], axis=1)
    vcol = (np.clip(colors[rr, cc], 0, 1) * 255).astype(np.uint8)

    a = idx[:-1, :-1]
    b = idx[:-1, 1:]
    c = idx[1:, :-1]
    d = idx[1:, 1:]  # 左上、右上、左下、右下
    faces = []
    full = (a >= 0) & (b >= 0) & (c >= 0) & (d >= 0)
    faces += [np.stack([a[full], c[full], d[full]], 1), np.stack([a[full], d[full], b[full]], 1)]
    # 只有三个角在前景里的块,补一个三角形,让边缘更完整
    for m, tri in [
        ((a < 0) & (b >= 0) & (c >= 0) & (d >= 0), (b, c, d)),
        ((b < 0) & (a >= 0) & (c >= 0) & (d >= 0), (a, c, d)),
        ((c < 0) & (a >= 0) & (b >= 0) & (d >= 0), (a, d, b)),
        ((d < 0) & (a >= 0) & (b >= 0) & (c >= 0), (a, c, b)),
    ]:
        faces.append(np.stack([t[m] for t in tri], 1))
    faces = np.concatenate(faces, 0)
    return verts, faces, vcol


# ------------------------------------------------------------------------------
# 方法 B:筛选泊松表面重建
# ------------------------------------------------------------------------------


def screened_poisson(verts, normals, colors, depth=9, trim_quantile=0.05):
    """
    verts:[N, 3] 三维点;normals:[N, 3] 每个点的法线(网络预测的,已换到同一坐标系);
    colors:[N, 3] 0~255。
    depth:八叉树深度,越大细节越多、越慢,9 对 512×512 的输入比较合适。
    trim_quantile:重建完成后,把"支撑点密度"最低的这部分顶点删掉。密度低说明附近没有观测点,
                   这些面是算法凭空推出来的。在单张图片上,主要是从物体四周延伸到包围盒边缘的那片平面。
    """
    # open3d 启动时需要 libGL.so.1。集群系统里没有这个库,我们用 conda 把它装进了环境自己的 lib 目录
    # (conda install -n idarb -c conda-forge libgl),但 open3d 不会自动去那里找,
    # 所以这里先手动加载一次。在装有 libGL 的电脑上(包括 Mac、Windows)这个文件不存在,会直接跳过。
    import ctypes

    libgl = os.path.join(sys.prefix, "lib", "libGL.so.1")
    if os.path.exists(libgl):
        ctypes.CDLL(libgl, mode=ctypes.RTLD_GLOBAL)
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts.astype(np.float64))
    pcd.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
    # n_threads=1:只用一个线程。默认的多线程在 Mac(M 系列芯片)上经常报错 "Failed to close loop"
    # 并直接退出,单线程则每次都成功,而且同样的输入每次得到完全相同的网格。代价是慢几秒。
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, n_threads=1
    )
    dens = np.asarray(dens)
    if trim_quantile > 0:
        mesh.remove_vertices_by_mask(dens < np.quantile(dens, trim_quantile))
    return (
        np.asarray(mesh.vertices),
        np.asarray(mesh.triangles),
        (np.clip(np.asarray(mesh.vertex_colors), 0, 1) * 255).astype(np.uint8),
    )


# ------------------------------------------------------------------------------
# 预览渲染:不依赖显卡和图形界面,用 numpy 做一个简单的软件光栅化
# ------------------------------------------------------------------------------


def render_fast(verts, faces, vcol, yaw_deg, pitch_deg=15.0, res=320, shaded=True):
    """
    把网格绕竖直轴转 yaw_deg 度、绕水平轴转 pitch_deg 度,再做正交投影,生成一张预览图。
    三角形有几十万个,逐个光栅化太慢,所以只给每个三角形的重心所在的 2×2 像素上色,
    从远到近画,近的覆盖远的。
    三角形本身约一个像素大小,这个近似在预览时看不出差别。
    shaded=True 时按"光从相机方向照过来"做明暗着色,只看形状;False 时用顶点的 albedo 颜色。
    着色用的是面法线 z 分量的绝对值,一层表面从正面看和从反面看亮度相同。所以转到 180 度时
    看到的是同一层正面的反面,不能据此判断网格有没有背面。
    """
    ya, pa = np.radians(yaw_deg), np.radians(pitch_deg)
    Ry = np.array([[np.cos(ya), 0, np.sin(ya)], [0, 1, 0], [-np.sin(ya), 0, np.cos(ya)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(pa), -np.sin(pa)], [0, np.sin(pa), np.cos(pa)]])
    v = (verts - verts.mean(0)) @ (Rx @ Ry).T
    s = 0.9 * res / 2 / (np.abs(v[:, :2]).max() + 1e-8)
    cen = v[faces].mean(1)
    e1 = v[faces[:, 1]] - v[faces[:, 0]]
    e2 = v[faces[:, 2]] - v[faces[:, 0]]
    fn = np.cross(e1, e2)
    fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
    col = (
        (np.abs(fn[:, 2:3]) * 0.85 + 0.15) * 255 * np.ones((1, 3))
        if shaded
        else vcol[faces].mean(1)
    )

    order = np.argsort(cen[:, 2])  # 从远到近画,近的覆盖远的
    img = np.full((res, res, 3), 255, np.float32)
    for dy in (0, 1):
        for dx in (0, 1):
            x = np.clip((cen[order, 0] * s + res / 2).astype(int) + dx, 0, res - 1)
            y = np.clip((res / 2 - cen[order, 1] * s).astype(int) + dy, 0, res - 1)
            img[y, x] = col[order]
    return img.astype(np.uint8)


def make_preview(panels, labels, out_path):
    T = panels[0].shape[0]
    canvas = Image.new("RGB", (len(panels) * (T + 4), T + 22), (32, 32, 36))
    dr = ImageDraw.Draw(canvas)
    # 没有用 zip(panels, labels):ruff 会要求加 strict= 参数,而它需要 Python 3.10,
    # 服务器上 IDArb 的环境是 Python 3.8
    for i, p in enumerate(panels):
        t = labels[i]
        canvas.paste(Image.fromarray(p), (i * (T + 4), 22))
        dr.text((i * (T + 4) + 4, 5), t, fill=(255, 225, 130))
    canvas.save(out_path)


# ------------------------------------------------------------------------------
# 保存
# ------------------------------------------------------------------------------


def save_mesh(path, verts, faces, vcol):
    import trimesh

    m = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=vcol, process=False)
    m.export(path)


def process_one(
    result_path,
    input_path,
    out_dir,
    methods,
    poisson_depth=9,
    erode_px=2,
    min_nz=0.1,
    z_scale=1.0,
    depth_path=None,
    depth_weight=0.01,
    jump_px=3.0,
    min_part=300,
):
    os.makedirs(out_dir, exist_ok=True)
    albedo, normal_rgb, _, _ = load_result(result_path)
    mask = load_mask(input_path, normal_rgb.shape[:2])
    n = decode_normal(normal_rgb)
    m_int = erode(mask, erode_px)

    # 粗略检查坐标约定:对大致凸起的物体,越靠右的像素法线 x 分量应该越大(右边缘的表面朝右)。
    # 这只是参考指标,非凸物体(镂空、细腿)不满足这个规律。
    # 坐标约定本身已用 --selftest 和训练数据的真值法线核实过。
    rr, cc = np.nonzero(m_int)
    corr = np.corrcoef(cc - cc.mean(), n[rr, cc, 0])[0, 1]
    corr_y = np.corrcoef(-(rr - rr.mean()), n[rr, cc, 1])[0, 1]
    print(
        f"  坐标约定检查:列号与法线 x 分量的相关系数 {corr:.2f},"
        f"行号(朝上)与法线 y 分量的相关系数 {corr_y:.2f}"
        "(仅作参考:对熊猫这类大致凸起的物体两者应为正;椅子、动物腿这类非凸形状接近 0 或为负也正常)"
    )

    if depth_path is None:
        h = integrate_normals(n, m_int, min_nz=min_nz)
    else:
        d = load_depth(depth_path, normal_rgb.shape[:2])
        h, info = integrate_normals_with_depth(
            n, d, m_int, weight=depth_weight, jump_px=jump_px, min_part=min_part, min_nz=min_nz
        )
        print(
            f"  深度先验:深度图换算比例 {info['scale']:.0f} 像素,"
            f"判定为断崖的相邻像素对占 {info['jump_fraction'] * 100:.1f}%,"
            f"物体被断崖分成 {info['parts']} 块"
        )
    n_back = normals_from_depth(h)
    mean_err, med_err = angular_error(n, n_back, erode(m_int, 1))
    print(
        f"  积分一致性:由恢复出的深度重新计算法线,与输入法线的夹角"
        f"平均 {mean_err:.1f}°,中位数 {med_err:.1f}°"
    )

    # 深度图可视化
    d = h.copy()
    d = (d - np.nanmin(d)) / (np.nanmax(d) - np.nanmin(d) + 1e-8)
    d_img = np.where(np.isfinite(d), d, 0.0)
    Image.fromarray((d_img * 255).astype(np.uint8)).save(os.path.join(out_dir, "depth.png"))

    verts, faces, vcol = heightmap_to_mesh(h, m_int, albedo)
    if depth_path is not None:
        n_faces = len(faces)
        faces = drop_cut_faces(faces, info["cut_i"], info["cut_j"], len(verts))
        print(f"  去掉跨在断崖上的竖直面:{n_faces - len(faces)} 个三角形")
    # 中心化:把所有顶点坐标的平均值移到原点(0, 0, 0)。
    # heightmap_to_mesh 生成的坐标里,x、y 以图片中心为原点,z 以最远的点为 0,物体本身并不在原点上。
    # 不中心化的话,在 MeshLab、Blender 里旋转查看时,物体会绕着一个偏离自身的点转。
    verts = verts - verts.mean(axis=0)
    # 沿 Z 方向(深度方向)拉伸:中心化之后,z 坐标整体乘以 z_scale。z_scale=1 表示不拉伸。
    verts[:, 2] *= z_scale
    panels, labels = [], []
    if "relief" in methods:
        save_mesh(os.path.join(out_dir, "relief.ply"), verts, faces, vcol)
        save_mesh(os.path.join(out_dir, "relief.obj"), verts, faces, vcol)
        print(f"  方法 A(泊松法线积分):{len(verts)} 个顶点,{len(faces)} 个三角形")
        for yaw in (0, 45, 90):
            panels.append(render_fast(verts, faces, vcol, yaw))
            labels.append(f"A relief yaw {yaw}")
        panels.append(render_fast(verts, faces, vcol, 30, shaded=False))
        labels.append("A albedo yaw 30")

    if "poisson" in methods:
        pn = n[m_int].copy()  # 网络预测的法线,和顶点一一对应(同样的掩膜顺序)
        # 几何沿 z 放大 z_scale 倍后,法线必须同步变换,否则法线和拉伸后的表面对不上:
        # 法线的 z 分量除以 z_scale,再重新归一化(这是法线在非均匀缩放下的标准变换规则)
        pn[:, 2] /= z_scale
        pn /= np.linalg.norm(pn, axis=1, keepdims=True)
        pv, pf, pc = screened_poisson(verts, pn, vcol, depth=poisson_depth)
        # 方法 B 重新划分了三角形,顶点和输入点不同,裁剪后剩下的范围也略有差别,
        # 所以按它自己的顶点均值再中心化一次
        pv = pv - pv.mean(axis=0)
        save_mesh(os.path.join(out_dir, "poisson.ply"), pv, pf, pc)
        print(f"  方法 B(筛选泊松表面重建):{len(pv)} 个顶点,{len(pf)} 个三角形")
        for yaw in (0, 90, 180):
            panels.append(render_fast(pv, pf, pc, yaw))
            labels.append(f"B poisson yaw {yaw}")

    make_preview(panels, labels, os.path.join(out_dir, "preview.png"))
    return dict(mean_err=mean_err, med_err=med_err, corr_x=corr, corr_y=corr_y)


# ------------------------------------------------------------------------------
# 自检:理想半球
# ------------------------------------------------------------------------------


def selftest(out_dir):
    """
    造一个半径 R 的理想半球,按 IDArb 的编码方式(lub 坐标、颜色 = (n+1)/2)生成法线图,
    再用 integrate_normals 积分,和真实的半球高度比较。这样可以验证符号和坐标约定都是对的。
    """
    os.makedirs(out_dir, exist_ok=True)
    H = W = 256
    R = 100.0
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    x = xx - W / 2 + 0.5
    y_up = -(yy - H / 2 + 0.5)
    rho2 = x**2 + y_up**2
    mask = rho2 < (R * 0.97) ** 2
    z = np.sqrt(np.clip(R**2 - rho2, 0, None))
    n = np.stack([x / R, y_up / R, z / R], -1)  # x 朝右的法线
    n_lub = n.copy()
    n_lub[..., 0] = -n_lub[..., 0]  # 换成 IDArb 的 lub 编码
    rgb = (n_lub + 1) / 2

    n_dec = decode_normal(rgb.astype(np.float32))
    h = integrate_normals(n_dec, mask, min_nz=0.05, max_grad=50)
    # 深度只能恢复到相差一个常数,所以比较前把两者的平均值对齐
    gt = z - z[mask].mean()
    est = h - np.nanmean(h[mask])
    err = np.abs(est[mask] - gt[mask])
    print(
        f"理想半球自检:半径 {R:.0f} 像素,"
        f"真实最高点比边缘高 {z[mask].max() - z[mask].min():.1f} 像素"
    )
    print(
        f"  恢复出的高度差 {np.nanmax(h[mask]) - np.nanmin(h[mask]):.1f} 像素;"
        f"逐像素高度误差 平均 {err.mean():.2f} 像素,最大 {err.max():.2f} 像素"
    )
    center_higher = h[H // 2, W // 2] > h[H // 2, int(W / 2 + R * 0.8)]
    print(f"  中心比边缘更靠近相机:{center_higher}(应为 True,否则说明符号反了)")
    verts, faces, vcol = heightmap_to_mesh(h, mask, np.ones((H, W, 3)) * 0.8)
    make_preview(
        [render_fast(verts, faces, vcol, yaw, res=240) for yaw in (0, 45, 90)],
        [f"hemisphere yaw {a}" for a in (0, 45, 90)],
        os.path.join(out_dir, "selftest_hemisphere.png"),
    )
    print("  预览图已保存到", os.path.join(out_dir, "selftest_hemisphere.png"))
    selftest_occlusion(out_dir)


def selftest_occlusion(out_dir):
    """
    有遮挡的场景:大球(半径 90)前方悬着一个小球(半径 35),两者之间隔着一段空隙。
    只用法线积分时,这段空隙(深度断崖)会丢失;给一张理想的深度先验(真实深度经过任意线性变换),
    integrate_normals_with_depth 应该能把断崖恢复出来。
    """
    H = W = 256
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)

    def sphere(cx, cy, R, cz, shrink=0.97):
        x = xx - cx
        y = -(yy - cy)
        r2 = x**2 + y**2
        zz = np.sqrt(np.clip(R**2 - r2, 0, None))
        m = r2 < (shrink * R) ** 2
        return m, np.where(m, cz + zz, -np.inf), np.stack([x / R, y / R, zz / R], -1)

    mb, zb, nb = sphere(128, 128, 90, 0)
    mf, zf, nf = sphere(165, 160, 35, 150)
    mask = mb | mf
    front = zf > zb
    z = np.where(front, zf, zb)
    n = np.where(front[..., None], nf, nb)
    d = 0.3 + 0.5 * (z - z[mask].min()) / (z[mask].max() - z[mask].min())  # 真实深度的线性变换
    d = np.where(mask, d, 0.0)

    ring_in = mf & ~sphere(165, 160, 35, 150, 0.90)[0]  # 小球轮廓内侧一圈
    ring_out = mb & ~mf & sphere(165, 160, 35, 150, 1.10)[0]  # 紧挨着的大球表面

    def jump(h):
        return np.nanmean(h[ring_in]) - np.nanmean(h[ring_out])

    h_n = integrate_normals(n, mask, min_nz=0.02, max_grad=50)
    h_d, info = integrate_normals_with_depth(n, d, mask, min_nz=0.02, max_grad=50)
    print("遮挡场景自检:小球悬在大球前方")
    print(
        f"  断崖高度:真实 {jump(np.where(mask, z, np.nan)):.1f} 像素,"
        f"只用法线 {jump(h_n):.1f} 像素,法线 + 深度先验 {jump(h_d):.1f} 像素"
        "(后者应接近真实值)"
    )
    # 真实深度的网格按真实断崖(相邻像素深度差超过 3 像素)去掉竖直面;只用法线的结果没有断崖
    idx = -np.ones((H, W), dtype=np.int64)
    idx[mask] = np.arange(mask.sum())
    ri, ci, rj, cj, _ = neighbor_pairs(mask)
    true_jump = np.abs(z[rj, cj] - z[ri, ci]) > 3
    cuts = [
        (idx[ri, ci][true_jump], idx[rj, cj][true_jump]),
        (np.zeros(0, np.int64), np.zeros(0, np.int64)),
        (info["cut_i"], info["cut_j"]),
    ]
    heights = (np.where(mask, z - z[mask].min(), np.nan), h_n, h_d)
    panels = []
    for k, h in enumerate(heights):
        cut_i, cut_j = cuts[k]
        verts, faces, vcol = heightmap_to_mesh(h, mask, np.ones((H, W, 3)) * 0.8)
        faces = drop_cut_faces(faces, cut_i, cut_j, len(verts))
        panels.append(render_fast(verts, faces, vcol, 75, res=240))
    make_preview(
        panels,
        ["ground truth", "normals only", "normals + depth prior"],
        os.path.join(out_dir, "selftest_occlusion.png"),
    )
    print("  预览图已保存到", os.path.join(out_dir, "selftest_occlusion.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--result", type=str, help="main.py 输出的四联图,例如 output/my_single/panda_00.png"
    )
    ap.add_argument("--input", type=str, help="对应的 RGBA 输入图,用来取前景掩膜")
    ap.add_argument("--result_dir", type=str, help="批量模式:结果文件夹")
    ap.add_argument("--input_dir", type=str, help="批量模式:输入 RGBA 图所在文件夹")
    ap.add_argument("--out_dir", type=str, default="outputs/mesh")
    ap.add_argument(
        "--methods", type=str, default="relief,poisson", help="relief、poisson 或两者用逗号连接"
    )
    ap.add_argument("--poisson_depth", type=int, default=9)
    ap.add_argument("--erode", type=int, default=2, help="掩膜向内收缩的像素数")
    ap.add_argument(
        "--min_nz", type=float, default=0.1, help="法线 z 分量的下限,越小边缘越陡但越容易出现尖刺"
    )
    ap.add_argument(
        "--z_scale", type=float, default=1.0, help="沿深度方向拉伸的倍数,例如 2 表示深度放大 2 倍"
    )
    ap.add_argument("--depth", type=str, help="单个物体的深度图(estimate_depth.py 生成的 PNG)")
    ap.add_argument("--depth_dir", type=str, help="批量模式:深度图所在文件夹,按物体名找 <名字>.png")
    ap.add_argument(
        "--depth_weight", type=float, default=0.01, help="深度先验的权重,越大越贴近深度图"
    )
    ap.add_argument(
        "--jump_px", type=float, default=3.0, help="深度差超过多少像素算作断崖(断开法线约束)"
    )
    ap.add_argument(
        "--min_part",
        type=int,
        default=300,
        help="断崖切出的块小于多少像素时接回去(深度模型在细小结构上容易误判)",
    )
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    methods = args.methods.split(",")

    if args.selftest:
        selftest(args.out_dir)
        return

    jobs = []
    if args.result:
        name = os.path.basename(args.result).rsplit("_", 1)[0]
        jobs.append((name, args.result, args.input, args.depth))
    elif args.result_dir:
        for f in sorted(os.listdir(args.result_dir)):
            if not f.endswith(".png"):
                continue
            name = f.rsplit("_", 1)[0]
            inp = os.path.join(args.input_dir, name + ".png")
            dep = os.path.join(args.depth_dir, name + ".png") if args.depth_dir else None
            if not os.path.exists(inp):
                print(f"跳过 {f}:找不到输入图 {inp}")
            elif dep is not None and not os.path.exists(dep):
                print(f"跳过 {f}:找不到深度图 {dep}")
            else:
                jobs.append((name, os.path.join(args.result_dir, f), inp, dep))
    else:
        ap.error("需要 --result 和 --input,或者 --result_dir 和 --input_dir,或者 --selftest")

    for name, res, inp, dep in jobs:
        print(f"[{name}]")
        process_one(
            res,
            inp,
            os.path.join(args.out_dir, name),
            methods,
            poisson_depth=args.poisson_depth,
            erode_px=args.erode,
            min_nz=args.min_nz,
            z_scale=args.z_scale,
            depth_path=dep,
            depth_weight=args.depth_weight,
            jump_px=args.jump_px,
            min_part=args.min_part,
        )


if __name__ == "__main__":
    main()
