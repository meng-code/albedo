# scripts/idarb

围绕 IDArb(代码在 `third_party/IDArb/`)新增的脚本。

## 分工

- **服务器(有 NVIDIA 显卡)**:运行 IDArb,把图片分解成 albedo、normal、metallic、roughness;
  用深度模型生成深度图;训练冒烟测试。
- **自己的电脑(不需要显卡,也不需要 PyTorch)**:用分解出来的法线图重建三维网格。

| 脚本 | 在哪里运行 | 作用 |
|---|---|---|
| `prepare_images.py` | 服务器 | 把普通照片处理成 IDArb 要求的 RGBA 四通道 PNG |
| `fetch_arbobjaverse_subset.py` | 服务器 | 从 Arb-Objaverse 数据集里只下载指定的几个物体,用于训练冒烟测试 |
| `estimate_depth.py` | 服务器 | 用 Depth Anything V2 生成相对深度图,给网格重建做深度先验 |
| `normal_to_mesh.py` | 自己的电脑或服务器 | 把 IDArb 的法线图重建成三维网格,可选加入深度先验 |
| `plot_failure_cases.py` | laptop | Draws failure-case figures (input, IDArb output, side views of both meshes), e.g. `results/analysis/failure_cats.png` |

## 1. 服务器:图片分解

以下命令在 `third_party/IDArb/` 目录下、IDArb 的 conda 环境(Python 3.8、torch 2.2.1)里运行。
这个环境的安装方法和已验证可用的版本见 `requirements-server.txt`。官方的
`third_party/IDArb/requirements.txt` 里有几个用不到、也不好装的包,不建议直接用。

`prepare_images.py` 用 rembg 去掉背景,把前景信息写进 alpha 通道。如果图片本身已经带有可用的透明通道,
就直接沿用,不再重新抠图。

```bash
python ../../scripts/idarb/prepare_images.py --input mydata/raw --output mydata/my_single
python main.py --data_dir mydata/my_single --output_dir output/my_single --input_type single
```

分解结果保存在 `output/my_single/<名字>_00.png`,每张是四张图横向拼接:albedo、normal、metallic、roughness。

`main.py` 默认使用官方训练好的权重。加上 `--pretrained output/train_smoke/pipeckpts` 可以换成自己训练的权重。

需要深度先验时,再为同一批输入图生成深度图(在仓库根目录运行,默认用 Depth-Anything-V2-Small):

```bash
python scripts/idarb/estimate_depth.py --input_dir results/idarb_single/inputs --out_dir results/idarb_single/depth
```

深度图是 16 位灰度 PNG,和 IDArb 的输出逐像素对齐,越亮越靠近相机。把它和分解结果一起推到 GitHub,
在自己电脑上做网格时就不需要 PyTorch。

## 2. 服务器:训练冒烟测试

目的是验证从 Stable Diffusion 2.1 的原始权重出发、用公开数据集训练 IDArb 的流程能跑通,不追求效果。

```bash
cd third_party/IDArb
python ../../scripts/idarb/fetch_arbobjaverse_subset.py --datalist datalist/my_train.json
accelerate launch --config_file configs/acc/1gpu.yaml train.py --config configs/my_train_smoke.yaml
```

第一条命令从约 10 GB 的压缩包里只取出 3 个物体(共约 77 MB),放到 `data/Arbobjaverse/`。
配置 `configs/my_train_smoke.yaml` 是官方 `train.yaml` 的单卡缩小版:batch 为 1、关闭 EMA、训练 300 步、
每 100 步验证一次。在一张 A40(24 GB)上用时约 7 分半,显存占用约 19 GB。

## 3. 自己的电脑:法线图转网格

### 安装(只需要一次)

```bash
conda create -n mesh python=3.10 -y
conda activate mesh
pip install -r scripts/idarb/requirements-local.txt
```

### 运行(在仓库根目录)

仓库里的 `results/idarb_single/` 带有一组示例数据:`decomposition/` 是 IDArb 的分解结果,
`inputs/` 是对应的抠图输入(用来算出物体所在的像素范围)。

```bash
# 处理一个物体
python scripts/idarb/normal_to_mesh.py \
    --result results/idarb_single/decomposition/panda_00.png \
    --input results/idarb_single/inputs/panda.png

# 批量处理
python scripts/idarb/normal_to_mesh.py \
    --result_dir results/idarb_single/decomposition --input_dir results/idarb_single/inputs

# 沿深度方向拉伸 2 倍
python scripts/idarb/normal_to_mesh.py \
    --result_dir results/idarb_single/decomposition --input_dir results/idarb_single/inputs \
    --out_dir outputs/mesh_z2 --z_scale 2

# 加入深度先验(推荐):补回前后部件之间的距离,纠正整体倾斜
python scripts/idarb/normal_to_mesh.py \
    --result_dir results/idarb_single/decomposition --input_dir results/idarb_single/inputs \
    --depth_dir results/idarb_single/depth --out_dir outputs/mesh_depth

# 检查算法是否正确(不需要任何输入文件):理想半球,以及有遮挡断崖的两个球
python scripts/idarb/normal_to_mesh.py --selftest
```

### 输出

默认保存在 `outputs/mesh/<名字>/`(这个文件夹不进 git):

| 文件 | 内容 |
|---|---|
| `relief.ply` / `relief.obj` | 方法 A(泊松法线积分)得到的网格,只有朝向相机的一面,带 albedo 颜色 |
| `poisson.ply` | 方法 B(筛选泊松表面重建)得到的网格。单张图片上它只有正面,和 `relief.ply` 几乎相同,不是封闭网格 |
| `depth.png` | 恢复出的深度图,越亮越靠近相机 |
| `preview.png` | 从几个角度渲染的预览图 |

所有网格的顶点均值都已经移到坐标原点。建议用 `.ply` 文件查看,颜色存在顶点上,`.ply` 对顶点颜色的支持是标准的。

方法 B 调用 Open3D 泊松重建时固定只用一个线程。默认的多线程在 Mac(M 系列芯片)上经常报错
"Failed to close loop" 并退出,而且同样的输入每次得到的三角形数量会差几个。单线程每次都能成功,结果也完全一样。

### 方法说明

- **方法 A:泊松法线积分。** 法线给出的是表面每一点的倾斜方向,也就是深度的梯度。求一张深度图,
  让它的梯度尽量接近法线给出的梯度,这个最小二乘问题化简后是一个泊松方程。解出深度图后,把像素网格连成三角形。
- **方法 B:筛选泊松表面重建**(Kazhdan & Hoppe 2013)。把方法 A 得到的三维点和网络预测的法线交给 Open3D,
  重建曲面。点覆盖物体各个方向时,它输出封闭曲面;但单张图片只有正面的点,所以它不会补出背面。
  实测得到的是物体正面,外加一大片从物体四周延伸到包围盒边缘的平面;按点的密度裁掉这片平面之后,
  剩下的正面和方法 A 的表面相差不到半个像素(距离中位数约 0.45 像素)。预览图里转到 180 度的画面
  是同一层正面的反面,不是背面。
- **方法 A 加深度先验(`--depth_dir`)。** 只用法线时,前后遮挡处的深度断崖会丢失(见下面的已知局限)。
  加入 `estimate_depth.py` 生成的相对深度图 d 之后,求解
  `min Σ(相邻深度差 − 法线给出的深度差)² + w·Σ(h − s·d)²`:法线负责表面细节,深度图负责部件之间的前后关系。
  深度图认为有断崖的相邻像素之间,不再要求平滑相连;生成网格时再去掉正好跨在断崖上的三角形("墙")。
  深度模型在椅子腿这类细小结构上估得不准,会切出很多漂浮的小碎块,所以小于 `--min_part` 像素的碎块
  会被接回到与它交界最多的那一个相邻块上(只接一个,避免把披风和身体这类本该分开的部件粘在一起)。
  相关参数:`--depth_weight`(默认 0.01)、`--jump_px`(默认 3,深度差超过多少像素算断崖)、
  `--min_part`(默认 300,小于多少像素的碎块要接回去)。

### 已知局限和实验结果

- 单张图片只能恢复朝向相机的一面。
- **只用法线时**,物体部件之间前后遮挡的地方(比如手臂挡在身体前面),深度上的断崖会被抹平,
  前面的部件会被"贴"到后面的部件上,导致整体偏扁。用深度已知的合成场景测试过:
  真实断崖 49、99、149 像素,恢复出来都只有约 5 像素;没有遮挡时(单个球)能完全准确地恢复。
- **加入深度先验后**,同一个合成场景的断崖(真实 99.4 像素)恢复为 99.4 像素。在示例数据上:
  - chair1 的深度与宽度之比从 0.28 变为 0.82,座面和靠背的前后关系变对了;
  - labubu 原本整体倾斜,现在是直立的;
  - bird、panda、superman 这类没有明显遮挡的物体变化不大;
  - 细小结构不再碎成漂浮的小块:chair1 原本被切成 67 块(其中 64 块小于 300 像素),
    现在只剩主体和两条后腿 3 块。后腿仍和椅子分开,是因为照片里后腿的上半截被座面挡住,
    看不到它在哪里和椅子相连,只能按深度模型估计的位置放在后方;
  - 代价是重建表面和 IDArb 法线的偏差变大(例如 chair1 从 4.7° 变为约 11°)。
- **When the normals fail completely, the depth prior fails too.** The depth-prior scale s is fitted to the
  normal slopes. On cat2, cat3 and cat4, IDArb's normals almost all face the camera, so s is only 3 to 16 px
  (about 100 to 250 px on other objects), and the meshes are still a line when seen from the side.
  See `results/analysis/failure_cats.md`.
    权重从 0.01 调到 0.2,深度比例几乎不变,偏差却继续变大,所以默认取 0.01。
  - 对比图:`results/idarb_single/depth_prior_compare.png`。
- 网络预测的法线有随机误差。在数据集的 16 个样本上与真值平均相差 17.2°,积分出的深度是真值的 0.43 到 1.55 倍,
  中位数 1.00,没有系统性地偏平。
