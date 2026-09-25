# scripts/idarb

围绕 IDArb(代码在 `third_party/IDArb/`)新增的脚本。

## 分工

- **服务器(有 NVIDIA 显卡)**:运行 IDArb,把图片分解成 albedo、normal、metallic、roughness;训练冒烟测试。
- **自己的电脑(不需要显卡,也不需要 PyTorch)**:用分解出来的法线图重建三维网格。

| 脚本 | 在哪里运行 | 作用 |
|---|---|---|
| `prepare_images.py` | 服务器 | 把普通照片处理成 IDArb 要求的 RGBA 四通道 PNG |
| `fetch_arbobjaverse_subset.py` | 服务器 | 从 Arb-Objaverse 数据集里只下载指定的几个物体,用于训练冒烟测试 |
| `normal_to_mesh.py` | 自己的电脑或服务器 | 把 IDArb 的法线图重建成三维网格 |

## 1. 服务器:图片分解

以下命令在 `third_party/IDArb/` 目录下、IDArb 的 conda 环境(Python 3.8、torch 2.2.1)里运行。

`prepare_images.py` 用 rembg 去掉背景,把前景信息写进 alpha 通道。如果图片本身已经带有可用的透明通道,
就直接沿用,不再重新抠图。

```bash
python ../../scripts/idarb/prepare_images.py --input mydata/raw --output mydata/my_single
python main.py --data_dir mydata/my_single --output_dir output/my_single --input_type single
```

分解结果保存在 `output/my_single/<名字>_00.png`,每张是四张图横向拼接:albedo、normal、metallic、roughness。

`main.py` 默认使用官方训练好的权重。加上 `--pretrained output/train_smoke/pipeckpts` 可以换成自己训练的权重。

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

# 用理想半球检查算法是否正确(不需要任何输入文件)
python scripts/idarb/normal_to_mesh.py --selftest
```

### 输出

默认保存在 `outputs/mesh/<名字>/`(这个文件夹不进 git):

| 文件 | 内容 |
|---|---|
| `relief.ply` / `relief.obj` | 方法 A(泊松法线积分)得到的网格,只有朝向相机的一面,带 albedo 颜色 |
| `poisson.ply` | 方法 B(筛选泊松表面重建)得到的封闭网格,背面是算法补出来的 |
| `depth.png` | 恢复出的深度图,越亮越靠近相机 |
| `preview.png` | 从几个角度渲染的预览图 |

所有网格的顶点均值都已经移到坐标原点。建议用 `.ply` 文件查看,颜色存在顶点上,`.ply` 对顶点颜色的支持是标准的。

方法 B 用的 Open3D 泊松重建内部是多线程计算,同样的输入跑两次,三角形数量会有几个的差别,这是正常的。

### 两种方法

- **方法 A:泊松法线积分。** 法线给出的是表面每一点的倾斜方向,也就是深度的梯度。求一张深度图,
  让它的梯度尽量接近法线给出的梯度,这个最小二乘问题化简后是一个泊松方程。解出深度图后,把像素网格连成三角形。
- **方法 B:筛选泊松表面重建**(Kazhdan & Hoppe 2013)。把方法 A 得到的三维点和网络预测的法线交给 Open3D,
  输出一个封闭曲面。单张图片看不到背面,所以背面是算法补出来的,不可信。

### 已知局限

- 单张图片只能恢复朝向相机的一面。
- 物体部件之间前后遮挡的地方(比如手臂挡在身体前面),深度上的断崖会被抹平,前面的部件会被"贴"到后面的部件上,
  导致整体偏扁。用深度已知的合成场景测试过:真实断崖 49、99、149 像素,恢复出来都只有约 5 像素;
  没有遮挡时(单个球)能完全准确地恢复。
- 网络预测的法线有随机误差。在数据集的 16 个样本上与真值平均相差 17.2°,积分出的深度是真值的 0.43 到 1.55 倍,
  中位数 1.00,没有系统性地偏平。
