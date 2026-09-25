"""从 Arb-Objaverse 数据集中只下载指定的几个物体。

Arb-Objaverse(https://huggingface.co/datasets/lizb6626/Arb-Objaverse)的每个压缩包约 10 GB,
里面有几百个物体。做训练冒烟测试只需要其中两三个,整包下载没有必要。

这个脚本利用 HTTP 分段读取:zip 文件的目录表放在文件末尾,先只读目录表,找到目标物体对应的文件,
再只下载这些文件的数据。每个物体约 20 到 30 MB。

数据列表的格式和 IDArb 的 datalist 一致,是一个 JSON 列表,
每一项是 ["arbobjaverse", "<包名>/<物体编号>"],
例如 ["arbobjaverse", "000-000/0000ecca9a234cae994be239f6fec552"]。
包名 000-000 对应压缩包 data/000-000.zip。

用法(在 third_party/IDArb 目录下运行,这样文件会放到训练配置期望的位置 data/Arbobjaverse/):
    python ../../scripts/idarb/fetch_arbobjaverse_subset.py --datalist datalist/my_train.json
"""

import argparse
import json
import os
import zipfile
from collections import defaultdict

from huggingface_hub import HfFileSystem

REPO = "datasets/lizb6626/Arb-Objaverse/data"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datalist", required=True, help="JSON 数据列表")
    parser.add_argument("--root", default="data/Arbobjaverse", help="解压到哪个目录")
    args = parser.parse_args()

    with open(args.datalist) as f:
        entries = json.load(f)

    # 按压缩包分组:同一个压缩包里的物体只需要读一次目录表
    by_pack = defaultdict(list)
    for data_type, obj in entries:
        if data_type != "arbobjaverse":
            print(f"跳过不是 Arb-Objaverse 的条目: {data_type} {obj}")
            continue
        by_pack[obj.split("/")[0]].append(obj)

    fs = HfFileSystem()
    for pack, objs in by_pack.items():
        todo = [o for o in objs if not os.path.isdir(os.path.join(args.root, o))]
        if not todo:
            print(f"{pack}.zip: 需要的 {len(objs)} 个物体都已存在,跳过")
            continue
        print(f"{pack}.zip: 读取目录表,准备提取 {len(todo)} 个物体")
        with fs.open(f"{REPO}/{pack}.zip", "rb", block_size=8 * 1024 * 1024) as remote:
            archive = zipfile.ZipFile(remote)
            names = archive.namelist()
            for obj in todo:
                members = [n for n in names if n.startswith(obj + "/") and not n.endswith("/")]
                if not members:
                    print(f"  找不到物体 {obj}")
                    continue
                for member in members:
                    archive.extract(member, args.root)
                print(f"  {obj}: {len(members)} 个文件")


if __name__ == "__main__":
    main()
