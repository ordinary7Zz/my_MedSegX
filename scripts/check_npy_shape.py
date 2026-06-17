import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np


def check_npy_shapes(directory: Path) -> None:
    shapes = defaultdict(list)

    npy_files = sorted(directory.glob("*.npy"))
    if not npy_files:
        print("目录下没有找到 .npy 文件")
        return

    for path in npy_files:
        try:
            arr = np.load(path)
            shapes[arr.shape].append(path.name)
        except Exception as e:
            print(f"读取失败: {path.name} -> {e}")

    if not shapes:
        print("没有成功读取任何 .npy 文件")
        return

    if len(shapes) == 1:
        shape = next(iter(shapes))
        print(f"所有 npy 文件尺寸一致，shape = {shape}")
        return

    print("发现不同尺寸的 npy 文件：")
    for shape, files in shapes.items():
        print(f"\nshape = {shape}，数量 = {len(files)}")
        for name in files:
            print(f"  - {name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="检查目录下所有 npy 文件的尺寸是否一致")
    parser.add_argument("directory", help="npy 文件所在目录")
    args = parser.parse_args()

    check_npy_shapes(Path(args.directory))
