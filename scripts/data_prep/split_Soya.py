#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
功能：
1. 对三个类别文件夹的数据划分为 train / valid / test
2. 比例 = 6 : 3 : 1
3. 自动创建输出文件夹结构并复制文件

使用方式：
    python scripts/data_prep/split_Soya.py  <数据根目录>  <输出目录>

示例：
    python scripts/data_prep/split_Soya.py  data/SoyaHealthVision  data/SoyaHealthVision


"""

import shutil
import random
from pathlib import Path
import sys

def get_project_root() -> Path:
    current = Path(__file__).resolve().parent
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return Path.cwd()


PROJECT_ROOT = get_project_root()


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


# 固定随机种子保证可复现
random.seed(42)


def make_dir(path: Path):
    """如果目录不存在，则创建"""
    path.mkdir(parents=True, exist_ok=True)


def split_list(items, train_ratio=0.6, valid_ratio=0.3):
    """按比例拆分列表：train / valid / test"""
    n = len(items)
    train_end = int(n * train_ratio)
    valid_end = int(n * (train_ratio + valid_ratio))
    return items[:train_end], items[train_end:valid_end], items[valid_end:]


def copy_files(files, dst_dir: Path):
    """将文件复制到指定目录"""
    for f in files:
        shutil.copy(f, dst_dir / f.name)
        

def main(data_root: Path, output_root: Path):

    # 三个类别名称（可按需扩展）
    class_names = [
        "Soyabean Semilooper_Pest_Attack",
        "Soyabean_Mosaic",
        "Soyabean_Rust"
    ]

    # 创建 train/valid/test 基础目录
    for split in ["train", "valid", "test"]:
        make_dir(output_root / split)

    for cls in class_names:
        cls_path = data_root / cls
        if not cls_path.exists():
            print(f"❌ 类别目录不存在: {cls_path}")
            continue

        print(f"📌 处理类别：{cls}")

        # 获取所有图片
        files = [f for f in cls_path.iterdir() if f.is_file()]
        random.shuffle(files)

        train_files, valid_files, test_files = split_list(files)

        # 输出目录
        train_dir = output_root / "train" / cls
        valid_dir = output_root / "valid" / cls
        test_dir = output_root / "test" / cls

        make_dir(train_dir)
        make_dir(valid_dir)
        make_dir(test_dir)

        # 复制文件
        copy_files(train_files, train_dir)
        copy_files(valid_files, valid_dir)
        copy_files(test_files, test_dir)

        print(f"   ✔ Train: {len(train_files)}, Valid: {len(valid_files)}, Test: {len(test_files)}")

    print("\n🎉 数据集划分完成！输出位于：", output_root)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法：python scripts/data_prep/split_Soya.py <数据根目录> <输出目录>")
        sys.exit(1)

    data_root = resolve_path(sys.argv[1])
    output_root = resolve_path(sys.argv[2])

    main(data_root, output_root)
