# -*- coding: utf-8 -*-
"""
将已是“官网结构”的 UCMerced_LandUse 数据集按 5:1:4 分层随机划分到
- ASE-JSCC/data/UCMerced_LandUse-train
- ASE-JSCC/data/UCMerced_LandUse-valid
- ASE-JSCC/data/UCMerced_LandUse-test

用法：
  在 ASE-JSCC 根目录运行：
    python split_ucmerced_official.py

说明：
  - 按“每个类别”分别随机划分，保证类别分布一致（分层采样）。
  - 复制文件（不移动），安全；如需移动可把 shutil.copy2 改为 shutil.move。
  - 固定随机种子以保证可复现。
"""

import os
import shutil
import random
from pathlib import Path

# ---- 可根据需要调整的参数（也可改为 argparse） ----
REPO_ROOT = Path("data")  # 工程根目录
SRC_DIR   =  REPO_ROOT / "UCMerced_LandUse" / "images"          # 官方结构的源目录
DST_TRAIN =   REPO_ROOT / "UCMerced_LandUse-train"    # 训练集输出
DST_VALID =   REPO_ROOT / "UCMerced_LandUse-valid"    # 验证集输出
DST_TEST  =   REPO_ROOT / "UCMerced_LandUse-test"     # 测试集输出

RATIOS    = (0.5, 0.1, 0.4)  # (train, valid, test) = 5:1:4
SEED      = 42               # 随机种子，保证可复现

# 允许的图片扩展名（大小写都接受）
IMG_EXTS  = {".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp"}


def list_images_in_dir(dir_path: Path):
    """列出某个类别目录下的所有图片文件（仅顶层，不递归子目录）"""
    files = []
    for p in dir_path.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)
    return files


def stratified_split(file_list, ratios):
    """对一个类别的文件列表按 ratios 分割并返回 (train, valid, test)"""
    n = len(file_list)
    # 先固定顺序，再随机打乱（种子在外部设定）
    file_list = sorted(file_list)
    random.shuffle(file_list)

    n_train = int(round(n * ratios[0]))
    n_valid = int(round(n * ratios[1]))
    n_test  = n - n_train - n_valid  # 保证总数不丢失

    train = file_list[:n_train]
    valid = file_list[n_train:n_train + n_valid]
    test  = file_list[n_train + n_valid:]
    return train, valid, test


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def main():
    # 基本检查
    if not SRC_DIR.is_dir():
        raise FileNotFoundError(f"源目录不存在：{SRC_DIR}")

    # 固定随机种子
    random.seed(SEED)

    # 创建目标根目录
    for root in (DST_TRAIN, DST_VALID, DST_TEST):
        ensure_dir(root)

    # 逐类别处理
    total_train = total_valid = total_test = 0
    class_dirs = [d for d in SRC_DIR.iterdir() if d.is_dir()]

    if not class_dirs:
        raise RuntimeError(f"未在 {SRC_DIR} 发现任何类别子文件夹。请确认目录结构。")

    print(f"开始分层划分（5:1:4），源目录：{SRC_DIR}")
    for cls_dir in sorted(class_dirs):
        cls_name = cls_dir.name
        images = list_images_in_dir(cls_dir)

        if not images:
            print(f"[警告] 类别 {cls_name} 下没有找到图片，跳过。")
            continue

        train_files, valid_files, test_files = stratified_split(images, RATIOS)

        # 为该类别创建目标子目录
        for root in (DST_TRAIN, DST_VALID, DST_TEST):
            ensure_dir(root / cls_name)

        # 复制文件
        for src in train_files:
            dst = DST_TRAIN / cls_name / src.name
            shutil.copy2(src, dst)
        for src in valid_files:
            dst = DST_VALID / cls_name / src.name
            shutil.copy2(src, dst)
        for src in test_files:
            dst = DST_TEST / cls_name / src.name
            shutil.copy2(src, dst)

        print(f"[{cls_name:<20}] total={len(images):4d} -> "
              f"train={len(train_files):3d}, valid={len(valid_files):3d}, test={len(test_files):3d}")

        total_train += len(train_files)
        total_valid += len(valid_files)
        total_test  += len(test_files)

    total = total_train + total_valid + total_test
    print("\n✅ 划分完成！")
    print(f"总计：{total} 张（train={total_train}, valid={total_valid}, test={total_test}）")
    print(f"训练集目录：{DST_TRAIN}")
    print(f"验证集目录：{DST_VALID}")
    print(f"测试集目录：{DST_TEST}")


if __name__ == "__main__":
    main()
