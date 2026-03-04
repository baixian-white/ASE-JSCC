# ASE-JSCC

Attention-enhanced Semantic Extraction and Joint Source Channel Coding for remote sensing image classification.

## 重构后的项目结构

```text
ASE-JSCC/
├── scripts/
│   ├── train/
│   │   └── ASE-JSCCtrain.py
│   ├── eval/
│   │   ├── confusion_tsne_UM.py
│   │   ├── confusion_tsne_AID.py
│   │   └── confusion_tsne_Soya.py
│   ├── visualization/
│   │   └── grad_cam_visualize.py
│   ├── infer/
│   │   └── predict_demo.py
│   └── data_prep/
│       ├── split_ucmerced_landuse.py
│       ├── split_AID.py
│       └── split_Soya.py
├── configs/
│   ├── ase-jscc-gpu.yml
│   └── environment.yml
├── docs/
│   ├── dataset.txt
│   ├── 命令行提示.md
│   ├── 项目说明书.md
│   └── 项目结构说明书.md
├── notebooks/
│   ├── Untitled.ipynb
│   └── Untitled1.ipynb
├── data/
├── checkpoint/
├── logs/
└── runs/
```

> 所有脚本已改为“自动定位项目根目录（.git）”，可在任意工作目录执行。

## 环境准备

```bash
conda env create -f configs/ase-jscc-gpu.yml
conda activate ASE-JSCC
```

## 常用命令

### 1) 数据划分

```bash
python scripts/data_prep/split_ucmerced_landuse.py
python scripts/data_prep/split_AID.py
python scripts/data_prep/split_Soya.py data/SoyaHealthVision data/SoyaHealthVision
```

### 2) 训练

```bash
python scripts/train/ASE-JSCCtrain.py --task train --cr 0.8 --num_epochs 150 --channel_type Combined_channel
```

断点续训：

```bash
python scripts/train/ASE-JSCCtrain.py --task continue --cr 0.8 --num_epochs 30 --pre_checkpoint checkpoint/xxx.pth --channel_type Combined_channel
```

### 3) 评估可视化（混淆矩阵 + t-SNE）

```bash
python scripts/eval/confusion_tsne_UM.py
python scripts/eval/confusion_tsne_AID.py
python scripts/eval/confusion_tsne_Soya.py
```

### 4) Grad-CAM

```bash
python scripts/visualization/grad_cam_visualize.py -c overpass -n 6
```

### 5) 推理

单图：

```bash
python scripts/infer/predict_demo.py --checkpoint checkpoint/xxx.pth --image data/xxx.jpg --channel_type Combined_channel --cr 0.8
```

批量：

```bash
python scripts/infer/predict_demo.py --checkpoint checkpoint/xxx.pth --data_dir data/UCMerced_LandUse-test --csv_out logs/predictions.csv
```

## 说明

- 数据下载链接见 [docs/dataset.txt](docs/dataset.txt)
- 详细项目说明见 [docs/项目说明书.md](docs/项目说明书.md)
- 本次重构说明见 [docs/项目结构说明书.md](docs/项目结构说明书.md)
