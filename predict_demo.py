# predict_demo.py
import argparse
import csv
import os
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from PIL import Image

# ==== 引入你训练脚本里的模型依赖 ====
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn.functional as F

# ========= 复用你训练时的组件（SE_Block / Autoencoder / Channel 等） =========
mean = 0
std_dev = 0.1
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def AWGN_channel(x, snr, P=2):
    batch_size, channels, height, width = x.shape
    gamma = 10 ** (snr / 10.0)
    noise = torch.sqrt(P / gamma) * torch.randn(batch_size, channels, height, width).to(device)
    y = x + noise
    return y

def Fading_channel(x, snr, P=2):
    gamma = 10 ** (snr / 10.0)
    [batch_size, feature_length] = x.shape
    K = feature_length // 2
    h_I = torch.randn(batch_size, K).to(device)
    h_R = torch.randn(batch_size, K).to(device)
    h_com = torch.complex(h_I, h_R)
    x_com = torch.complex(x[:, 0:feature_length:2], x[:, 1:feature_length:2])
    y_com = h_com * x_com
    n_I = torch.sqrt(P / gamma) * torch.randn(batch_size, K).to(device)
    n_R = torch.sqrt(P / gamma) * torch.randn(batch_size, K).to(device)
    noise = torch.complex(n_I, n_R)
    y_add = y_com + noise
    y = y_add / h_com
    y_out = torch.zeros(batch_size, feature_length).to(device)
    y_out[:, 0:feature_length:2] = y.real
    y_out[:, 1:feature_length:2] = y.imag
    return y_out

def Combined_channel(x, snr, batch_size, channel, height, width):
    P = 2
    x_faded = Fading_channel(x, snr, P)
    x_faded = x_faded.view((batch_size, channel, height, width))
    snr = torch.randint(0, 28, (x_faded.shape[0], x_faded.shape[1], x_faded.shape[2], 1)).to(device)
    x_combined = AWGN_channel(x_faded, snr, P)
    return x_combined

def Channel(z, snr, channel_type, batch_size, channel, height, width):
    if channel_type == 'AWGN':
        z = AWGN_channel(z, snr)
    elif channel_type == 'Fading':
        z = Fading_channel(z, snr)
    elif channel_type == 'Combined_channel':
        z = Combined_channel(z, snr, batch_size, channel, height, width)
    return z

def mask_gen(weights, cr):
    position = round(cr * weights.size(1))
    weights_sorted, index = torch.sort(weights, dim=1)
    mask = torch.zeros_like(weights)
    for i in range(weights.size(0)):
        weight = weights_sorted[i, position - 1]
        for j in range(weights.size(1)):
            if weights[i, j] <= weight:
                mask[i, j] = 1
    return mask

class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(SE_Block, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, cr=0.8):
        b, c, h, w = x.size()
        y = self.gap(x).view(b, c)
        y = self.fc(y)
        mask = mask_gen(y, cr).view(b, c, 1, 1)
        return x * mask

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x, channel_type):
        x = self.encoder(x)
        noise = torch.randn_like(x) * std_dev + mean
        x = x + noise
        batch_size, channel, height, width = x.shape
        if channel_type in ['Fading', 'Combined_channel']:
            x = self.flatten(x)
            SNR = torch.randint(0, 28, (x.shape[0], 1)).to(device)
        else:
            SNR = torch.randint(0, 28, (x.shape[0], x.shape[1], x.shape[2], 1)).to(device)
        x = Channel(x, SNR, channel_type, batch_size, channel, height, width)
        x = x.view((batch_size, channel, height, width))
        x = self.decoder(x)
        return x

class SatelliteClassifierWithAttention(nn.Module):
    def __init__(self, num_classes):
        super(SatelliteClassifierWithAttention, self).__init__()
        self.resnet18 = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet18.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        in_features = self.resnet18.fc.in_features
        self.attention_module = SE_Block(in_features)
        self.antoencoder = Autoencoder()
        self.resnet18.fc = nn.Linear(in_features, num_classes)

    def forward(self, x, cr, channel_type):
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)
        x = self.resnet18.layer1(x)
        x = self.resnet18.layer2(x)
        x = self.resnet18.layer3(x)
        x = self.resnet18.layer4(x)
        x = self.attention_module(x, cr)
        x = self.antoencoder(x, channel_type)
        x = self.resnet18.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.resnet18.fc(x)
        return x

# ========= 预测工具函数 =========
def build_transform():
    # 必须与训练时一致
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

def load_classes_from_file(classes_file):
    # 每行一个类别名，顺序必须与训练时一致
    with open(classes_file, 'r', encoding='utf-8') as f:
        classes = [line.strip() for line in f if line.strip() != ""]
    return classes

def prepare_model(ckpt_path, num_classes):
    # 构建模型并加载权重
    model = SatelliteClassifierWithAttention(num_classes).to(device)
    state = torch.load(ckpt_path, map_location=device)
    # 兼容 strict=False 的方式，以防环境差异导致某些键不匹配
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[Warn] Missing keys when loading: {missing}")
    if unexpected:
        print(f"[Warn] Unexpected keys when loading: {unexpected}")
    model.eval()
    return model

@torch.no_grad()
def predict_image(model, img_path, transform, cr, channel_type, class_names=None, topk=1, seed=None):
    # 单张图片预测
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    img = Image.open(img_path).convert('RGB')
    x = transform(img).unsqueeze(0).to(device)  # (1,3,256,256)
    logits = model(x, cr, channel_type)         # (1,num_classes)
    probs = F.softmax(logits, dim=1).cpu().squeeze(0)  # (num_classes,)
    topk_vals, topk_ids = torch.topk(probs, k=topk)
    topk_vals = topk_vals.tolist()
    topk_ids = topk_ids.tolist()

    if class_names is None:
        labels = [str(i) for i in topk_ids]
    else:
        labels = [class_names[i] for i in topk_ids]

    return list(zip(labels, topk_vals))  # [(label, prob), ...]

@torch.no_grad()
def predict_folder(model, data_dir, transform, cr, channel_type, batch_size=64, num_workers=2, seed=None, csv_out=None):
    # 以 ImageFolder 的目录结构批量预测
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    ds = datasets.ImageFolder(root=data_dir, transform=transform)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    idx_to_class = {v: k for k, v in ds.class_to_idx.items()}  # 预测类别索引 -> 类别名

    results = []  # 保存 (path, pred_label, prob)
    for images, _ in dl:
        images = images.to(device)
        logits = model(images, cr, channel_type)          # (B,num_classes)
        probs = F.softmax(logits, dim=1).cpu()            # (B,num_classes)
        conf, pred = torch.max(probs, dim=1)              # (B,), (B,)
        for i in range(images.size(0)):
            # 需要拿到原始图像路径：ImageFolder的samples存了路径，DataLoader按顺序加载
            index_in_dataset = len(results) + i
            path, _ = ds.samples[index_in_dataset]
            label_name = idx_to_class[int(pred[i])]
            results.append((path, label_name, float(conf[i])))

    # 可选：存 CSV
    if csv_out:
        csv_path = Path(csv_out)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['path', 'pred', 'confidence'])
            for row in results:
                writer.writerow(row)
        print(f"[Info] Saved predictions to {csv_path}")

    return results, [idx_to_class[i] for i in range(len(idx_to_class))]

def parse_args():
    p = argparse.ArgumentParser(description="Prediction demo for SatelliteClassifierWithAttention")
    p.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint (.pth)')
    p.add_argument('--channel_type', choices=['AWGN', 'Fading', 'Combined_channel'], default='Combined_channel',
                   help='Channel type used in forward (must match training)')
    p.add_argument('--cr', type=float, default=0.8, help='Compression ratio used in SE_Block during forward')
    p.add_argument('--seed', type=int, default=None, help='Set this for deterministic predictions')

    # 单图/整目录二选一
    p.add_argument('--image', type=str, default=None, help='Path of a single image to predict')
    p.add_argument('--data_dir', type=str, default=None, help='ImageFolder dir to predict in batch')

    # 类别名来源（单图预测时建议提供；目录预测时会用 ImageFolder 的子目录名）
    p.add_argument('--classes_file', type=str, default=None, help='Optional: a text file with one class name per line')

    # 批量预测选项
    p.add_argument('--batch_size', type=int, default=64, help='Batch size for folder prediction')
    p.add_argument('--num_workers', type=int, default=2, help='Num workers for DataLoader')
    p.add_argument('--csv_out', type=str, default=None, help='Optional path to save CSV results for folder prediction')

    # 单图 top-k
    p.add_argument('--topk', type=int, default=3, help='Top-k results for single-image prediction')

    return p.parse_args()

def main():
    args = parse_args()

    # 预处理（需与训练一致）
    transform = build_transform()

    if args.image is None and args.data_dir is None:
        raise ValueError("Please provide either --image or --data_dir")

    # 确定 num_classes 与类名
    if args.data_dir is not None:
        # 批量：根据 ImageFolder 的子目录数量确定类别数
        ds_tmp = datasets.ImageFolder(root=args.data_dir, transform=transform)
        num_classes = len(ds_tmp.classes)
        class_names = ds_tmp.classes  # 用子目录名作为标签
    else:
        # 单图：需要 classes_file，或者你清楚 num_classes（若不给 classes_file，将用索引）
        if args.classes_file is not None:
            class_names = load_classes_from_file(args.classes_file)
            num_classes = len(class_names)
        else:
            # 如果没有 classes_file，就需要你明确告诉 num_classes；这里假设21（UCMerced）
            # 你可以修改为你的任务类别数
            print("[Warn] --classes_file not provided. Using default num_classes=21 and labels=0..20.")
            class_names = None
            num_classes = 21

    # 构建并加载模型
    model = prepare_model(args.checkpoint, num_classes)

    if args.image is not None:
        # 单张图片预测
        pairs = predict_image(
            model=model,
            img_path=args.image,
            transform=transform,
            cr=args.cr,
            channel_type=args.channel_type,
            class_names=class_names,
            topk=args.topk,
            seed=args.seed
        )
        print(f"[Result] {args.image}")
        for rank, (label, prob) in enumerate(pairs, 1):
            print(f"  Top{rank}: {label}  ({prob:.4f})")

    if args.data_dir is not None:
        # 整目录批量预测
        results, ordered_class_names = predict_folder(
            model=model,
            data_dir=args.data_dir,
            transform=transform,
            cr=args.cr,
            channel_type=args.channel_type,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            csv_out=args.csv_out
        )
        print(f"[Summary] {len(results)} images predicted.")
        print("[Classes]", ordered_class_names)

if __name__ == "__main__":
    main()
