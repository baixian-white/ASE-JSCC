#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
多层 Grad-CAM 可视化（适配 ASE-JSCC 完整模型）

更新功能：
- 支持从命令行传入“目标类别名称”（比如 overpass）
- 支持从命令行指定“要画哪些图片的 Grad-CAM”（用数据集索引，例：-i 10 23 99）
- 默认情况下：随机从该类别中抽若干张图片做 Grad-CAM

可视化内容：
- 对同一张图，在多层卷积特征上分别做 Grad-CAM
- 每张图输出 2 行：
    第 1 行：针对“预测类别”的 Grad-CAM（模型认为这张图是什么）
    第 2 行：针对“真实类别（命令行指定的类别）”的 Grad-CAM
  每行列结构：
    [原图 | layer2 CAM | layer3 CAM | layer4 CAM | ...]

使用示例：
(ASE-JSCC) $ python grad_cam_multilayer.py -c overpass
(ASE-JSCC) $ python grad_cam_multilayer.py -c overpass -n 6
(ASE-JSCC) $ python grad_cam_multilayer.py -c overpass -i 10 23 99
"""

import argparse
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights

# ==========================
# 一、基础配置（按需修改）
# ==========================

# 1) 训练好的完整模型权重
weight_path = "checkpoint/best_classifier_attention_auto_UCMerced_LandUse_Combined_channel_ResNet18_150epoch_0.8.pth"

# 2) 测试集路径
data_root = "data/UCMerced_LandUse-test"

# 3) 信道类型
channel_type_for_full = "Combined_channel"  # "AWGN" / "Fading" / "Combined_channel"

# 4) SE 压缩率
cr_for_attention = 0.8

# 5) 默认随机可视化多少张图（如果命令行不指定索引 && 不改数量，就用这个）
default_num_images_to_vis = 6

# 6) 选择要做 Grad-CAM 的层（名字: 取层的 lambda）
TARGET_LAYERS = {
    "layer2": lambda m: m.resnet18.layer2,  # 32x32，比较细腻
    "layer3": lambda m: m.resnet18.layer3,  # 16x16
    "layer4": lambda m: m.resnet18.layer4,  # 8x8，语义最强
    # "decoder": lambda m: m.antoencoder.decoder[6],  # 如需看 AE decoder 的最后一层，可打开
}

# 设备与输出目录
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
out_dir = Path("logs/ResNet18_gradcam_multi")
out_dir.mkdir(parents=True, exist_ok=True)

# 固定随机种子，方便复现
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


# ==========================
# 二、数据集 & 预处理
# ==========================

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

test_dataset = datasets.ImageFolder(root=data_root, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
CLASS_NAMES = test_dataset.classes

# 反归一化，用于可视化原图
inv_normalize = transforms.Normalize(
    mean=[-m / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)],
    std=[1.0 / s for s in IMAGENET_STD],
)


# ==========================
# 三、信道相关模块（与训练脚本一致）
# ==========================

mean = 0.0
std_dev = 0.1


def AWGN_channel(x, snr, P=2):
    gamma = 10 ** (snr / 10.0)
    noise = torch.sqrt(P / gamma) * torch.randn_like(x).to(device)
    return x + noise


def Fading_channel(x, snr, P=2):
    gamma = 10 ** (snr / 10.0)
    batch_size, feature_length = x.shape
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
    snr_map = torch.randint(0, 28, (x_faded.shape[0], x_faded.shape[1], x_faded.shape[2], 1)).to(device)
    return AWGN_channel(x_faded, snr_map, P)


def Channel(z, snr, channel_type, batch_size, channel, height, width):
    if channel_type == 'AWGN':
        return AWGN_channel(z, snr)
    elif channel_type == 'Fading':
        return Fading_channel(z, snr)
    elif channel_type == 'Combined_channel':
        return Combined_channel(z, snr, batch_size, channel, height, width)
    else:
        raise ValueError(f"Unknown channel_type: {channel_type}")


# ==========================
# 四、自编码器 & 注意力模块
# ==========================

def mask_gen(weights, cr):
    """
    根据 SE 的通道权重 + 压缩率 cr 生成 mask
    weights: [B, C]
    cr: 比例（例如 0.8）
    返回：mask [B, C]，0/1
    """
    position = round(cr * weights.size(1))
    weights_sorted, _ = torch.sort(weights, dim=1)
    mask = torch.zeros_like(weights)
    for i in range(weights.size(0)):
        threshold = weights_sorted[i, position - 1]
        mask[i] = (weights[i] <= threshold).float()
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
        """
        x: [B,C,H,W]
        返回：x * mask，按通道做抑制/保留
        """
        b, c, h, w = x.size()
        y = self.gap(x).view(b, c)
        y = self.fc(y)  # [B, C]，每个通道一个权重(0~1)
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
        """
        x: [B,512,8,8]（来自 resnet18.layer4+SE）
        """
        x = self.encoder(x)
        noise = torch.randn_like(x) * std_dev + mean
        x = x + noise

        batch_size, channel, height, width = x.shape

        if channel_type in ['Fading', 'Combined_channel']:
            x = self.flatten(x)  # [B, C*H*W]
            snr = torch.randint(0, 28, (x.shape[0], 1)).to(device)
        else:
            snr = torch.randint(0, 28, (x.shape[0], x.shape[1], x.shape[2], 1)).to(device)

        x = Channel(x, snr, channel_type, batch_size, channel, height, width)
        x = x.view((batch_size, channel, height, width))
        x = self.decoder(x)
        return x


# ==========================
# 五、完整模型定义
# ==========================

class SatelliteClassifierWithAttention(nn.Module):
    def __init__(self, num_classes):
        super(SatelliteClassifierWithAttention, self).__init__()
        self.resnet18 = resnet18(weights=ResNet18_Weights.DEFAULT)
        in_features = self.resnet18.fc.in_features

        self.attention_module = SE_Block(in_features)
        self.antoencoder = Autoencoder()

        # 替换 fc 为 num_classes 分类
        self.resnet18.fc = nn.Linear(in_features, num_classes)

    def forward(self, x, cr, channel_type):
        # ResNet18 前半部分（conv1 ~ layer4）
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)

        x = self.resnet18.layer1(x)
        x = self.resnet18.layer2(x)
        x = self.resnet18.layer3(x)
        x = self.resnet18.layer4(x)  # [B,512,8,8]

        # SE 通道注意力
        x = self.attention_module(x, cr=cr)

        # 自编码 + 信道
        x = self.antoencoder(x, channel_type)

        # 平均池化 + 全连接分类
        x = self.resnet18.avgpool(x)   # [B,512,1,1]
        x = x.view(x.size(0), -1)      # [B,512]
        x = self.resnet18.fc(x)        # [B,num_classes]
        return x


# ==========================
# 六、Grad-CAM 实现
# ==========================

class GradCAM:
    """
    针对单个 target_layer 的 Grad-CAM 实现：
    - 使用 forward hook 获取 activations
    - 使用 backward hook 获取 gradients
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        # 前向 hook：拿到该层输出特征图
        def forward_hook(module, inp, out):
            self.activations = out

        # 反向 hook：拿到该层输出的梯度
        def backward_hook(module, grad_in, grad_out):
            # grad_out[0] 对应该层输出的梯度，形状 [B,C,H,W]
            self.gradients = grad_out[0]

        self.fwd_handle = target_layer.register_forward_hook(forward_hook)
        # 用 full_backward_hook 兼容新版本
        self.bwd_handle = target_layer.register_full_backward_hook(backward_hook)

    def remove_hooks(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()

    def generate(self, input_tensor, class_idx, cr, channel_type):
        """
        生成单层 Grad-CAM

        input_tensor: [1,3,H,W]
        class_idx: 目标类别的索引
                   若为 None，则使用模型预测出的类别
        """
        # 清空旧的梯度
        self.model.zero_grad()

        # 正常前向传播
        outputs = self.model(input_tensor, cr, channel_type)  # [1, num_classes]

        # 若未指定类别，则默认用预测类别
        if class_idx is None:
            class_idx = outputs.argmax(dim=1).item()

        # 取这张图在该类别上的分数（logit）
        score = outputs[0, class_idx]

        # 对该分数做反向传播，梯度会流到 target_layer，
        # 同时触发 backward_hook，把梯度存到 self.gradients
        score.backward()

        # 取出该层的梯度和特征图（只取 batch 中第 0 个样本）
        grads = self.gradients[0]      # [C, H, W]
        acts = self.activations[0]     # [C, H, W]

        # 对每个通道的梯度在空间上做平均，得到权重 α_c
        weights = grads.mean(dim=(1, 2))  # [C]

        # 按 Grad-CAM 公式：CAM = ReLU( ∑_c α_c * A_c )
        cam = torch.zeros(acts.shape[1:], dtype=torch.float32).to(device)  # [H,W]
        for c, w in enumerate(weights):
            cam += w * acts[c, :, :]

        cam = torch.relu(cam)

        # 归一化到 [0,1]，方便可视化
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        # 转到 CPU + numpy
        cam = cam.detach().cpu().numpy()
        return cam, class_idx


# ==========================
# 七、可视化辅助函数
# ==========================

def tensor_to_numpy_image(tensor):
    """反归一化 + 转为 [H,W,3] numpy RGB"""
    tensor = inv_normalize(tensor).clamp(0.0, 1.0)
    return tensor.permute(1, 2, 0).cpu().numpy()


def upsample_cam_to_input(cam, input_size):
    """把 CAM 插值到与输入相同的空间尺寸 input_size=(H,W)"""
    cam_tensor = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0)  # [1,1,h,w]
    cam_upsampled = torch.nn.functional.interpolate(
        cam_tensor, size=input_size, mode='bilinear', align_corners=False
    )
    cam_upsampled = cam_upsampled.squeeze().numpy()
    cam_upsampled -= cam_upsampled.min()
    cam_upsampled /= (cam_upsampled.max() + 1e-8)
    return cam_upsampled


def save_multi_layer_gradcam_compare(orig_img, cams_pred, cams_true,
                                     true_label, pred_label,
                                     save_path, target_class_name):
    """
    两行对比可视化：
    第 1 行：针对“预测类别”的 Grad-CAM
    第 2 行：针对“真实类别（命令行指定的类别）”的 Grad-CAM

    列布局：
    [ 原图 | layer2 | layer3 | layer4 | ... ]
    """
    layer_names = list(cams_pred.keys())  # 假设两边 layer 一致
    num_layers = len(layer_names)

    fig, axes = plt.subplots(2, num_layers + 1, figsize=(4 * (num_layers + 1), 8))

    # --------- 第一行：Pred 类别 ---------
    axes[0, 0].imshow(orig_img)
    axes[0, 0].set_title(f"Original\nTrue: {true_label}\nPred: {pred_label}")
    axes[0, 0].axis("off")

    for j, layer_name in enumerate(layer_names, start=1):
        cam = cams_pred[layer_name]
        axes[0, j].imshow(orig_img)
        axes[0, j].imshow(cam, cmap="jet", alpha=0.4)
        axes[0, j].set_title(f"[Pred:{pred_label}]\n{layer_name}")
        axes[0, j].axis("off")

    # --------- 第二行：True 目标类别 ---------
    axes[1, 0].imshow(orig_img)
    axes[1, 0].set_title(f"Original\nTarget class: {target_class_name}")
    axes[1, 0].axis("off")

    for j, layer_name in enumerate(layer_names, start=1):
        cam = cams_true[layer_name]
        axes[1, j].imshow(orig_img)
        axes[1, j].imshow(cam, cmap="jet", alpha=0.4)
        axes[1, j].set_title(f"[True:{target_class_name}]\n{layer_name}")
        axes[1, j].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"[Multi-layer Grad-CAM 对比] 已保存: {save_path}")


# ==========================
# 八、主流程
# ==========================

def main(args):
    """
    args.target_class: 目标类别名称（例如 "overpass"，对应数据集文件夹名）
    args.indices:      要可视化的样本索引列表（数据集下标），如果为 None 则随机抽样
    args.num_images:   随机模式下要可视化的图片数量
    """
    num_classes = len(CLASS_NAMES)
    model = SatelliteClassifierWithAttention(num_classes=num_classes).to(device)

    # 加载训练好的权重
    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"已从 {weight_path} 加载模型。")
    print(f"共有 {num_classes} 个类别：{CLASS_NAMES}")

    # 查找目标类别索引
    target_class_name = args.target_class
    if target_class_name not in CLASS_NAMES:
        raise ValueError(f"找不到类别 {target_class_name}，当前类别有：{CLASS_NAMES}")
    target_class_idx = CLASS_NAMES.index(target_class_name)
    print(f"目标分析类别: {target_class_name}, 索引: {target_class_idx}")

    # 预先构建每个层对应的 GradCAM 对象
    gradcam_dict = {}
    for name, get_layer in TARGET_LAYERS.items():
        target_layer = get_layer(model)
        gradcam_dict[name] = GradCAM(model, target_layer)
    print(f"将对以下层做 Grad-CAM: {list(TARGET_LAYERS.keys())}")

    # =============== 选择要可视化的样本索引 ===============
    if args.indices is not None and len(args.indices) > 0:
        # 用户通过命令行显式指定索引
        indices = []
        skipped = 0
        for idx in args.indices:
            if idx < 0 or idx >= len(test_dataset):
                print(f"[警告] 索引 {idx} 超出数据集范围 [0, {len(test_dataset)-1}]，跳过。")
                skipped += 1
                continue
            img, lbl = test_dataset[idx]
            if lbl != target_class_idx:
                print(f"[警告] 索引 {idx} 的真实类别为 {CLASS_NAMES[lbl]} "
                      f"≠ 目标类别 {target_class_name}，暂时跳过（如需强制可视化，可以改代码）。")
                skipped += 1
                continue
            indices.append(idx)

        if len(indices) == 0:
            raise ValueError("命令行指定的索引中，没有任何一个属于目标类别，无法可视化。")
        print(f"根据命令行指定索引，将可视化 {len(indices)} 张图（{skipped} 个被跳过）。")

    else:
        # 未指定索引，则随机从目标类别中抽样
        over_indices = [i for i, (_, lbl) in enumerate(test_dataset) if lbl == target_class_idx]
        if len(over_indices) == 0:
            raise ValueError(f"测试集中没有类别 {target_class_name} 的样本，请检查 data_root。")

        random.shuffle(over_indices)
        num_to_take = min(args.num_images, len(over_indices))
        indices = over_indices[:num_to_take]
        print(f"随机从类别 {target_class_name} 中抽取 {len(indices)} 张图像进行可视化。")

    print("将要可视化的样本索引：", indices)

    # =============== 对每一张样本生成 Grad-CAM ===============
    for k, idx in enumerate(indices, start=1):
        img_tensor, label = test_dataset[idx]          # [3,H,W]
        input_tensor = img_tensor.unsqueeze(0).to(device)

        # 先跑一遍，得到预测类别（所有 Grad-CAM 共用）
        with torch.no_grad():
            outputs = model(input_tensor, cr_for_attention, channel_type_for_full)
            pred_idx = outputs.argmax(dim=1).item()

        true_name = CLASS_NAMES[label]
        pred_name = CLASS_NAMES[pred_idx]
        orig_img = tensor_to_numpy_image(img_tensor)
        H, W = orig_img.shape[:2]

        print(f"[{k}/{len(indices)}] idx={idx}, True={true_name}, Pred={pred_name}")

        # 针对“预测类别”的 Grad-CAM
        cams_pred = {}
        for layer_name, cam_obj in gradcam_dict.items():
            cam_pred, _ = cam_obj.generate(
                input_tensor=input_tensor,
                class_idx=pred_idx,
                cr=cr_for_attention,
                channel_type=channel_type_for_full,
            )
            cam_pred_up = upsample_cam_to_input(cam_pred, (H, W))
            cams_pred[layer_name] = cam_pred_up

        # 针对“真实目标类别（命令行指定）”的 Grad-CAM
        cams_true = {}
        for layer_name, cam_obj in gradcam_dict.items():
            cam_true, _ = cam_obj.generate(
                input_tensor=input_tensor,
                class_idx=target_class_idx,
                cr=cr_for_attention,
                channel_type=channel_type_for_full,
            )
            cam_true_up = upsample_cam_to_input(cam_true, (H, W))
            cams_true[layer_name] = cam_true_up

        # 保存对比图
        save_path = out_dir / f"{target_class_name}_gradcam_idx{idx}_true-{true_name}_pred-{pred_name}.png"
        save_multi_layer_gradcam_compare(
            orig_img, cams_pred, cams_true,
            true_label=true_name, pred_label=pred_name,
            save_path=save_path,
            target_class_name=target_class_name,
        )

    # 清理 hook
    for cam_obj in gradcam_dict.values():
        cam_obj.remove_hooks()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多层 Grad-CAM 可视化（ASE-JSCC）")
    parser.add_argument(
        "-c", "--target-class",
        type=str,
        required=True,
        help="要分析的目标类别名称（必须等于数据集文件夹名，如 overpass）"
    )
    parser.add_argument(
        "-i", "--indices",
        type=int,
        nargs="+",
        default=None,
        help="可选：指定要可视化的样本索引（数据集下标），例如：-i 10 23 99"
    )
    parser.add_argument(
        "-n", "--num-images",
        type=int,
        default=default_num_images_to_vis,
        help="随机模式下：从目标类别中抽取多少张图像进行可视化"
    )

    args = parser.parse_args()
    main(args)
