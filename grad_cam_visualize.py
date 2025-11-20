#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
多层 Grad-CAM 可视化（适配 ASE-JSCC 完整模型）

功能：
- 从 UCMerced 测试集中随机选若干张图
- 对同一张图，在多层卷积特征上分别做 Grad-CAM
- 每张图输出：
    [原图] + [layer2 Overlay] + [layer3 Overlay] + [decoder Overlay] ...
- 用于观察不同深度特征的关注区域（浅层更细腻）

使用方法：
(ASE-JSCC) $ python grad_cam_multilayer.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random

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

# 5) 随机可视化多少张图
num_images_to_vis = 6

# 6) 选择要做 Grad-CAM 的层（名字: 取层的 lambda）
#   建议至少包含 layer2（32x32），也可以加 layer3（16x16）、decoder 等
TARGET_LAYERS = {
    "layer2": lambda m: m.resnet18.layer2,                # 32x32，比较细腻
    "layer3": lambda m: m.resnet18.layer3,                # 16x16
    "layer4": lambda m: m.resnet18.layer4,                # 8x8，语义最强
    # "decoder": lambda m: m.antoencoder.decoder[-2],     # 8x8，信道恢复后
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
            snr = torch.randint(0, 28, (x.shape[0], 1)).to(device)
        else:
            snr = torch.randint(0, 28, (x.shape[0], x.shape[1], x.shape[2], 1)).to(device)

        x = Channel(x, snr, channel_type, batch_size, channel, height, width)
        x = x.view((batch_size, channel, height, width))
        x = self.decoder(x)
        return x

def mask_gen(weights, cr):
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
        b, c, h, w = x.size()
        y = self.gap(x).view(b, c)
        y = self.fc(y)
        mask = mask_gen(y, cr).view(b, c, 1, 1)
        return x * mask


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

        x = self.attention_module(x, cr=cr)
        x = self.antoencoder(x, channel_type)

        x = self.resnet18.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.resnet18.fc(x)
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

        def forward_hook(module, inp, out):
            self.activations = out

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.fwd_handle = target_layer.register_forward_hook(forward_hook)
        # 推荐用 full_backward_hook，兼容新版本
        self.bwd_handle = target_layer.register_full_backward_hook(backward_hook)

    def remove_hooks(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()

    def generate(self, input_tensor, class_idx, cr, channel_type):
        """
        input_tensor: [1,3,H,W]
        class_idx: 若为 None，则使用预测类别
        """
        self.model.zero_grad()
        outputs = self.model(input_tensor, cr, channel_type)  # [1, num_classes]

        if class_idx is None:
            class_idx = outputs.argmax(dim=1).item()

        score = outputs[0, class_idx]
        score.backward()

        grads = self.gradients[0]      # [C, H, W]
        acts = self.activations[0]     # [C, H, W]

        weights = grads.mean(dim=(1, 2))  # [C]

        cam = torch.zeros(acts.shape[1:], dtype=torch.float32).to(device)
        for c, w in enumerate(weights):
            cam += w * acts[c, :, :]

        cam = torch.relu(cam)
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
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
    """把 CAM 插值到与输入相同的空间尺寸"""
    cam_tensor = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0)  # [1,1,h,w]
    cam_upsampled = torch.nn.functional.interpolate(
        cam_tensor, size=input_size, mode='bilinear', align_corners=False
    )
    cam_upsampled = cam_upsampled.squeeze().numpy()
    cam_upsampled -= cam_upsampled.min()
    cam_upsampled /= (cam_upsampled.max() + 1e-8)
    return cam_upsampled

def save_multi_layer_gradcam(orig_img, cams_dict, true_label, pred_label, save_path):
    """
    cams_dict: {layer_name: cam_upsampled_numpy}
    输出图像结构：
    [ 原图 | layer2 | layer3 | layer4 | ... ]
    """
    num_layers = len(cams_dict)
    fig, axes = plt.subplots(1, num_layers + 1, figsize=(4 * (num_layers + 1), 4))

    # 第一列：原图
    axes[0].imshow(orig_img)
    axes[0].set_title(f"Original\nTrue: {true_label}\nPred: {pred_label}")
    axes[0].axis("off")

    # 后面每一列：不同层的 Overlay
    for i, (layer_name, cam) in enumerate(cams_dict.items(), start=1):
        axes[i].imshow(orig_img)
        axes[i].imshow(cam, cmap="jet", alpha=0.4)
        axes[i].set_title(f"{layer_name} CAM")
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"[Multi-layer Grad-CAM] 已保存: {save_path}")


# ==========================
# 八、主流程
# ==========================

def main():
    num_classes = len(CLASS_NAMES)
    model = SatelliteClassifierWithAttention(num_classes=num_classes).to(device)

    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"已从 {weight_path} 加载模型。")

    # 预先构建每个层对应的 GradCAM 对象
    gradcam_dict = {}
    for name, get_layer in TARGET_LAYERS.items():
        target_layer = get_layer(model)
        gradcam_dict[name] = GradCAM(model, target_layer)

    # 随机挑选若干张图片
    indices = list(range(len(test_dataset)))
    random.shuffle(indices)
    indices = indices[:min(num_images_to_vis, len(indices))]

    for idx in indices:
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

        # 对每一个 target layer 生成 CAM，并插值到输入尺寸
        cams_upsampled = {}
        for layer_name, cam_obj in gradcam_dict.items():
            cam, _ = cam_obj.generate(
                input_tensor=input_tensor,
                class_idx=pred_idx,  # 针对预测类别做 Grad-CAM
                cr=cr_for_attention,
                channel_type=channel_type_for_full,
            )
            cam_up = upsample_cam_to_input(cam, (H, W))
            cams_upsampled[layer_name] = cam_up

        # 保存一张拼好的多层图
        save_path = out_dir / f"multi_gradcam_idx{idx}_true-{true_name}_pred-{pred_name}.png"
        save_multi_layer_gradcam(orig_img, cams_upsampled, true_name, pred_name, save_path)

    # 清理 hook（可选）
    for cam_obj in gradcam_dict.values():
        cam_obj.remove_hooks()


if __name__ == "__main__":
    main()

