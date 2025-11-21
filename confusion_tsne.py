import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

# 为 21 个类别手动指定一组差异非常大的颜色（来自 ColorBrewer/Tab 调色板混合）
COLOR_LIST = [
    "#e6194b",  # 0  红
    "#3cb44b",  # 1  绿
    "#4363d8",  # 2  蓝
    "#f58231",  # 3  橙
    "#911eb4",  # 4  紫
    "#46f0f0",  # 5  青
    "#f032e6",  # 6  粉紫
    "#bcf60c",  # 7  黄绿
    "#fabebe",  # 8  浅粉
    "#008080",  # 9  深青
    "#e6beff",  # 10 淡紫
    "#9a6324",  # 11 棕
    "#fffac8",  # 12 很浅黄
    "#800000",  # 13 深红
    "#aaffc3",  # 14 薄荷绿
    "#808000",  # 15 橄榄
    "#ffd8b1",  # 16 肉色
    "#000075",  # 17 深蓝
    "#808080",  # 18 灰
    "#ffe119",  # 19 亮黄
    "#469990",  # 20 蓝绿
]

# 再准备几种不同的点形状，颜色接近时还能靠形状区分
MARKER_LIST = ['o', 's', '^', 'v', 'D', 'P', 'X']


# ==========================
# 一、基础配置（按需改这几行）
# ==========================

# 1) 模型权重路径（建议用 best_*.pth ）
weight_path = "checkpoint/best_classifier_attention_auto_UCMerced_LandUse_Combined_channel_ResNet18_150epoch_0.8.pth"

# 2) 测试集路径（和训练脚本里 valid_dataset 一样就行）
data_root = "data/UCMerced_LandUse-test"

# 3) 信道类型（只作用在“完整模型阶段”）
#    可选：'AWGN' / 'Fading' / 'Combined_channel'
channel_type_for_full = "Combined_channel"

# 4) SE Block 使用的压缩率 cr（完整模型 & 注意力阶段需要）
cr_for_attention = 0.8

# 5) batch size（测试用，不影响结果，只影响速度/显存）
batch_size = 64

# 设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 输出目录
Path("logs/ResNet18").mkdir(parents=True, exist_ok=True)


# ==========================
# 二、数据集 & 预处理
# ==========================

test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
])

test_dataset = datasets.ImageFolder(root=data_root, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
CLASS_NAMES = test_dataset.classes  # 类别名列表，例如 21 个场景类别


# ==========================
# 三、信道相关模块（和训练脚本保持一致）
# ==========================

mean = 0.0          # 自编码器中加噪的均值
std_dev = 0.1       # 自编码器中加噪的标准差

def AWGN_channel(x, snr, P=2):
    """
    模拟 AWGN 信道
    x: (B, C, H, W)
    snr: 可以是标量 or 与 x 形状广播兼容的张量（单位：dB）
    P: 符号平均功率，这里默认 2，和你训练脚本一致
    """
    batch_size, channels, height, width = x.shape
    gamma = 10 ** (snr / 10.0)  # SNR 线性值
    noise = torch.sqrt(P / gamma) * torch.randn_like(x).to(device)
    y = x + noise
    return y


def Fading_channel(x, snr, P=2):
    """
    模拟平坦瑞利衰落信道 + 复高斯噪声
    输入 x: (B, F) 一维特征
    """
    gamma = 10 ** (snr / 10.0)
    batch_size, feature_length = x.shape
    K = feature_length // 2

    # 生成复衰落系数 h_com ~ CN(0,1)
    h_I = torch.randn(batch_size, K).to(device)
    h_R = torch.randn(batch_size, K).to(device)
    h_com = torch.complex(h_I, h_R)

    # 将原本实数特征两两合并成复数：偶数索引为实部，奇数为虚部
    x_com = torch.complex(x[:, 0:feature_length:2], x[:, 1:feature_length:2])

    # 衰落
    y_com = h_com * x_com

    # 加复高斯噪声
    n_I = torch.sqrt(P / gamma) * torch.randn(batch_size, K).to(device)
    n_R = torch.sqrt(P / gamma) * torch.randn(batch_size, K).to(device)
    noise = torch.complex(n_I, n_R)

    y_add = y_com + noise

    # 简单均衡：除以 h_com
    y = y_add / h_com

    # 还原为实数向量 [Re, Im, Re, Im, ...]
    y_out = torch.zeros(batch_size, feature_length).to(device)
    y_out[:, 0:feature_length:2] = y.real
    y_out[:, 1:feature_length:2] = y.imag
    return y_out


def Combined_channel(x, snr, batch_size, channel, height, width):
    """
    组合信道：
    1) 先走 Fading_channel（在展平的一维特征上）
    2) 再还原成 4D 特征图
    3) 再按 (B,C,H,1) 生成位置相关 SNR，叠加一次 AWGN
    """
    P = 2
    x_faded = Fading_channel(x, snr, P)  # (B, F)
    x_faded = x_faded.view((batch_size, channel, height, width))  # (B, 32, H, W)

    # 每个样本、每个通道、每一行一个 SNR，沿 W 共享
    snr_map = torch.randint(0, 28, (x_faded.shape[0], x_faded.shape[1], x_faded.shape[2], 1)).to(device)
    x_combined = AWGN_channel(x_faded, snr_map, P)
    return x_combined


def Channel(z, snr, channel_type, batch_size, channel, height, width):
    """
    根据 channel_type 选择具体信道模型
    z: 对于 Fading/Combined 是展平的一维向量；对于 AWGN 是 4D 特征图
    """
    if channel_type == 'AWGN':
        z = AWGN_channel(z, snr)
    elif channel_type == 'Fading':
        z = Fading_channel(z, snr)
    elif channel_type == 'Combined_channel':
        z = Combined_channel(z, snr, batch_size, channel, height, width)
    else:
        raise ValueError(f"Unknown channel_type: {channel_type}")
    return z


# ==========================
# 四、自编码器 & 注意力模块
# ==========================

class Autoencoder(nn.Module):
    """
    和训练脚本一致的自编码器：
    512 通道特征 -> 32 通道压缩 -> 加噪 -> 通过信道 -> 解码回 512 通道
    """
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
        x: (B, 512, H, W)
        """
        x = self.encoder(x)  # (B, 32, H, W)

        # 在编码空间加 AWGN 噪声
        noise = torch.randn_like(x) * std_dev + mean
        x = x + noise

        batch_size, channel, height, width = x.shape

        # Fading / Combined: 展平再走信道
        if channel_type in ['Fading', 'Combined_channel']:
            x = self.flatten(x)  # (B, F)
            snr = torch.randint(0, 28, (x.shape[0], 1)).to(device)
        else:
            # AWGN: 保持 4D，在 (B,32,H,1) 上采样 SNR
            snr = torch.randint(0, 28, (x.shape[0], x.shape[1], x.shape[2], 1)).to(device)

        x = Channel(x, snr, channel_type, batch_size, channel, height, width)
        x = x.view((batch_size, channel, height, width))
        x = self.decoder(x)  # (B, 512, H, W)
        return x


def mask_gen(weights, cr):
    """
    根据通道注意力权重生成二值掩码：
    - 每个样本排序
    - 取最小的 cr * C 个通道置 1，其余 0
    """
    position = round(cr * weights.size(1))
    weights_sorted, _ = torch.sort(weights, dim=1)
    mask = torch.zeros_like(weights)

    for i in range(weights.size(0)):
        threshold = weights_sorted[i, position - 1]
        mask[i] = (weights[i] <= threshold).float()

    return mask


class SE_Block(nn.Module):
    """
    通道注意力模块：SE + 二值掩码
    """
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
# 五、完整模型定义（与训练一致）
# ==========================

class SatelliteClassifierWithAttention(nn.Module):
    def __init__(self, num_classes):
        super(SatelliteClassifierWithAttention, self).__init__()
        # 主干 ResNet18
        self.resnet18 = resnet18(weights=ResNet18_Weights.DEFAULT)
        in_features = self.resnet18.fc.in_features
        # 通道注意力
        self.attention_module = SE_Block(in_features)
        # 自编码器（压缩 + 信道 + 解码）
        self.antoencoder = Autoencoder()
        # 分类头
        self.resnet18.fc = nn.Linear(in_features, num_classes)

    def forward(self, x, cr, channel_type):
        # 手工展开 ResNet18 的前向，方便在中间插入注意力和信道
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)

        x = self.resnet18.layer1(x)
        x = self.resnet18.layer2(x)
        x = self.resnet18.layer3(x)
        x = self.resnet18.layer4(x)   # (B, 512, H, W)

        # 1) SE 通道注意力 + 二值掩码
        x = self.attention_module(x, cr=cr)

        # 2) 自编码器 + 信道
        x = self.antoencoder(x, channel_type)

        # 3) 平均池化 + 分类头
        x = self.resnet18.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.resnet18.fc(x)
        return x


# ==========================
# 六、前向封装
# ==========================

def forward_full_pipeline(model, x, cr, channel_type):
    """
    阶段3：完整流水线（ResNet18 + 注意力 + 自编码 + 信道）
    """
    return model(x, cr, channel_type)

def forward_full_pipeline_feature(model, x, cr, channel_type):
    """
    Full Pipeline 的特征提取版：
    - 和 forward_full_pipeline 基本一样
    - 但在 fc 之前停下，只返回特征向量 (B, 512)
    """
    # ===== ResNet18 前半部分 =====
    x = model.resnet18.conv1(x)
    x = model.resnet18.bn1(x)
    x = model.resnet18.relu(x)
    x = model.resnet18.maxpool(x)

    x = model.resnet18.layer1(x)
    x = model.resnet18.layer2(x)
    x = model.resnet18.layer3(x)
    x = model.resnet18.layer4(x)   # (B, 512, H, W)

    # ===== 通道注意力 =====
    x = model.attention_module(x, cr=cr)

    # ===== 自编码器 + 信道 =====
    x = model.antoencoder(x, channel_type)

    # ===== 平均池化，展开为 512 维特征 =====
    x = model.resnet18.avgpool(x)
    x = x.view(x.size(0), -1)  # (B, 512)
    return x


def extract_features_and_labels(model, data_loader, feature_fn):
    """
    跑一遍 data_loader，抽取全部样本的特征和标签，用于 t-SNE。

    :param model: 已经 to(device) 的模型
    :param data_loader: 一般就是 test_loader
    :param feature_fn: 一个函数，接受 (model, images) 返回特征张量 [B, D]
    :return:
        features: [N, D] 的 numpy 数组
        labels:   [N] 的 numpy 数组（类别索引）
    """
    model.eval()
    all_feats = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            feats = feature_fn(model, images)      # [B, D]
            feats = feats.detach().cpu().numpy()   # 转 numpy
            all_feats.append(feats)
            all_labels.append(labels.cpu().numpy())

    features = np.concatenate(all_feats, axis=0) #把多个数组按指定维度拼接在一起
    labels = np.concatenate(all_labels, axis=0)
    return features, labels


def plot_tsne_for_full_pipeline(model, cr, channel_type, save_path_png):
    """
    对完整模型（Stage3）在测试集上做 t-SNE 可视化，并保存图像。
    """
    print("开始提取 Full Pipeline 特征并进行 t-SNE 降维...")

    # 1) 抽取特征和标签（fc 前 512 维特征）
    features, labels = extract_features_and_labels(
        model=model,
        data_loader=test_loader,
        feature_fn=lambda m, x: forward_full_pipeline_feature(m, x, cr, channel_type)
    )

    # 2) t-SNE 降到 2 维
    tsne = TSNE(
        n_components=2,
        init="pca",          # 先用 PCA 初始化，收敛更稳定
        random_state=0,
        learning_rate="auto"
    )
    embeddings = tsne.fit_transform(features)   # [N, 2]

     # 3) 画散点图（使用 seaborn 的 husl 调色板，21 种明显不同的颜色）
    plt.figure(figsize=(8, 8))
    num_classes = len(CLASS_NAMES)

    for class_idx in range(num_classes):
        idxs = (labels == class_idx)
        if np.sum(idxs) == 0:
            continue

        # 颜色：从 21 个手动颜色里取（你的类刚好是 21 个）
        color = COLOR_LIST[class_idx % len(COLOR_LIST)]
        # 形状：如果以后类别多于颜色数，会自动轮换 marker
        marker = MARKER_LIST[(class_idx // len(COLOR_LIST)) % len(MARKER_LIST)]

        plt.scatter(
            embeddings[idxs, 0],
            embeddings[idxs, 1],
            s=18,                # 点稍微画大一点，更好分
            alpha=0.9,
            color=color,
            marker=marker,
            edgecolors="black",  # 黑色细边，和背景/其他类拉开
            linewidth=0.2,
            label=CLASS_NAMES[class_idx]
        )

    plt.title(f"t-SNE - Full Pipeline (cr={cr}, channel={channel_type})")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.legend(
        bbox_to_anchor=(1.05, 1.0),
        loc="upper left",
        fontsize=6,
        ncol=1
    )
    plt.tight_layout()
    plt.savefig(save_path_png, dpi=300)
    plt.close()
    print(f"[t-SNE] 已保存到: {save_path_png}")

# ==========================
# 七、通用的“计算+画图”函数
# ==========================

def eval_and_plot_confusion(model, forward_fn, stage_name, save_path_png):
    """
    在 test_loader 上跑一遍，计算混淆矩阵并画图。
    forward_fn: 一个函数，形如 logits = forward_fn(model, images)
    stage_name: 在图上显示用的标题，例如 "Backbone" / "Backbone+Attention" / "Full"
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = forward_fn(model, images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(CLASS_NAMES))))

    # 画图
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(len(CLASS_NAMES)))
    ax.set_yticks(np.arange(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Confusion Matrix - {stage_name}")

    # 在方格中写数字
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, int(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=6
            )

    fig.tight_layout()
    plt.savefig(save_path_png, dpi=300)
    plt.close(fig)
    print(f"[{stage_name}] 混淆矩阵已保存到: {save_path_png}")


# ==========================
# 八、主流程：加载模型 + 三阶段画图
# ==========================

def main():
    num_classes = len(CLASS_NAMES)
    model = SatelliteClassifierWithAttention(num_classes=num_classes).to(device)

    # 加载权重
    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state, strict=True)
    print(f"已从 {weight_path} 加载模型参数。")

    # 完整流水线（注意力 + 自编码 + 指定信道）
    eval_and_plot_confusion(
        model=model,
        forward_fn=lambda m, x: forward_full_pipeline(m, x, cr_for_attention, channel_type_for_full),
        stage_name=f"Stage3: Full Pipeline (cr={cr_for_attention}, channel={channel_type_for_full})",
        save_path_png=f"logs/ResNet18/confusion_stage3_full_{channel_type_for_full}.png"
    )
     # ===== 对  Full Pipeline 画一张 t-SNE 图 =====
    tsne_path = f"logs/ResNet18/tsne_stage3_full_{channel_type_for_full}.png"
    plot_tsne_for_full_pipeline(
        model=model,
        cr=cr_for_attention,
        channel_type=channel_type_for_full,
        save_path_png=tsne_path
    )


if __name__ == "__main__":
    main()
