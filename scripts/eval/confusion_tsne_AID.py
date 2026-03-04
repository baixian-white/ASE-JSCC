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

def get_project_root() -> Path:
    current = Path(__file__).resolve().parent
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return Path.cwd()


PROJECT_ROOT = get_project_root()

# ==========================
# 0. 颜色 & Marker 设置（30 类也够用，不够会自动循环）
# ==========================

COLOR_LIST = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#008080",
    "#e6beff", "#9a6324", "#fffac8", "#800000", "#aaffc3",
    "#808000", "#ffd8b1", "#000075", "#808080", "#ffe119",
    "#469990",
]
MARKER_LIST = ['o', 's', '^', 'v', 'D', 'P', 'X']

# ==========================
# 一、基础配置：改成 AID
# ==========================

# 1) AID 实验的根目录（和训练脚本 get_exp_dirs 对齐）
#    比如训练时用的是：AID_150_combine_0.8
exp_dir = PROJECT_ROOT / "AID_150_combine_0.8"

# 2) 模型权重路径（建议用 best_* 那个）
weight_path = (
    exp_dir
    / "checkpoint"
    / "best_classifier_attention_auto_AID_Combined_channel_ResNet18_150epoch_0.8.pth"
)

# 3) AID 测试集路径（就是你刚才截图的那个 AID-test）
data_root = PROJECT_ROOT / "data" / "AID-test"

# 4) 信道类型（完整流水线阶段用）
channel_type_for_full = "Combined_channel"

# 5) SE Block 压缩率 cr
cr_for_attention = 0.8

# 6) batch size
batch_size = 64

# 设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 输出目录：直接复用训练时的 logs/ResNet18
out_dir = Path(exp_dir) / "logs" / "ResNet18"
out_dir.mkdir(parents=True, exist_ok=True)

# ==========================
# 二、AID 测试集 & 预处理
# ==========================

test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
])

test_dataset = datasets.ImageFolder(root=str(data_root), transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# AID 的类别名（30 类）自动从文件夹名读出来
CLASS_NAMES = test_dataset.classes

# ==========================
# 三、信道相关模块（跟训练脚本完全一致）
# ==========================

mean = 0.0
std_dev = 0.1


def AWGN_channel(x, snr, P=2):
    batch_size, channels, height, width = x.shape
    gamma = 10 ** (snr / 10.0)
    noise = torch.sqrt(P / gamma) * torch.randn_like(x).to(device)
    y = x + noise
    return y


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

    snr_map = torch.randint(0, 28,
                            (x_faded.shape[0], x_faded.shape[1], x_faded.shape[2], 1)
                            ).to(device)
    x_combined = AWGN_channel(x_faded, snr_map, P)
    return x_combined


def Channel(z, snr, channel_type, batch_size, channel, height, width):
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
# 四、自编码器 & SE 注意力
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
            snr = torch.randint(0, 28,
                                (x.shape[0], x.shape[1], x.shape[2], 1)
                                ).to(device)

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
# 五、完整模型（跟训练版 AID 一样）
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
# 六、前向工具函数（Full + Feature）
# ==========================


def forward_full_pipeline(model, x, cr, channel_type):
    return model(x, cr, channel_type)


def forward_full_pipeline_feature(model, x, cr, channel_type):
    x = model.resnet18.conv1(x)
    x = model.resnet18.bn1(x)
    x = model.resnet18.relu(x)
    x = model.resnet18.maxpool(x)

    x = model.resnet18.layer1(x)
    x = model.resnet18.layer2(x)
    x = model.resnet18.layer3(x)
    x = model.resnet18.layer4(x)

    x = model.attention_module(x, cr=cr)
    x = model.antoencoder(x, channel_type)

    x = model.resnet18.avgpool(x)
    x = x.view(x.size(0), -1)
    return x

# ==========================
# 七、特征抽取 & 画图
# ==========================


def extract_features_and_labels(model, data_loader, feature_fn):
    model.eval()
    all_feats, all_labels = [], []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            feats = feature_fn(model, images)  # [B, D]
            feats = feats.detach().cpu().numpy()
            all_feats.append(feats)
            all_labels.append(labels.cpu().numpy())

    features = np.concatenate(all_feats, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return features, labels


def plot_tsne_for_full_pipeline(model, cr, channel_type, save_path_png):
    print("开始提取 AID Full Pipeline 特征并进行 t-SNE 降维...")

    features, labels = extract_features_and_labels(
        model=model,
        data_loader=test_loader,
        feature_fn=lambda m, x: forward_full_pipeline_feature(m, x, cr, channel_type)
    )

    tsne = TSNE(
        n_components=2,
        init="pca",
        random_state=0,
        learning_rate="auto"
    )
    embeddings = tsne.fit_transform(features)

    plt.figure(figsize=(8, 8))
    num_classes = len(CLASS_NAMES)

    for class_idx in range(num_classes):
        idxs = (labels == class_idx)
        if np.sum(idxs) == 0:
            continue

        color = COLOR_LIST[class_idx % len(COLOR_LIST)]
        marker = MARKER_LIST[(class_idx // len(COLOR_LIST)) % len(MARKER_LIST)]

        plt.scatter(
            embeddings[idxs, 0],
            embeddings[idxs, 1],
            s=18,
            alpha=0.9,
            color=color,
            marker=marker,
            edgecolors="black",
            linewidth=0.2,
            label=CLASS_NAMES[class_idx]
        )

    plt.title(f"AID t-SNE - Full Pipeline (cr={cr}, channel={channel_type})")
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


def eval_and_plot_confusion(model, forward_fn, stage_name, save_path_png):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = forward_fn(model, images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    cm = confusion_matrix(
        all_labels, all_preds,
        labels=list(range(len(CLASS_NAMES)))
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(len(CLASS_NAMES)))
    ax.set_yticks(np.arange(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"AID Confusion Matrix - {stage_name}")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, int(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=5
            )

    fig.tight_layout()
    plt.savefig(save_path_png, dpi=300)
    plt.close(fig)
    print(f"[{stage_name}] 混淆矩阵已保存到: {save_path_png}")

# ==========================
# 八、主函数
# ==========================


def main():
    num_classes = len(CLASS_NAMES)
    model = SatelliteClassifierWithAttention(num_classes=num_classes).to(device)

    state = torch.load(str(weight_path), map_location=device)
    model.load_state_dict(state, strict=True)
    print(f"已从 {weight_path} 加载 AID 模型参数。")

    # 1) 混淆矩阵
    cm_path = out_dir / f"confusion_AID_stage3_full_{channel_type_for_full}.png"
    eval_and_plot_confusion(
        model=model,
        forward_fn=lambda m, x: forward_full_pipeline(m, x, cr_for_attention, channel_type_for_full),
        stage_name=f"Full Pipeline (cr={cr_for_attention}, channel={channel_type_for_full})",
        save_path_png=str(cm_path)
    )

    # 2) t-SNE
    tsne_path = out_dir / f"tsne_AID_stage3_full_{channel_type_for_full}.png"
    plot_tsne_for_full_pipeline(
        model=model,
        cr=cr_for_attention,
        channel_type=channel_type_for_full,
        save_path_png=str(tsne_path)
    )


if __name__ == "__main__":
    main()
