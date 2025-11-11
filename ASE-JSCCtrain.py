import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18,ResNet18_Weights  # 以ResNet18为例，也可以根据实际情况选择其他模型
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
import math

# 注意，在单卡训练，cuda 编号是0
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 训练输出目录（先全局建一次，避免保存时报错）
Path("checkpoint").mkdir(parents=True, exist_ok=True)
Path("logs/ResNet18").mkdir(parents=True, exist_ok=True)

#注意，在单卡训练，cuda 编号是0 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


transform = transforms.Compose([
    transforms.Resize((256, 256)), #把图像缩放为固定尺寸 256×256
    transforms.ToTensor(), #将图像像素值从[0-255]缩放为[0,1]，调整维度顺序为【c,h,w】
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))#将范围从 [0,1] 映射到 [-1,1]（即 (x-0.5)/0.5）
])

mean = 0  
std_dev = 0.1  
train_dataset = datasets.ImageFolder(root='data/UCMerced_LandUse-train', transform=transform)
test_dataset = datasets.ImageFolder(root='data/UCMerced_LandUse-test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True) #shuffle = True 训练集打乱样本顺序，有助于梯度稳定和泛化
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# The (real) AWGN channel   
# 模拟的AWGN信道 
def AWGN_channel(x, snr, P=2):
    batch_size, channels, height, width = x.shape  
    gamma = 10 ** (snr / 10.0)      
    noise = torch.sqrt(P / gamma) * torch.randn(batch_size, channels, height, width).to(device)  
    y = x + noise   
    return y


# Please set the symbol power if it is not a default value
#如果你的输入信号 x 并不是单位功率（即平均功率 ≠ 1），那么需要根据你实际信号的功率手动修改 P 参数，以便让信噪比计算正确。
# 模拟一个平坦瑞利衰落信道 + 复高斯加性噪声
def Fading_channel(x, snr, P = 2):
    gamma = 10 ** (snr / 10.0)
    [batch_size, feature_length] = x.shape
    K = feature_length//2
    
    h_I = torch.randn(batch_size, K).to(device)
    h_R = torch.randn(batch_size, K).to(device) 
    #生成随机的复数信号
    h_com = torch.complex(h_I, h_R) 
    #将原本的X调整为复数信号  
    x_com = torch.complex(x[:, 0:feature_length:2], x[:, 1:feature_length:2])
    # 实际模拟的衰落
    y_com = h_com*x_com
    
    #根据信噪比和 信号强度来算出噪声强度并产生随机噪声
    n_I = torch.sqrt(P/gamma)*torch.randn(batch_size, K).to(device)                                                                                          
    n_R = torch.sqrt(P/gamma)*torch.randn(batch_size, K).to(device)

    #将原本的实数信号调整为负数信号
    noise = torch.complex(n_I, n_R)
    #模型信号经过信道衰落后，叠加噪声后的信号
    y_add = y_com + noise

    y = y_add/h_com
    
    y_out = torch.zeros(batch_size, feature_length).to(device)
    y_out[:, 0:feature_length:2] = y.real
    y_out[:, 1:feature_length:2] = y.imag
    return y_out


#第一步先走瑞利衰落信道并做均衡（Fading_channel，输入是展平的一维特征向量）。

#第二步把一维结果还原回 4D 特征图 (B, C, H, W)。

#第三步为每个位置（按 B×C×H 粗粒度，沿 W 方向共享）随机生成一个 SNR(dB)。

#第四步在还原后的 4D 特征图上再叠加一次AWGN 噪声（AWGN_channel），SNR 采用上一步生成的张量，实现衰落 + 空间可变 SNR 的 AWGN组合信道。

def Combined_channel(x, snr, batch_size, channel, height, width):
    P=2
    x_faded = Fading_channel(x, snr, P)
    print ("x_faded.shape:",x_faded.shape)
    x_faded = x_faded.view((batch_size, channel, height, width))
    print ("x_faded.view.shape:",x_faded.shape)
    snr = torch.randint(0, 28, (x_faded.shape[0], x_faded.shape[1], x_faded.shape[2], 1)).to(device)
    x_combined = AWGN_channel(x_faded, snr, P)
    return x_combined

#根据 channel_type 把输入 z 送到对应的信道模型（AWGN / Fading / Combined）里处理，然后把结果返回。
def Channel(z, snr, channel_type, batch_size, channel, height, width):
    if channel_type == 'AWGN':
        z = AWGN_channel(z, snr)
    elif channel_type == 'Fading':
        z = Fading_channel(z, snr)
    elif channel_type == 'Combined_channel':
        z = Combined_channel(z, snr, batch_size, channel, height, width)
    return z

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
        # 输入期望是 (B, 512, H, W)，四个 3×3 卷积把通道从 512→256→128→64→32，保持空间尺寸不变（stride=1, padding=1）
        # 输出形状：(B, 32, H, W)
        x = self.encoder(x)
        print("encoder x.shape:",x.shape)
        #在编码空间加加性高斯噪声：mean 与 std_dev 分别是均值和标准差（前面定义了 mean=0, std_dev=0.1）
        noise = torch.randn_like(x) * std_dev + mean
        x = x + noise

        #记录当前 4D 形状，后面还原用
        batch_size, channel, height, width = x.shape
        #对 Fading / Combined_channel：
            #先展平到 (B, F)，其中 F = 32*H*W（后续 Fading 要把偶/奇位合成复符号）。
            #采样一个 每个样本一个 SNR(dB) 的列向量 (B,1)。
        if channel_type == 'Fading' or channel_type == 'Combined_channel':
            x = self.flatten(x)
            print("after flatten x.shape", x.shape)
            #为每个样本在【0，28】之间随机生成一个整数，作为该样本传输信道的信噪比
            SNR = torch.randint(0, 28, (x.shape[0], 1)).to(device)
        else :
            #对 AWGN：保持 4D，不展平。
            # 采样形状 (B,32,H,1) 的 SNR(dB)：每个样本、每个通道、每一行一个 SNR，沿 W 方向共享。
            SNR = torch.randint(0, 28, (x.shape[0], x.shape[1], x.shape[2], 1)).to(device)
        #传入信道
        x = Channel(x, SNR, channel_type, batch_size, channel, height, width)
        #把 2D（Fading）或 4D（AWGN/Combined）的输出，强制整形成 (B,32,H,W)，方便后面的解码
        print("after Channel x.shape:",x.shape)
        x = x.view((batch_size, channel, height, width))   
        #用四个反卷积把通道从 32→64→128→256→512 还原；最后 Sigmoid 把输出限制在 [0,1]，形状回到 (B, 512, H, W)。  
        x = self.decoder(x)
        return x


# cr压缩率（compression ratio）取值 0~1，表示“保留多少比例的最小权重通道”
# weights张量，形状 (B, C)一般是通道权重（来自 SE Block 或全连接层输出）
def mask_gen(weights, cr):
    position = round(cr*weights.size(1)) #weights.size(1) 是通道数 C，cr = 0.8 → 取 80% 的最小权重
    weights_sorted, index = torch.sort(weights, dim=1) #对每个样本（batch 维）在通道维上排序。
    mask = torch.zeros_like(weights) #先创建与 weights 同形状的 0 张量，用于存放最终掩码。

    for i in range(weights.size(0)):#weights.size(0) = B
        weight = weights_sorted[i, position-1] #找到当前样本第 position 小的权重值
        # print(weight)
        for j in range(weights.size(1)):#若该通道权重 ≤ 阈值，则 mask=1；否则mask = 0
            if weights[i, j] <= weight:
                mask[i, j] = 1
    return mask  
  

class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(SE_Block, self).__init__()
        #自适应平均池化到 1×1，把每个通道的空间维 (H,W) 压缩成一个数（通道的全局平均）
        #输入 x 形状 (B, C, H, W)，gap(x) 输出 (B, C, 1, 1)。
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        #两层全连接瓶颈结构：C → C/ratio → C，最后接 Sigmoid 把权重压到 [0,1]。
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )

    def forward(self, x, cr=0.8):
        b, c, h, w = x.size()
        y = self.gap(x).view(b, c)
        print("特征选择-平均池化:", y.shape)
        y = self.fc(y)
        print("特征选择-全连接层:", y.shape)
        mask = mask_gen(y, cr).view(b,c,1,1)
        print("特征选择-掩码生成，掩码形状：", mask.shape)
        print("源数据形状:", x.shape)
        return x * mask


class SatelliteClassifierWithAttention(nn.Module):
    def __init__(self, num_classes):
        #初始化
        super(SatelliteClassifierWithAttention, self).__init__()
        #加载一个 ResNet-18 主干网络
        self.resnet18 = resnet18(weights=ResNet18_Weights.DEFAULT)#pretrained=True会自动下载或加载 ImageNet 上训练好的参数
        # ✅ 替换原resnet18第一层为7*7这里改第一层卷积为 3×3
        self.resnet18.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=2,   # 下采样保持不变
            padding=1,
            bias=False
        )
        in_features = self.resnet18.fc.in_features #获取 ResNet18 最后全连接层的输入特征维度。对 ResNet18，这个值是 512。
        self.attention_module = SE_Block(in_features) #在 ResNet 的特征输出后增加一个通道注意力模块
        self.antoencoder = Autoencoder() #模拟信道传输与恢复过程，连续卷积将特征从 (B,512,H,W) 压缩到 (B,32,H,W)。
        self.resnet18.fc = nn.Linear(in_features, num_classes)#把原来的 ResNet-18 最后一层（输出 1000 类）替换成任务所需的分类头，比如使用使用UCMerced_LandUse，那就是21类

    def forward(self, x, cr, channel_type):
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)

        x = self.resnet18.layer1(x)
        x = self.resnet18.layer2(x)
        x = self.resnet18.layer3(x)
        x = self.resnet18.layer4(x)

        #到这里是ResNet18 前半段，输入通常是 (B,3,H,W) 的图像。到 layer4 结束，得到语义特征 (B,512,H',W')（H', W' 是下采样后的尺寸，默认下采样32倍）
        print("before x.shape:",x.shape)#（B,512,H/32,H/32）
        # print("before x:",x)
        #加入自注意力
        x = self.attention_module(x, cr)#SE+二值掩码
        print("after attention_module x.shape:",x.shape)#（B,512,H/32,H/32）->（B,512,1,1）->(B,512/CR,1,1)->(B,512,1,1)
        #加入信道编码并传入信道并解码
        x = self.antoencoder(x, channel_type) #(B,410,1,1)->(B,32,1,1)->加噪->传入信道->解码->(B,512,1,1)->（B,512,H/32,H/32）
        print("after antoencoder x.shape:",x.shape)
        x = self.resnet18.avgpool(x) ## (B,512,1,1)
        # in_features_fc = x.size(1)
        # self.resnet18.fc = nn.Linear(in_features_fc, num_classes)
        x = x.view(x.size(0), -1) ## (B,512)
        x = self.resnet18.fc(x)  # # (B,num_classes)】

        return x

def continue_train(cr, num_epochs, pre_checkpoint, channel_type):
    start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    num_classes = len(train_dataset.classes)  
    model = SatelliteClassifierWithAttention(num_classes)
    model = model.to(device)

    pretrained_dict = torch.load(f'{pre_checkpoint}')
    model_dict = model.state_dict()

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter()

    # num_epochs = 50 

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images, cr, channel_type)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')
        avg_train_loss = running_loss / len(train_loader)
        scheduler.step(avg_train_loss)
        writer.add_scalar('Training Loss', running_loss/len(train_loader), epoch + 1)

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images, cr, channel_type)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f'Test Accuracy: {accuracy}')

        writer.add_scalar('Test Accuracy', accuracy)
# Save the model with the specified cr and num_epochs in the file name
    save_path = f'checkpoint/classifier_attention_auto_UCMerced_LandUse_{channel_type}_ResNet18_20epoch_0.8_up_{num_epochs}epoch_{cr}.pth'
    torch.save(model.state_dict(), save_path)

    writer.close()

    current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    # Write results to a txt file
    with open(f'logs/ResNet18/classifier_attention_auto_UCMerced_LandUse_{channel_type}_ResNet18_60_up_{num_epochs}epoch_{cr}.txt', 'w') as file:
        file.write('strat comtinue training...\n')
        file.write(f'Time: {start_time}----------{current_time}\n')
        file.write(f'model name:{save_path}\n')
        file.write(f'channel_type:{channel_type}\n')
        file.write(f'CR (Compression Ratio): {cr}\n')
        file.write(f'Num Epochs: {num_epochs}\n')
        file.write(f'Test Accuracy: {accuracy}\n')
        file.write('train over!\n')

def train(cr, num_epochs, channel_type):#传入三个参数：cr-压缩比；num_epochs-训练周期；channel_type-信道类型
    start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    num_classes = len(train_dataset.classes)  #num_classes类别数
    model = SatelliteClassifierWithAttention(num_classes) 
    model = model.to(device) #
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001) #Adam，初始学习率 1e-3。
    #学习率调度器，传入的是训练集平均损失（见后文），当它“无改进”持续 patience=5 个 epoch，就把 LR 乘以 0.1。
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    writer = SummaryWriter() #用于 TensorBoard 记录训练 loss / 测试 acc
    # num_epochs = 20

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            print("epoch:",epoch)
            print("image.shape",images.shape)
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad() # 清理梯度
            outputs = model(images, cr, channel_type) #前向
            loss = criterion(outputs, labels) #损失
            loss.backward() #反向传播
            optimizer.step() #更新参数

            running_loss += loss.item() # 累积当前 batch 的损失

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')
 
        avg_train_loss = running_loss / len(train_loader)

        scheduler.step(avg_train_loss)
        # 会在 epoch 结束时除以 batch 数，得到平均训练损失
        writer.add_scalar('Training Loss', running_loss/len(train_loader), epoch + 1)

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device) #
                outputs = model(images, cr, channel_type)#output = (B,21)
                _, predicted = torch.max(outputs, 1) #在当前批次中每一个样本都找到预测的类型
                total += labels.size(0) #labels.size(0)当前batch的样本数量，进行累加
                correct += (predicted == labels).sum().item() #统计预测正确的样本数

        accuracy = correct / total
        print(f'Test Accuracy: {accuracy}')
        writer.add_scalar('Test Accuracy', accuracy)

    # Save the model with the specified cr and num_epochs in the file name
    save_path = f'checkpoint/classifier_attention_auto_UCMerced_LandUse_Combined_channel_ResNet18_{num_epochs}epoch_{cr}.pth'
    torch.save(model.state_dict(), save_path)

    
    writer.close()
    current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    # Write results to a txt file
    with open(f'logs/ResNet18/classifier_attention_auto_UCMerced_LandUse_Combined_channel_ResNet18_{num_epochs}epoch_{cr}.txt', 'w') as file:
        file.write('strat training...\n')
        file.write(f'Time: {start_time}----------{current_time}\n')
        file.write(f'model name:{save_path}\n')
        file.write(f'model name:{channel_type}\n')
        file.write(f'CR (Compression Ratio): {cr}\n')
        file.write(f'Num Epochs: {num_epochs}\n')
        file.write(f'Test Accuracy: {accuracy}\n')
        file.write('train over!\n')

def main(task,cr,num_epochs,pre_checkpoint,channel_type):
    if task == 'continue':
        print("continue_train start!")
        continue_train(cr, num_epochs,pre_checkpoint,channel_type)
        print("continue_train over!")
    else :
        print("train start!")
        train(cr, num_epochs,channel_type)
        print("train over!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or continue training a model.")
    parser.add_argument('--task', choices=['continue', 'train'], default='train', required=True, help='Specify the task (continue or train).')
    parser.add_argument('--cr', type=float, default=0.1, help='Specify the compression ratio (cr) for the SE Block.')
    parser.add_argument('--num_epochs', type=int, default=60, help='Specify the number of epochs for training.')
    parser.add_argument('--pre_checkpoint', type=str, default=None, help='Specify the pretrained checkpoint for continue train.')
    parser.add_argument('--channel_type', choices=['AWGN', 'Fading',"Combined_channel"], default='Combined_channel', help='Specify the channel_type for transfer.')
    args = parser.parse_args()
    main(args.task, args.cr, args.num_epochs,args.pre_checkpoint,args.channel_type)

