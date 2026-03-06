import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18,ResNet18_Weights  # 浠esNet18涓轰緥锛屼篃鍙互鏍规嵁瀹為檯鎯呭喌閫夋嫨鍏朵粬妯″瀷
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
import math
import os
from typing import Dict

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

# 娉ㄦ剰锛屽湪鍗曞崱璁粌锛宑uda 缂栧彿鏄?
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 璁粌杈撳嚭鐩綍锛堝厛鍏ㄥ眬寤轰竴娆★紝閬垮厤淇濆瓨鏃舵姤閿欙級
# Path("AID_150_combine_0.8/checkpoint").mkdir(parents=True, exist_ok=True)
# Path("AID_150_combine_0.8/logs/ResNet18").mkdir(parents=True, exist_ok=True)

def get_exp_dirs(
    num_epochs,
    channel_type,
    cr,
    dataset_name="Soya",
    output_dir="runs/original_train",
    run_name=None,
):
    channel_tag = {
        "AWGN": "awgn",
        "Fading": "fading",
        "Combined_channel": "combine",
    }[channel_type]

    dataset_tag = dataset_name.replace(" ", "_")
    output_root = resolve_path(output_dir)
    if run_name:
        exp_name = run_name
    else:
        exp_name = f"{dataset_tag}_{num_epochs}_{channel_tag}_{cr}_{time.strftime('%Y%m%d_%H%M%S')}"

    base_dir = output_root / exp_name
    ckpt_dir = base_dir / "checkpoint"
    log_dir = base_dir / "tensorboard"

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    return base_dir, ckpt_dir, log_dir


def write_run_summary_md(summary_path: Path, payload: Dict[str, object]) -> None:
    lines = [
        "# Original Model Training Summary",
        "",
        "## Run",
        f"- task: `{payload['task']}`",
        f"- run_dir: `{payload['run_dir']}`",
        f"- start_time: `{payload['start_time']}`",
        f"- end_time: `{payload['end_time']}`",
        "",
        "## Data & Config",
        f"- dataset_name: `{payload['dataset_name']}`",
        f"- train_dir: `{payload['train_dir']}`",
        f"- valid_dir: `{payload['valid_dir']}`",
        f"- num_classes: `{payload['num_classes']}`",
        f"- channel_type: `{payload['channel_type']}`",
        f"- cr: `{payload['cr']}`",
        f"- num_epochs: `{payload['num_epochs']}`",
        f"- batch_size: `{payload['batch_size']}`",
        f"- num_workers: `{payload['num_workers']}`",
        f"- device: `{payload['device']}`",
        "",
        "## Outputs",
        f"- best_checkpoint: `{payload['best_checkpoint']}`",
        f"- final_checkpoint: `{payload['final_checkpoint']}`",
        f"- tensorboard_dir: `{payload['tensorboard_dir']}`",
        f"- txt_log: `{payload['txt_log']}`",
        "",
        "## Metrics",
        f"- final_valid_accuracy: `{payload['final_valid_accuracy']:.6f}`",
        f"- final_valid_loss: `{payload['final_valid_loss']:.6f}`",
        f"- best_valid_loss: `{payload['best_valid_loss']:.6f}`",
        f"- best_epoch: `{payload['best_epoch']}`",
        "",
    ]
    summary_path.write_text("\n".join(lines), encoding="utf-8")

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
])
# 璁粌闆嗙敤 train_transform锛?

# 楠岃瘉闆嗕粛鐢ㄦ棤澧炲己鐨?transform锛堟棤澧炲己锛?
valid_transform = transforms.Compose([
    transforms.Resize((256, 256)),   # 鎶婂師濮嬪浘鐗囩缉鏀惧埌 256脳256 灏哄
    transforms.ToTensor(),           # 杞垚寮犻噺锛屽苟鎶婂儚绱犲€间粠 [0,255] 鈫?[0,1]
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406), # 瀵瑰簲 ImageNet 鏁版嵁闆嗙殑閫氶亾鍧囧€?
        std=(0.229, 0.224, 0.225)   # 瀵瑰簲 ImageNet 鏁版嵁闆嗙殑閫氶亾鏍囧噯宸?
    ),
])


mean = 0  
std_dev = 0.1 

def build_dataloaders(train_dir: Path, valid_dir: Path, batch_size: int = 64, num_workers: int = 0):
    train_dataset = datasets.ImageFolder(root=str(train_dir), transform=train_transform)
    valid_dataset = datasets.ImageFolder(root=str(valid_dir), transform=valid_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, valid_loader, train_dataset


# The (real) AWGN channel   
# 妯℃嫙鐨凙WGN淇￠亾 
def AWGN_channel(x, snr, P=2):
    batch_size, channels, height, width = x.shape  
    gamma = 10 ** (snr / 10.0)      
    noise = torch.sqrt(P / gamma) * torch.randn(batch_size, channels, height, width).to(device)  
    y = x + noise   
    return y


# Please set the symbol power if it is not a default value
#濡傛灉浣犵殑杈撳叆淇″彿 x 骞朵笉鏄崟浣嶅姛鐜囷紙鍗冲钩鍧囧姛鐜?鈮?1锛夛紝閭ｄ箞闇€瑕佹牴鎹綘瀹為檯淇″彿鐨勫姛鐜囨墜鍔ㄤ慨鏀?P 鍙傛暟锛屼互渚胯淇″櫔姣旇绠楁纭€?
# 妯℃嫙涓€涓钩鍧︾憺鍒╄“钀戒俊閬?+ 澶嶉珮鏂姞鎬у櫔澹?
def Fading_channel(x, snr, P = 2):
    gamma = 10 ** (snr / 10.0)
    [batch_size, feature_length] = x.shape
    K = feature_length//2
    
    h_I = torch.randn(batch_size, K).to(device)
    h_R = torch.randn(batch_size, K).to(device) 
    #鐢熸垚闅忔満鐨勫鏁颁俊鍙?
    h_com = torch.complex(h_I, h_R) 
    #灏嗗師鏈殑X璋冩暣涓哄鏁颁俊鍙? 
    x_com = torch.complex(x[:, 0:feature_length:2], x[:, 1:feature_length:2])
    # 瀹為檯妯℃嫙鐨勮“钀?
    y_com = h_com*x_com
    
    #鏍规嵁淇″櫔姣斿拰 淇″彿寮哄害鏉ョ畻鍑哄櫔澹板己搴﹀苟浜х敓闅忔満鍣０
    n_I = torch.sqrt(P/gamma)*torch.randn(batch_size, K).to(device)                                                                                          
    n_R = torch.sqrt(P/gamma)*torch.randn(batch_size, K).to(device)

    #灏嗗師鏈殑瀹炴暟淇″彿璋冩暣涓鸿礋鏁颁俊鍙?
    noise = torch.complex(n_I, n_R)
    #妯″瀷淇″彿缁忚繃淇￠亾琛拌惤鍚庯紝鍙犲姞鍣０鍚庣殑淇″彿
    y_add = y_com + noise

    y = y_add/h_com
    
    y_out = torch.zeros(batch_size, feature_length).to(device)
    y_out[:, 0:feature_length:2] = y.real
    y_out[:, 1:feature_length:2] = y.imag
    return y_out


#绗竴姝ュ厛璧扮憺鍒╄“钀戒俊閬撳苟鍋氬潎琛★紙Fading_channel锛岃緭鍏ユ槸灞曞钩鐨勪竴缁寸壒寰佸悜閲忥級銆?

#绗簩姝ユ妸涓€缁寸粨鏋滆繕鍘熷洖 4D 鐗瑰緛鍥?(B, C, H, W)銆?

#绗笁姝ヤ负姣忎釜浣嶇疆锛堟寜 B脳C脳H 绮楃矑搴︼紝娌?W 鏂瑰悜鍏变韩锛夐殢鏈虹敓鎴愪竴涓?SNR(dB)銆?

#绗洓姝ュ湪杩樺師鍚庣殑 4D 鐗瑰緛鍥句笂鍐嶅彔鍔犱竴娆WGN 鍣０锛圓WGN_channel锛夛紝SNR 閲囩敤涓婁竴姝ョ敓鎴愮殑寮犻噺锛屽疄鐜拌“钀?+ 绌洪棿鍙彉 SNR 鐨?AWGN缁勫悎淇￠亾銆?

def Combined_channel(x, snr, batch_size, channel, height, width):
    P=2
    x_faded = Fading_channel(x, snr, P)
    # print ("x_faded.shape:",x_faded.shape)
    x_faded = x_faded.view((batch_size, channel, height, width))
    # print ("x_faded.view.shape:",x_faded.shape)
    snr = torch.randint(0, 28, (x_faded.shape[0], x_faded.shape[1], x_faded.shape[2], 1)).to(device)
    x_combined = AWGN_channel(x_faded, snr, P)
    return x_combined

#鏍规嵁 channel_type 鎶婅緭鍏?z 閫佸埌瀵瑰簲鐨勪俊閬撴ā鍨嬶紙AWGN / Fading / Combined锛夐噷澶勭悊锛岀劧鍚庢妸缁撴灉杩斿洖銆?
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
        # 杈撳叆鏈熸湜鏄?(B, 512, H, W)锛屽洓涓?3脳3 鍗风Н鎶婇€氶亾浠?512鈫?56鈫?28鈫?4鈫?2锛屼繚鎸佺┖闂村昂瀵镐笉鍙橈紙stride=1, padding=1锛?
        # 杈撳嚭褰㈢姸锛?B, 32, H, W)
        x = self.encoder(x)
        # print("encoder x.shape:",x.shape)
        #鍦ㄧ紪鐮佺┖闂村姞鍔犳€ч珮鏂櫔澹帮細mean 涓?std_dev 鍒嗗埆鏄潎鍊煎拰鏍囧噯宸紙鍓嶉潰瀹氫箟浜?mean=0, std_dev=0.1锛?
        noise = torch.randn_like(x) * std_dev + mean
        x = x + noise

        #璁板綍褰撳墠 4D 褰㈢姸锛屽悗闈㈣繕鍘熺敤
        batch_size, channel, height, width = x.shape
        #瀵?Fading / Combined_channel锛?
            #鍏堝睍骞冲埌 (B, F)锛屽叾涓?F = 32*H*W锛堝悗缁?Fading 瑕佹妸鍋?濂囦綅鍚堟垚澶嶇鍙凤級銆?
            #閲囨牱涓€涓?姣忎釜鏍锋湰涓€涓?SNR(dB) 鐨勫垪鍚戦噺 (B,1)銆?
        if channel_type == 'Fading' or channel_type == 'Combined_channel':
            x = self.flatten(x)
            # print("after flatten x.shape", x.shape)
            #涓烘瘡涓牱鏈湪銆?锛?8銆戜箣闂撮殢鏈虹敓鎴愪竴涓暣鏁帮紝浣滀负璇ユ牱鏈紶杈撲俊閬撶殑淇″櫔姣?
            SNR = torch.randint(0, 28, (x.shape[0], 1)).to(device)
        else :
            #瀵?AWGN锛氫繚鎸?4D锛屼笉灞曞钩銆?
            # 閲囨牱褰㈢姸 (B,32,H,1) 鐨?SNR(dB)锛氭瘡涓牱鏈€佹瘡涓€氶亾銆佹瘡涓€琛屼竴涓?SNR锛屾部 W 鏂瑰悜鍏变韩銆?
            SNR = torch.randint(0, 28, (x.shape[0], x.shape[1], x.shape[2], 1)).to(device)
        #浼犲叆淇￠亾
        x = Channel(x, SNR, channel_type, batch_size, channel, height, width)
        #鎶?2D锛團ading锛夋垨 4D锛圓WGN/Combined锛夌殑杈撳嚭锛屽己鍒舵暣褰㈡垚 (B,32,H,W)锛屾柟渚垮悗闈㈢殑瑙ｇ爜
        # print("after Channel x.shape:",x.shape)
        x = x.view((batch_size, channel, height, width))   
        #鐢ㄥ洓涓弽鍗风Н鎶婇€氶亾浠?32鈫?4鈫?28鈫?56鈫?12 杩樺師锛涙渶鍚?Sigmoid 鎶婅緭鍑洪檺鍒跺湪 [0,1]锛屽舰鐘跺洖鍒?(B, 512, H, W)銆? 
        x = self.decoder(x)
        return x


# cr鍘嬬缉鐜囷紙compression ratio锛夊彇鍊?0~1锛岃〃绀衡€滀繚鐣欏灏戞瘮渚嬬殑鏈€灏忔潈閲嶉€氶亾鈥?
# weights寮犻噺锛屽舰鐘?(B, C)涓€鑸槸閫氶亾鏉冮噸锛堟潵鑷?SE Block 鎴栧叏杩炴帴灞傝緭鍑猴級
def mask_gen(weights, cr):
    position = round(cr*weights.size(1)) #weights.size(1) 鏄€氶亾鏁?C锛宑r = 0.8 鈫?鍙?80% 鐨勬渶灏忔潈閲?
    weights_sorted, index = torch.sort(weights, dim=1) #瀵规瘡涓牱鏈紙batch 缁达級鍦ㄩ€氶亾缁翠笂鎺掑簭銆?
    mask = torch.zeros_like(weights) #鍏堝垱寤轰笌 weights 鍚屽舰鐘剁殑 0 寮犻噺锛岀敤浜庡瓨鏀炬渶缁堟帺鐮併€?

    for i in range(weights.size(0)):#weights.size(0) = B
        weight = weights_sorted[i, position-1] #鎵惧埌褰撳墠鏍锋湰绗?position 灏忕殑鏉冮噸鍊?
        # print(weight)
        for j in range(weights.size(1)):#鑻ヨ閫氶亾鏉冮噸 鈮?闃堝€硷紝鍒?mask=1锛涘惁鍒檓ask = 0
            if weights[i, j] <= weight:
                mask[i, j] = 1
    return mask  
  

class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(SE_Block, self).__init__()
        #鑷€傚簲骞冲潎姹犲寲鍒?1脳1锛屾妸姣忎釜閫氶亾鐨勭┖闂寸淮 (H,W) 鍘嬬缉鎴愪竴涓暟锛堥€氶亾鐨勫叏灞€骞冲潎锛?
        #杈撳叆 x 褰㈢姸 (B, C, H, W)锛実ap(x) 杈撳嚭 (B, C, 1, 1)銆?
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        #涓ゅ眰鍏ㄨ繛鎺ョ摱棰堢粨鏋勶細C 鈫?C/ratio 鈫?C锛屾渶鍚庢帴 Sigmoid 鎶婃潈閲嶅帇鍒?[0,1]銆?
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),  # 浠?c -> c/r
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),  # 浠?c/r -> c
            nn.Sigmoid()
        )

    def forward(self, x, cr=0.8):
        b, c, h, w = x.size()
        y = self.gap(x).view(b, c)
        # print("鐗瑰緛閫夋嫨-骞冲潎姹犲寲:", y.shape)
        y = self.fc(y)
        # print("鐗瑰緛閫夋嫨-鍏ㄨ繛鎺ュ眰:", y.shape)
        mask = mask_gen(y, cr).view(b,c,1,1)
        # print("鐗瑰緛閫夋嫨-鎺╃爜鐢熸垚锛屾帺鐮佸舰鐘讹細", mask.shape)
        # print("婧愭暟鎹舰鐘?", x.shape)
        return x * mask


class SatelliteClassifierWithAttention(nn.Module):
    def __init__(self, num_classes):
        #鍒濆鍖?
        super(SatelliteClassifierWithAttention, self).__init__()
        #鍔犺浇涓€涓?ResNet-18 涓诲共缃戠粶
        self.resnet18 = resnet18(weights=ResNet18_Weights.DEFAULT)#pretrained=True浼氳嚜鍔ㄤ笅杞芥垨鍔犺浇 ImageNet 涓婅缁冨ソ鐨勫弬鏁?
        # # 鉁?鏇挎崲鍘焤esnet18绗竴灞備负7*7杩欓噷鏀圭涓€灞傚嵎绉负 3脳3
        # self.resnet18.conv1 = nn.Conv2d(
        #     in_channels=3,
        #     out_channels=64,
        #     kernel_size=3,
        #     stride=2,   # 涓嬮噰鏍蜂繚鎸佷笉鍙?
        #     padding=1,
        #     bias=False
        # )
        in_features = self.resnet18.fc.in_features #鑾峰彇 ResNet18 鏈€鍚庡叏杩炴帴灞傜殑杈撳叆鐗瑰緛缁村害銆傚 ResNet18锛岃繖涓€兼槸 512銆?
        self.attention_module = SE_Block(in_features) #鍦?ResNet 鐨勭壒寰佽緭鍑哄悗澧炲姞涓€涓€氶亾娉ㄦ剰鍔涙ā鍧?
        self.antoencoder = Autoencoder() #妯℃嫙淇￠亾浼犺緭涓庢仮澶嶈繃绋嬶紝杩炵画鍗风Н灏嗙壒寰佷粠 (B,512,H,W) 鍘嬬缉鍒?(B,32,H,W)銆?
        self.resnet18.fc = nn.Linear(in_features, num_classes)#鎶婂師鏉ョ殑 ResNet-18 鏈€鍚庝竴灞傦紙杈撳嚭 1000 绫伙級鏇挎崲鎴愪换鍔℃墍闇€鐨勫垎绫诲ご锛屾瘮濡備娇鐢ㄤ娇鐢║CMerced_LandUse锛岄偅灏辨槸21绫?

    def forward(self, x, cr, channel_type):
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)

        x = self.resnet18.layer1(x)
        x = self.resnet18.layer2(x)
        x = self.resnet18.layer3(x)
        x = self.resnet18.layer4(x)

        #鍒拌繖閲屾槸ResNet18 鍓嶅崐娈碉紝杈撳叆閫氬父鏄?(B,3,H,W) 鐨勫浘鍍忋€傚埌 layer4 缁撴潫锛屽緱鍒拌涔夌壒寰?(B,512,H',W')锛圚', W' 鏄笅閲囨牱鍚庣殑灏哄锛岄粯璁や笅閲囨牱32鍊嶏級
        # print("before x.shape:",x.shape)#锛圔,512,H/32,H/32锛?
        # print("before x:",x)
        #鍔犲叆鑷敞鎰忓姏
        x = self.attention_module(x, cr)#SE+浜屽€兼帺鐮?
        # print("after attention_module x.shape:",x.shape)#锛圔,512,H/32,H/32锛?>锛圔,512,1,1锛?>(B,512/CR,1,1)->(B,512,1,1)
        #鍔犲叆淇￠亾缂栫爜骞朵紶鍏ヤ俊閬撳苟瑙ｇ爜
        x = self.antoencoder(x, channel_type) #(B,410,1,1)->(B,32,1,1)->鍔犲櫔->浼犲叆淇￠亾->瑙ｇ爜->(B,512,1,1)->锛圔,512,H/32,H/32锛?
        # print("after antoencoder x.shape:",x.shape)
        x = self.resnet18.avgpool(x) ## (B,512,1,1)
        # in_features_fc = x.size(1)
        # self.resnet18.fc = nn.Linear(in_features_fc, num_classes)
        x = x.view(x.size(0), -1) ## (B,512)
        x = self.resnet18.fc(x)  # # (B,num_classes)銆?

        return x

def continue_train(
    cr,
    num_epochs,
    pre_checkpoint,
    channel_type,
    train_loader,
    valid_loader,
    num_classes,
    dataset_name,
    train_dir,
    valid_dir,
    batch_size,
    num_workers,
    output_dir,
    run_name=None,
):
    dataset_tag = dataset_name.replace(" ", "_")
    run_dir, ckpt_dir, log_dir = get_exp_dirs(
        num_epochs,
        channel_type,
        cr,
        dataset_name=dataset_tag,
        output_dir=output_dir,
        run_name=run_name,
    )

    start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    model = SatelliteClassifierWithAttention(num_classes).to(device)

    if pre_checkpoint is None:
        raise ValueError("--pre_checkpoint is required when --task continue is used.")

    pre_checkpoint_path = resolve_path(pre_checkpoint)
    pretrained_dict = torch.load(str(pre_checkpoint_path), map_location=device)
    model_dict = model.state_dict()

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    try:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    except TypeError:
        # Compatible with torch versions whose ReduceLROnPlateau has no `verbose` arg.
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter(str(log_dir))
    best_valid_loss = float('inf')
    best_epoch = 0
    accuracy = 0.0
    avg_valid_loss = 0.0
    best_model_path = ckpt_dir / f'best_classifier_attention_auto_{dataset_tag}_{channel_type}_ResNet18_up_{num_epochs}epoch_{cr}.pth'

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

        avg_train_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_train_loss}')
        writer.add_scalar('Training Loss', avg_train_loss, epoch + 1)

        model.eval()
        correct = 0
        total = 0
        valid_loss_sum = 0.0

        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images, cr, channel_type)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss_val = criterion(outputs, labels)
                valid_loss_sum += loss_val.item()

        accuracy = correct / total
        avg_valid_loss = valid_loss_sum / len(valid_loader)
        scheduler.step(avg_valid_loss)

        print(f'Valid Accuracy: {accuracy}')
        writer.add_scalar('Valid Loss', avg_valid_loss, epoch + 1)
        writer.add_scalar('Valid Accuracy', accuracy, epoch + 1)
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_model_path)

    save_path = ckpt_dir / f'classifier_attention_auto_{dataset_tag}_{channel_type}_ResNet18_up_{num_epochs}epoch_{cr}.pth'
    torch.save(model.state_dict(), save_path)
    log_path = log_dir / f'classifier_attention_auto_{dataset_tag}_{channel_type}_ResNet18_up_{num_epochs}epoch_{cr}.txt'
    writer.close()
    current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

    with open(log_path, 'w') as file:
        file.write('strat continue training...\n')
        file.write(f'Time: {start_time}----------{current_time}\n')
        file.write(f'model name:{save_path}\n')
        file.write(f'dataset:{dataset_name}\n')
        file.write(f'channel_type:{channel_type}\n')
        file.write(f'CR (Compression Ratio): {cr}\n')
        file.write(f'Num Epochs: {num_epochs}\n')
        file.write(f'Valid Accuracy: {accuracy}\n')
        file.write('train over!\n')
    summary_md = run_dir / "run_summary.md"
    write_run_summary_md(
        summary_md,
        {
            "task": "continue",
            "run_dir": str(run_dir),
            "start_time": start_time,
            "end_time": current_time,
            "dataset_name": dataset_name,
            "train_dir": str(train_dir),
            "valid_dir": str(valid_dir),
            "num_classes": num_classes,
            "channel_type": channel_type,
            "cr": cr,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "device": str(device),
            "best_checkpoint": str(best_model_path),
            "final_checkpoint": str(save_path),
            "tensorboard_dir": str(log_dir),
            "txt_log": str(log_path),
            "final_valid_accuracy": float(accuracy),
            "final_valid_loss": float(avg_valid_loss),
            "best_valid_loss": float(best_valid_loss if best_valid_loss < float('inf') else avg_valid_loss),
            "best_epoch": int(best_epoch),
        },
    )

    return {
        "run_dir": run_dir,
        "summary_md": summary_md,
        "best_checkpoint": best_model_path,
        "final_checkpoint": save_path,
    }
def train(
    cr,
    num_epochs,
    channel_type,
    train_loader,
    valid_loader,
    num_classes,
    dataset_name,
    train_dir,
    valid_dir,
    batch_size,
    num_workers,
    output_dir,
    run_name=None,
):
    dataset_tag = dataset_name.replace(" ", "_")
    run_dir, ckpt_dir, log_dir = get_exp_dirs(
        num_epochs,
        channel_type,
        cr,
        dataset_name=dataset_tag,
        output_dir=output_dir,
        run_name=run_name,
    )
    start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

    model = SatelliteClassifierWithAttention(num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    try:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    except TypeError:
        # Compatible with torch versions whose ReduceLROnPlateau has no `verbose` arg.
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    writer = SummaryWriter(str(log_dir))
    best_valid_loss = float('inf')
    best_epoch = 0
    accuracy = 0.0
    avg_valid_loss = 0.0
    best_model_path = ckpt_dir / f'best_classifier_attention_auto_{dataset_tag}_{channel_type}_ResNet18_{num_epochs}epoch_{cr}.pth'

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

        avg_train_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_train_loss}')
        writer.add_scalar('Training Loss', avg_train_loss, epoch + 1)

        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning Rate', current_lr, epoch + 1)

        model.eval()
        correct = 0
        total = 0
        valid_loss_sum = 0.0

        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images, cr, channel_type)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss_val = criterion(outputs, labels)
                valid_loss_sum += loss_val.item()

        accuracy = correct / total
        avg_valid_loss = valid_loss_sum / len(valid_loader)
        writer.add_scalar('Valid Loss', avg_valid_loss, epoch + 1)

        scheduler.step(avg_valid_loss)

        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_model_path)
            print(f'>>> New best model saved: {best_model_path} (valid_loss={best_valid_loss:.6f})')

        print(f'valid Accuracy: {accuracy}')
        writer.add_scalar('valid Accuracy', accuracy, epoch + 1)

    save_path = ckpt_dir / f'classifier_attention_auto_{dataset_tag}_{channel_type}_ResNet18_{num_epochs}epoch_{cr}.pth'
    torch.save(model.state_dict(), save_path)

    writer.close()
    current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

    log_path = log_dir / f'classifier_attention_auto_{dataset_tag}_{channel_type}_ResNet18_{num_epochs}epoch_{cr}.txt'
    with open(log_path, 'w') as file:
        file.write('strat training...\n')
        file.write(f'Time: {start_time}----------{current_time}\n')
        file.write(f'model name:{save_path}\n')
        file.write(f'dataset:{dataset_name}\n')
        file.write(f'channel name:{channel_type}\n')
        file.write(f'CR (Compression Ratio): {cr}\n')
        file.write(f'Num Epochs: {num_epochs}\n')
        file.write(f'Valid Accuracy: {accuracy}\n')
        file.write('train over!\n')
    summary_md = run_dir / "run_summary.md"
    write_run_summary_md(
        summary_md,
        {
            "task": "train",
            "run_dir": str(run_dir),
            "start_time": start_time,
            "end_time": current_time,
            "dataset_name": dataset_name,
            "train_dir": str(train_dir),
            "valid_dir": str(valid_dir),
            "num_classes": num_classes,
            "channel_type": channel_type,
            "cr": cr,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "device": str(device),
            "best_checkpoint": str(best_model_path),
            "final_checkpoint": str(save_path),
            "tensorboard_dir": str(log_dir),
            "txt_log": str(log_path),
            "final_valid_accuracy": float(accuracy),
            "final_valid_loss": float(avg_valid_loss),
            "best_valid_loss": float(best_valid_loss if best_valid_loss < float('inf') else avg_valid_loss),
            "best_epoch": int(best_epoch),
        },
    )

    return {
        "run_dir": run_dir,
        "summary_md": summary_md,
        "best_checkpoint": best_model_path,
        "final_checkpoint": save_path,
    }
def main(
    task,
    cr,
    num_epochs,
    pre_checkpoint,
    channel_type,
    dataset_name,
    train_dir,
    valid_dir,
    batch_size,
    num_workers,
    output_dir,
    run_name,
):
    train_dir_path = resolve_path(train_dir)
    valid_dir_path = resolve_path(valid_dir)

    if not train_dir_path.exists():
        raise FileNotFoundError(f"Train directory does not exist: {train_dir_path}")
    if not valid_dir_path.exists():
        raise FileNotFoundError(f"Valid directory does not exist: {valid_dir_path}")

    train_loader, valid_loader, train_dataset = build_dataloaders(
        train_dir=train_dir_path,
        valid_dir=valid_dir_path,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    num_classes = len(train_dataset.classes)

    print(f"Dataset: {dataset_name}")
    print(f"Train dir: {train_dir_path}")
    print(f"Valid dir: {valid_dir_path}")
    print(f"Num classes: {num_classes}")
    print(f"Output root: {resolve_path(output_dir)}")

    if task == 'continue':
        print("continue_train start!")
        result = continue_train(
            cr=cr,
            num_epochs=num_epochs,
            pre_checkpoint=pre_checkpoint,
            channel_type=channel_type,
            train_loader=train_loader,
            valid_loader=valid_loader,
            num_classes=num_classes,
            dataset_name=dataset_name,
            train_dir=train_dir_path,
            valid_dir=valid_dir_path,
            batch_size=batch_size,
            num_workers=num_workers,
            output_dir=output_dir,
            run_name=run_name,
        )
        print("continue_train over!")
    else:
        print("train start!")
        result = train(
            cr=cr,
            num_epochs=num_epochs,
            channel_type=channel_type,
            train_loader=train_loader,
            valid_loader=valid_loader,
            num_classes=num_classes,
            dataset_name=dataset_name,
            train_dir=train_dir_path,
            valid_dir=valid_dir_path,
            batch_size=batch_size,
            num_workers=num_workers,
            output_dir=output_dir,
            run_name=run_name,
        )
        print("train over!")

    print(f"Run dir: {result['run_dir']}")
    print(f"Summary md: {result['summary_md']}")
    print(f"Best checkpoint: {result['best_checkpoint']}")
    print(f"Final checkpoint: {result['final_checkpoint']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or continue training a model.")
    parser.add_argument('--task', choices=['continue', 'train'], default='train', required=True, help='Specify the task (continue or train).')
    parser.add_argument('--cr', type=float, default=0.1, help='Specify the compression ratio (cr) for the SE Block.')
    parser.add_argument('--num_epochs', type=int, default=60, help='Specify the number of epochs for training.')
    parser.add_argument('--pre_checkpoint', type=str, default=None, help='Specify the pretrained checkpoint for continue train.')
    parser.add_argument('--channel_type', choices=['AWGN', 'Fading', 'Combined_channel'], default='Combined_channel', help='Specify the channel_type for transfer.')
    parser.add_argument('--dataset_name', type=str, default='Soya', help='Dataset tag used for experiment outputs.')
    parser.add_argument('--train_dir', type=str, default='data/SoyaHealthVision/train', help='ImageFolder train directory.')
    parser.add_argument('--valid_dir', type=str, default='data/SoyaHealthVision/valid', help='ImageFolder validation directory.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training and validation loaders.')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of dataloader workers.')
    parser.add_argument('--output_dir', type=str, default='runs/original_train', help='Output root directory for original-model runs.')
    parser.add_argument('--run_name', type=str, default='', help='Optional custom run folder name under output_dir.')
    args = parser.parse_args()

    main(
        args.task,
        args.cr,
        args.num_epochs,
        args.pre_checkpoint,
        args.channel_type,
        args.dataset_name,
        args.train_dir,
        args.valid_dir,
        args.batch_size,
        args.num_workers,
        args.output_dir,
        args.run_name.strip() if isinstance(args.run_name, str) else "",
    )

