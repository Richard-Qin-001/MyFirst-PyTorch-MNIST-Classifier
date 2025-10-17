import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import os

# 目录设置
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')
'''
超参数设置
BATCH_SIZE：批量大小，一次性将多少张图片送入网络进行训练
64 是常用的数值，数值太小训练不稳定，太大占用内存多。
Python 没有常量，但是可以类似定义一个全局变量 
'''
BATCH_SIZE = 64
'''
数据预处理
transforms.Compose() 将多个预处理操作按顺序组合起来
'''
transform = transforms.Compose([
    
    # 1. 将 PIL 图像 (或 NumPy 数组) 转换为 PyTorch 张量
    #    - 它会将数据从 [H, W, C] (高, 宽, 通道) 转换为 PyTorch 的 [C, H, W] 格式。
    #    - 它还会将像素值从 0-255 范围缩放到 0.0-1.0 范围。
    
    transforms.ToTensor(), 
    
    # 2. 标准化 (Normalization)：将像素值调整到均值为 0、标准差为 1 的分布。
    #     - 公式：输出 = (输入 - 均值) / 标准差
    #     - MNIST 是单通道灰度图，所以均值和标准差都只写一个值。
    #     - 0.1307 是 MNIST 训练集的平均值，0.3081 是标准差，这是公认的经验值。
    
    transforms.Normalize((0.1307,), (0.3081,))
])

# 数据集的下载
# 训练集
train_dataset = torchvision.datasets.MNIST(
    root=DATA_ROOT,           # 数据的存储位置
    train=True,              # 指定我们要获取的是训练集（包含 60,000 张图片）
    download=True,           # 如果本地没有数据，则自动从网上下载
    transform=transform      # 应用预处理流程
)

# 测试集
test_dataset = torchvision.datasets.MNIST(
    root=DATA_ROOT,
    train=False,             # 指定获取测试集（包含 10,000 张图片）
    download=True,           # 自动下载
    transform=transform      # 应用预处理流程
)

print(f"训练集图像数量: {len(train_dataset)}")
print(f"测试集图像数量: {len(test_dataset)}")

class SimpleCNN(nn.Module):
    # 定义网络的结构，‘层’
    def __init__(self, *args, **kwargs):
        super(SimpleCNN, self).__init__(*args, **kwargs)

        # 1. 卷积层 1 (用于提取基本特征)
        # nn.Conv2d(in_channels, out_channels, kernel_size)
        # MNIST 是灰度图，所以输入通道 in_channels=1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        
        # 2. 卷积层 2 (用于提取更复杂的特征)
        # 输入通道是上一层的 out_channels=10
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
        
        # 3. Dropout 层 (防止在卷积层之间产生过拟合)
        self.conv2_drop = nn.Dropout2d()  # 随机关闭一些神经元
        
        # 4. 全连接层 1 (将特征连接到 50 个神经元)
        # 320 是根据卷积和池化操作计算出的特征数量
        self.fc1 = nn.Linear(in_features=320, out_features=50) 
        
        # 5. 全连接层 2 (输出层)
        # 输出是 10 个类别 (0, 1, ..., 9)
        self.fc2 = nn.Linear(in_features=50, out_features=10)

    # 定义数据 x 如何流过上面定义的层
    def forward():
        pass
    
    


if __name__ == "__main__":
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的计算设备: {device}")


    # 数据加载器DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, # 要加载的数据集
        batch_size=BATCH_SIZE, # 每批次加载数量
        shuffle=True, # 打乱数据
        num_workers=4 # 用四个子进程并行加载    
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,         # 指定要加载的数据集（测试集）
        batch_size=BATCH_SIZE,
        shuffle=False,         # 测试时不需要打乱，方便按顺序评估
        num_workers=4
    )

    print("-" * 30)
    print(f"训练集的批次数量 (总图片数 / 批量大小): {len(train_loader)}")
    print(f"测试集的批次数量: {len(test_loader)}")

    # --- 简单检查：打印一个批次的数据形状 ---
    # next(iter(train_loader)) 用于从加载器中取出第一个批次的数据
    images, labels = next(iter(train_loader)) 

    # 形状：[批量大小, 通道数, 图像高度, 图像宽度]
    # MNIST 是 64x1x28x28
    print(f"一个批次的图像张量形状: {images.shape}") 
    # 形状：[批量大小]，即 64 个标签
    print(f"一个批次的标签张量形状: {labels.shape}")