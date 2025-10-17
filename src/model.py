import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os

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
    def forward(self, x):
        # x 的初始形状是：[BATCH_SIZE, 1, 28, 28]

        # 1. 卷积层 1 (Conv1) -> ReLU -> Max Pooling 
        # 输出形状：[BATCH_SIZE, 10, 12, 12]
        x = self.conv1(x)         # 应用卷积层 1
        x = F.relu(x)             # 应用 ReLU 激活函数
        x = F.max_pool2d(x, 2)    # 应用 2x2 最大池化（尺寸减半）
        
        # 2. 卷积层 2 (Conv2) -> Dropout -> ReLU -> Max Pooling
        # 输出形状：[BATCH_SIZE, 20, 4, 4]
        x = self.conv2(x)         # 应用卷积层 2
        x = self.conv2_drop(x)    # 应用 Dropout (仅在训练时有效)
        x = F.relu(x)             # 应用 ReLU 激活函数
        x = F.max_pool2d(x, 2)    # 应用 2x2 最大池化（尺寸减半）
        
        # 3. 展平 (Flatten)：从 [Batch, 20, 4, 4] 变为 [Batch, 320]
        # x.size(0) 是获取当前的批量大小（例如 64）
        x = x.view(x.size(0), -1) 
        
        # 4. 全连接层 1 (FC1) -> ReLU
        # 输出形状：[BATCH_SIZE, 50]
        x = self.fc1(x)
        x = F.relu(x)
        
        # 5. Dropout (防止全连接层过拟合)
        x = F.dropout(x, training=self.training)
        
        # 6. 全连接层 2 (FC2) - 输出层
        # 最终输出形状：[BATCH_SIZE, 10]
        x = self.fc2(x)
        
        return x
    
def load_and_preprocess_image(image_path, device):
    img = Image.open(image_path).convert('L') 
    external_transform = transforms.Compose([
        transforms.Resize((28, 28)), # 确保图片尺寸是 28x28
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) 
    ])
    tensor = external_transform(img).unsqueeze(0).to(device)
    
    return tensor
