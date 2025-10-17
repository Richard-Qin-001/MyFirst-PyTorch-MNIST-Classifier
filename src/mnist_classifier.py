import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import os
import json
from PIL import Image

from model import SimpleCNN

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



def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    # 遍历 train_loader 中的所有数据批次
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() # 梯度归零
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')    

def test(model, device, test_loader, criterion):
    model.eval() 
    
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # reduction='sum' 表示计算当前批次的损失总和
            test_loss += criterion(output, target).item() 
            # output.data.max(1) 返回每一行（即每个样本）最大值的 (值, 索引)
            # pred 是索引，即预测的类别 (0-9)
            pred = output.data.max(1, keepdim=True)[1] 
            # pred.eq(target.data.view_as(pred)) 比较预测和真实标签，返回布尔张量
            # .sum() 将 True 转换为 1，False 转换为 0，并求和
            correct += pred.eq(target.data.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)\n')
    
    return accuracy

def predict(model, device, test_dataset, image_index, criterion):
    model_save_path = os.path.join(PROJECT_ROOT, 'models', f'mnist_cnn_best.pth')
    loaded_model = SimpleCNN().to(device)
    loaded_model.load_state_dict(torch.load(model_save_path))
    loaded_model.eval()
    image_tensor, true_label = test_dataset[image_index]
    input_tensor = image_tensor.unsqueeze(0).to(device) 
    with torch.no_grad():
        output = loaded_model(input_tensor)
        # Softmax 将 Logits 转换为概率
        probabilities = F.softmax(output, dim=1)
        # 获取最大概率的索引，即预测的类别
        predicted_class = probabilities.argmax(dim=1).item() 
        
    print("-" * 30)
    print(f"--- 预测结果 ---")
    print(f"图片索引: {image_index}")
    print(f"真实标签 (True Label): {true_label}")
    print(f"模型预测类别 (Prediction): {predicted_class}")
    print(f"概率分布: {probabilities.cpu().squeeze().tolist()}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的计算设备: {device}")
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

    images, labels = next(iter(train_loader)) 

    # 形状：[批量大小, 通道数, 图像高度, 图像宽度]
    # MNIST 是 64x1x28x28
    print(f"一个批次的图像张量形状: {images.shape}") 
    # 形状：[批量大小]，即 64 个标签
    print(f"一个批次的标签张量形状: {labels.shape}")
    #实例化模型并将其移动到 GPU
    model = SimpleCNN()
    model.to(device)  # 移动到显存
    print("-" * 30)
    print("模型已成功实例化并传输到:", device)

    # 损失函数 (Loss Function)
    # nn.CrossEntropyLoss 适用于多分类问题，它结合了 Softmax 和负对数似然损失。
    criterion = nn.CrossEntropyLoss()

    # 优化器 (Optimizer)
    # Adam 能自动调整学习率。
    # model.parameters()：告诉优化器需要更新哪些参数
    # lr=0.001：学习率 (Learning Rate)，控制每一步参数更新的幅度
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model_save_path = os.path.join(PROJECT_ROOT, 'models', 'mnist_cnn_best.pth')
    start_epoch = 1

    if os.path.exists(model_save_path):
        checkpoint = torch.load(model_save_path)
        model.load_state_dict(checkpoint['model_state_dict']) # 恢复模型权重
        
        optimizer.load_state_dict(checkpoint['optimizer_state_dict']) # 恢复优化器状态

        start_epoch = checkpoint['epoch'] + 1

        best_accuracy = checkpoint['best_accuracy']
        
        print("-" * 30)
        print(f"检测到已保存的模型！从第 {start_epoch} 周期继续训练，当前最佳准确率: {best_accuracy:.2f}%")
        print("-" * 30)
    else:
        best_accuracy = 0.0
        print("-" * 30)
        print("未检测到模型文件，开始全新训练。")
        print("-" * 30)

    # 超参数：训练周期 (Epochs)
    NUM_EPOCHS = 10

    print("-" * 30)
    print(f"开始在 {device} 上训练模型...")
    print("-" * 30)
    best_accuracy = 0.0
    for epoch in range(1, NUM_EPOCHS + 1):
        train(model, device, train_loader, optimizer, criterion, epoch)

        current_accuracy = test(model, device, test_loader, criterion)

        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_accuracy': best_accuracy
            }
            model_save_path = os.path.join(PROJECT_ROOT, 'models', f'mnist_cnn_best.pth')
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(checkpoint, model_save_path)
            # torch.save(model.state_dict(), model_save_path)
            print(f"模型准确率提高到 {best_accuracy:.2f}%，已保存到 {model_save_path}")

    print("-" * 30)
    print(f"训练完成！最高测试准确率为: {best_accuracy:.2f}%")