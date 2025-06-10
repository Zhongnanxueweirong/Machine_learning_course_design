import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    """卷积神经网络模型"""
    def __init__(self):
        super(CNN, self).__init__()
        # 第一个卷积层：1通道输入，32个3x3卷积核
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # 第二个卷积层：32通道输入，64个3x3卷积核
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # 最大池化层
        self.pool = nn.MaxPool2d(2, 2)
        # 全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        # Dropout层
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # 第一个卷积块：卷积->ReLU->池化
        x = self.pool(F.relu(self.conv1(x)))  # 28x28 -> 14x14
        # 第二个卷积块：卷积->ReLU->池化
        x = self.pool(F.relu(self.conv2(x)))  # 14x14 -> 7x7
        # 展平
        x = x.view(-1, 64 * 7 * 7)
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def predict(self, x):
        """预测函数"""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            _, predicted = torch.max(outputs, 1)
        return predicted