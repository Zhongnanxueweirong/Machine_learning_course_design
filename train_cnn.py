import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
import numpy as np
from tqdm import tqdm
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.cnn_model import CNN
from utils import get_data_loaders, plot_loss_curves, calculate_metrics, save_training_log
import time


def train_cnn(epochs=20, lr=0.001, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """训练CNN模型"""
    print(f"使用设备: {device}")

    # 记录开始时间
    start_time = time.time()

    # 获取数据
    train_loader, test_loader = get_data_loaders(batch_size=64)

    # 划分训练集和验证集
    train_dataset = train_loader.dataset
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)

    # 初始化模型
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 记录损失
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    print("开始训练CNN模型...")
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs} [训练]')

        for data, target in train_bar:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_bar.set_postfix({'loss': loss.item()})

        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        # 计算平均损失
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_accuracy = correct / len(val_dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f'Epoch {epoch + 1}: 训练损失={train_loss:.4f}, 验证损失={val_loss:.4f}, 验证准确率={val_accuracy:.4f}')

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs('models/saved', exist_ok=True)
            torch.save(model.state_dict(), 'models/saved/cnn_best.pth')
            print(f'保存最佳模型 (验证损失: {best_val_loss:.4f})')

    # 绘制损失曲线
    plot_loss_curves(train_losses, val_losses, 'results/figures/loss_curves.png')
    print("训练完成！损失曲线已保存到 results/figures/loss_curves.png")

    # 在测试集上评估
    print("\n在测试集上评估模型...")
    model.load_state_dict(torch.load('models/saved/cnn_best.pth'))
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.numpy())

    # 计算评价指标
    metrics = calculate_metrics(all_targets, all_preds)
    print("\nCNN模型测试集性能:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # 计算总训练时间
    total_time = time.time() - start_time
    print(f"\n总训练时间: {total_time:.2f} 秒 ({total_time / 60:.2f} 分钟)")

    # 保存训练日志
    training_log = {
        '模型': 'CNN',
        '超参数': {
            'epochs': epochs,
            'learning_rate': lr,
            'batch_size': 64,
            'optimizer': 'Adam'
        },
        '性能指标': metrics,
        '训练时间': f'{total_time:.2f}秒',
        '最佳验证损失': float(best_val_loss),
        '设备': device
    }
    save_training_log(training_log, 'cnn_training_log.json')
    print("训练日志已保存到 results/logs/cnn_training_log.json")

    return model, metrics


if __name__ == "__main__":
    train_cnn()
