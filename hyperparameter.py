import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
import numpy as np
from tqdm import tqdm
import os

from models.cnn_model import CNN
from utils import get_data_loaders, plot_hyperparameter_analysis, save_training_log
import time


def train_with_hyperparameters(lr, optimizer_name, epochs=10, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """使用指定超参数训练模型"""
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

    # 选择优化器
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr)

    # 训练模型
    best_val_acc = 0.0

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        # 验证阶段
        model.eval()
        correct = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        val_acc = correct / len(val_dataset)
        best_val_acc = max(best_val_acc, val_acc)

    # 在测试集上评估
    model.eval()
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_acc = correct / len(test_loader.dataset)

    return test_acc


def hyperparameter_tuning():
    """执行超参数调优实验"""
    print("开始超参数调优实验...")

    # 记录开始时间
    start_time = time.time()

    # 定义超参数组合
    learning_rates = [0.0001, 0.001, 0.01, 0.1]
    optimizers = ['Adam', 'SGD', 'RMSprop']

    results = []
    all_logs = []

    # 测试不同学习率（使用Adam优化器）
    print("\n测试不同学习率的影响...")
    for lr in learning_rates:
        print(f"训练模型: lr={lr}, optimizer=Adam")
        exp_start = time.time()
        acc = train_with_hyperparameters(lr, 'Adam', epochs=10)
        exp_time = time.time() - exp_start

        results.append({
            'lr': lr,
            'optimizer': 'Adam',
            'accuracy': acc
        })
        print(f"测试准确率: {acc:.4f}, 训练时间: {exp_time:.2f}秒")

        # 记录日志
        log_entry = {
            '实验类型': '学习率测试',
            '学习率': lr,
            '优化器': 'Adam',
            '准确率': float(acc),
            '训练时间': f'{exp_time:.2f}秒'
        }
        all_logs.append(log_entry)

    # 测试不同优化器（使用最佳学习率）
    best_lr = 0.001  # 通常是较好的默认值
    print(f"\n测试不同优化器的影响 (lr={best_lr})...")
    for opt in optimizers:
        if opt == 'Adam' and best_lr == 0.001:
            continue  # 已经测试过

        print(f"训练模型: lr={best_lr}, optimizer={opt}")
        exp_start = time.time()
        acc = train_with_hyperparameters(best_lr, opt, epochs=10)
        exp_time = time.time() - exp_start

        results.append({
            'lr': best_lr,
            'optimizer': opt,
            'accuracy': acc
        })
        print(f"测试准确率: {acc:.4f}, 训练时间: {exp_time:.2f}秒")

        # 记录日志
        log_entry = {
            '实验类型': '优化器测试',
            '学习率': best_lr,
            '优化器': opt,
            '准确率': float(acc),
            '训练时间': f'{exp_time:.2f}秒'
        }
        all_logs.append(log_entry)

    # 绘制分析图
    plot_hyperparameter_analysis(results, 'results/figures/hyperparameter_analysis.png')
    print("\n超参数分析图已保存到 results/figures/hyperparameter_analysis.png")

    # 打印最佳结果
    best_result = max(results, key=lambda x: x['accuracy'])
    print(f"\n最佳超参数组合:")
    print(f"学习率: {best_result['lr']}")
    print(f"优化器: {best_result['optimizer']}")
    print(f"测试准确率: {best_result['accuracy']:.4f}")

    # 计算总时间
    total_time = time.time() - start_time
    print(f"\n超参数调优总时间: {total_time:.2f} 秒 ({total_time / 60:.2f} 分钟)")

    # 保存所有实验日志
    for log in all_logs:
        save_training_log(log, 'hyperparameter_log.json')
    print("实验日志已保存到 results/logs/hyperparameter_log.json")



    return results


if __name__ == "__main__":
    hyperparameter_tuning()
