import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import json
import os
import time
from datetime import datetime

# 设置中文字体，防止乱码
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows
plt.rcParams['axes.unicode_minus'] = False


def get_data_loaders(batch_size=64, flatten=False):
    """获取MNIST数据加载器"""
    # 数据预处理
    if flatten:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.view(-1))  # 展平为向量
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    # 下载并加载数据集
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def calculate_metrics(y_true, y_pred):
    """计算评价指标"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    return {
        '准确率': accuracy,
        '精确率': precision,
        '召回率': recall,
        'F1值': f1
    }


def plot_confusion_matrix(y_true, y_pred, title, save_path):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10))
    plt.title(f'{title} - 混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.tight_layout()

    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_loss_curves(train_losses, val_losses, save_path):
    """绘制损失曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='训练损失', linewidth=2)
    plt.plot(val_losses, label='验证损失', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('损失值')
    plt.title('训练过程损失曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_metrics_comparison(metrics_dict, save_path):
    """绘制模型性能对比图"""
    models = list(metrics_dict.keys())
    metrics = ['准确率', '精确率', '召回率', 'F1值']

    # 准备数据
    data = {metric: [metrics_dict[model][metric] for model in models] for metric in metrics}

    # 设置图形
    x = np.arange(len(models))
    width = 0.2
    fig, ax = plt.subplots(figsize=(12, 6))

    # 绘制柱状图
    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, data[metric], width, label=metric)

    # 设置图形属性
    ax.set_xlabel('模型')
    ax.set_ylabel('分数')
    ax.set_title('不同模型性能对比')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for i, metric in enumerate(metrics):
        for j, v in enumerate(data[metric]):
            ax.text(j + i * width, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_training_log(log_data, filename):
    """保存训练日志"""
    os.makedirs('results/logs', exist_ok=True)
    log_path = f'results/logs/{filename}'

    # 添加时间戳
    log_data['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # 如果文件已存在，读取现有数据并追加
    if os.path.exists(log_path):
        with open(log_path, 'r', encoding='utf-8') as f:
            existing_logs = json.load(f)
        if isinstance(existing_logs, list):
            existing_logs.append(log_data)
        else:
            existing_logs = [existing_logs, log_data]
        logs_to_save = existing_logs
    else:
        logs_to_save = [log_data]

    # 保存日志
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(logs_to_save, f, ensure_ascii=False, indent=2)


def save_metrics(metrics_dict, save_path):
    """保存评价指标到JSON文件"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_dict, f, ensure_ascii=False, indent=2)


def plot_hyperparameter_analysis(results, save_path):
    """绘制超参数分析图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 学习率影响
    lr_results = [r for r in results if r['optimizer'] == 'Adam']
    lrs = [r['lr'] for r in lr_results]
    accs = [r['accuracy'] for r in lr_results]

    ax1.plot(lrs, accs, 'o-', markersize=8, linewidth=2)
    ax1.set_xlabel('学习率')
    ax1.set_ylabel('准确率')
    ax1.set_title('学习率对模型性能的影响')
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)

    # 优化器对比
    opt_names = list(set([r['optimizer'] for r in results]))
    opt_accs = []
    for opt in opt_names:
        opt_results = [r for r in results if r['optimizer'] == opt]
        avg_acc = np.mean([r['accuracy'] for r in opt_results])
        opt_accs.append(avg_acc)

    ax2.bar(opt_names, opt_accs)
    ax2.set_xlabel('优化器')
    ax2.set_ylabel('平均准确率')
    ax2.set_title('不同优化器的性能对比')
    ax2.grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for i, v in enumerate(opt_accs):
        ax2.text(i, v + 0.001, f'{v:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
