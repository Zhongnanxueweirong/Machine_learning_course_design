import torch
import numpy as np
import joblib
import os
from tqdm import tqdm

from models.cnn_model import CNN
from utils import (get_data_loaders, calculate_metrics, plot_confusion_matrix,
                   plot_metrics_comparison, save_metrics, save_training_log)
import time


def test_cnn(device='cuda' if torch.cuda.is_available() else 'cpu'):
    """测试CNN模型"""
    print("测试CNN模型...")

    # 加载模型
    model = CNN().to(device)
    model.load_state_dict(torch.load('models/saved/cnn_best.pth'))
    model.eval()

    # 获取测试数据
    _, test_loader = get_data_loaders(batch_size=64)

    # 预测
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="CNN预测"):
            data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.numpy())

    # 计算评价指标
    metrics = calculate_metrics(all_targets, all_preds)

    # 绘制混淆矩阵
    plot_confusion_matrix(all_targets, all_preds, 'CNN',
                          'results/figures/confusion_matrix/cnn.png')

    return metrics, all_targets, all_preds


def test_traditional_models():
    """测试传统机器学习模型"""
    # 准备数据
    _, test_loader = get_data_loaders(batch_size=1000, flatten=True)

    # 提取测试数据
    X_test, y_test = [], []
    for data, target in test_loader:
        X_test.append(data.numpy())
        y_test.append(target.numpy())

    X_test = np.vstack(X_test)
    y_test = np.hstack(y_test)

    # 模型名称映射
    model_files = {
        '逻辑回归': 'logistic_regression.pkl',
        '决策树':   'decision_tree.pkl',
        'SVM':     'svm.pkl',
        '随机森林': 'random_forest.pkl',
        'KNN':     'knn.pkl'
    }

    results = {}

    for name, file in model_files.items():
        print(f"\n测试 {name}...")

        model_path = f'models/saved/{file}'
        if not os.path.exists(model_path):
            print(f"警告: {model_path} 不存在，跳过")
            continue

        # 先 load
        loaded = joblib.load(model_path)

        # 如果 load 出来的是 (scaler, model) 这种 tuple，就解包
        if isinstance(loaded, tuple) and len(loaded) == 2:
            scaler, model = loaded
            X_test_use = scaler.transform(X_test)
        else:
            model = loaded
            X_test_use = X_test

        # 预测
        y_pred = model.predict(X_test_use)

        # 计算指标并保存结果
        metrics = calculate_metrics(y_test, y_pred)
        results[name] = metrics

        # 混淆矩阵
        plot_confusion_matrix(
            y_test, y_pred, name,
            f'results/figures/confusion_matrix/{name.lower().replace(" ", "_")}.png'
        )

        # 打印
        print(f"{name} 性能:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

    return results, y_test



def test_all_models():
    """测试所有模型并生成对比报告"""
    print("=" * 50)
    print("开始测试所有模型")
    print("=" * 50)

    all_metrics = {}

    # 测试CNN
    cnn_metrics, _, _ = test_cnn()
    all_metrics['CNN'] = cnn_metrics

    # 测试传统模型
    traditional_metrics, _ = test_traditional_models()
    all_metrics.update(traditional_metrics)

    # 生成对比图
    print("\n生成性能对比图...")
    plot_metrics_comparison(all_metrics, 'results/figures/metrics_comparison.png')

    # 保存所有评价指标
    save_metrics(all_metrics, 'results/metrics_summary.json')

    # 打印汇总报告
    print("\n" + "=" * 50)
    print("模型性能汇总")
    print("=" * 50)

    # 找出每个指标的最佳模型
    metrics_names = ['准确率', '精确率', '召回率', 'F1值']
    for metric in metrics_names:
        best_model = max(all_metrics.items(), key=lambda x: x[1][metric])
        print(f"\n{metric}最高的模型: {best_model[0]} ({best_model[1][metric]:.4f})")

    # 打印详细对比表
    print("\n详细性能对比:")
    print(f"{'模型':<12} {'准确率':<10} {'精确率':<10} {'召回率':<10} {'F1值':<10}")
    print("-" * 52)

    for model_name, metrics in all_metrics.items():
        print(f"{model_name:<12} "
              f"{metrics['准确率']:<10.4f} "
              f"{metrics['精确率']:<10.4f} "
              f"{metrics['召回率']:<10.4f} "
              f"{metrics['F1值']:<10.4f}")

    print("\n所有测试完成！结果已保存到 results/ 目录")


if __name__ == "__main__":
    test_all_models()
