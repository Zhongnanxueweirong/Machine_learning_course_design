import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os
from tqdm import tqdm

from utils import get_data_loaders, calculate_metrics, save_training_log
import time


def prepare_data():
    """准备传统ML方法所需的数据"""
    print("准备数据...")
    train_loader, test_loader = get_data_loaders(batch_size=1000, flatten=True)

    # 提取所有训练数据
    X_train, y_train = [], []
    for data, target in tqdm(train_loader, desc="加载训练数据"):
        X_train.append(data.numpy())
        y_train.append(target.numpy())

    X_train = np.vstack(X_train)
    y_train = np.hstack(y_train)

    # 提取所有测试数据
    X_test, y_test = [], []
    for data, target in tqdm(test_loader, desc="加载测试数据"):
        X_test.append(data.numpy())
        y_test.append(target.numpy())

    X_test = np.vstack(X_test)
    y_test = np.hstack(y_test)

    return X_train, y_train, X_test, y_test


def train_traditional_models():
    """训练传统机器学习模型"""
    # 准备数据
    X_train, y_train, X_test, y_test = prepare_data()

    # 数据标准化（对SVM特别重要）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 定义模型
    models = {
        'logistic_regression': LogisticRegression(max_iter=2000, solver='saga', random_state=42),
        'decision_tree': DecisionTreeClassifier(max_depth=20, random_state=42),
        'SVM': SVC(kernel='rbf', C=10, gamma='scale', random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }
    
    # 训练和评估每个模型
    results = {}

    os.makedirs('models/saved', exist_ok=True)
    
    for name, model in models.items():
        print(f"\n训练 {name}...")
        
        # 选择是否使用标准化数据
        if name in ['logistic_regression', 'SVM', 'KNN']:
            X_train_use = X_train_scaled
            X_test_use = X_test_scaled
        else:
            X_train_use = X_train
            X_test_use = X_test
        
        # 训练模型
        model.fit(X_train_use, y_train)
        
        # 预测
        y_pred = model.predict(X_test_use)
        
        # 计算评价指标
        metrics = calculate_metrics(y_test, y_pred)
        results[name] = metrics
        
        # 保存模型
        if name in ['logistic_regression', 'SVM', 'KNN']:
            # 同时保存标准化器
            joblib.dump((scaler, model), f'models/saved/{name.lower()}.pkl')
        else:
            joblib.dump(model, f'models/saved/{name.lower()}.pkl')
        
        # 打印结果
        print(f"{name} 测试集性能:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    print("\n所有传统模型训练完成！")
    return results



if __name__ == "__main__":
    train_traditional_models()
