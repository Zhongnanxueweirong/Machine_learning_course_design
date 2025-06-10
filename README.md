# 基于卷积神经网络的手写数字识别

## 项目简介
本项目实现了基于PyTorch的卷积神经网络(CNN)用于MNIST手写数字识别，并与传统机器学习方法进行对比分析。

## 环境配置

### 1. 创建conda环境
```bash
conda create -n ml_project python=3.9
conda activate ml_project
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

## 项目结构
```
handwritten_digit_recognition/
├── data/                    # MNIST数据集（自动下载）
├── models/                  # 模型定义和保存
│   ├── cnn_model.py        # CNN模型定义
│   └── saved/              # 保存的模型文件
├── results/                 # 实验结果
│   ├── figures/            # 可视化图表
│   ├── logs/               # 训练日志
│   └── metrics_summary.json # 评价指标汇总
├── train_cnn.py            # CNN训练脚本
├── train_traditional.py    # 传统方法训练脚本
├── test_all.py            # 统一测试脚本
├── hyperparameter_tuning.py # 超参数调优脚本
└── utils.py               # 工具函数
```

## 运行步骤

### 1. 训练CNN模型
```bash
python train_cnn.py
```
**功能说明：**
- 自动下载MNIST数据集到`data/`目录
- 使用80/20比例划分训练集和验证集
- 训练20个epoch，使用Adam优化器
- 保存最佳模型权重到`models/saved/cnn_best.pth`
- 生成训练损失曲线图
- 记录训练时间和详细日志

**输出文件：**
- `models/saved/cnn_best.pth`: 最佳模型权重
- `results/figures/loss_curves.png`: 损失曲线图
- `results/logs/cnn_training_log.json`: 训练日志

### 2. 训练传统机器学习模型
```bash
python train_traditional.py
```
**功能说明：**
- 训练5种传统机器学习模型
- 对SVM和KNN使用数据标准化
- SVM使用子采样(10000样本)以加快训练速度
- 并行计算加速(RandomForest, KNN)
- 保存所有模型和标准化器

**训练的模型：**
- 逻辑回归 (Logistic Regression)
- 决策树 (Decision Tree)
- 支持向量机 (SVM)
- 随机森林 (Random Forest)
- K近邻 (KNN)

**输出文件：**
- `models/saved/*.pkl`: 各模型文件
- `results/logs/traditional_models_log.json`: 训练日志

### 3. 超参数调优实验
```bash
python hyperparameter_tuning.py
```
**功能说明：**
- 测试4种学习率: [0.0001, 0.001, 0.01, 0.1]
- 测试3种优化器: Adam, SGD, RMSprop
- 每个实验训练10个epoch
- 记录每个实验的训练时间

**输出文件：**
- `results/figures/hyperparameter_analysis.png`: 超参数影响分析图
- `results/logs/hyperparameter_log.json`: 实验日志

### 4. 测试所有模型
```bash
python test_all.py
```
**功能说明：**
- 加载所有已训练的模型
- 在测试集上评估性能
- 计算4个评价指标：准确率、精确率、召回率、F1值
- 生成混淆矩阵和性能对比图
- 输出模型性能排名

**输出文件：**
- `results/figures/confusion_matrix/*.png`: 各模型混淆矩阵
- `results/figures/metrics_comparison.png`: 性能对比图
- `results/metrics_summary.json`: 所有模型评价指标

## 模型训练细节

### CNN模型训练过程
1. **数据预处理**：
   - 归一化到[0,1]区间
   - 标准化(mean=0.1307, std=0.3081)
   
2. **训练策略**：
   - 批量大小：64
   - 学习率：0.001
   - 优化器：Adam
   - 损失函数：交叉熵损失
   - Dropout：0.5（防止过拟合）

3. **早停策略**：
   - 监控验证集损失
   - 保存验证损失最低的模型

### 传统模型训练细节
1. **数据预处理**：
   - 图像展平为784维向量
   - SVM/KNN使用StandardScaler标准化
   
2. **模型参数**：
   - 逻辑回归：L2正则化，max_iter=2000
   - 决策树：最大深度20
   - SVM：RBF核，C=1（优化速度）
   - 随机森林：100棵树，最大深度20
   - KNN：k=5

## 模型预测方式

### 预测单张图片（CNN）
```python
import torch
from models.cnn_model import CNN
from PIL import Image
import torchvision.transforms as transforms

# 加载模型
model = CNN()
model.load_state_dict(torch.load('models/saved/cnn_best.pth'))
model.eval()

# 预处理图片
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 预测
image = Image.open('your_image.png').convert('L')
image_tensor = transform(image).unsqueeze(0)
with torch.no_grad():
    output = model(image_tensor)
    prediction = output.argmax(dim=1).item()
print(f'预测结果: {prediction}')
```

### 批量预测（传统模型）
```python
import joblib
import numpy as np

# 加载模型
model = joblib.load('models/saved/random_forest.pkl')

# 对于SVM/KNN需要加载标准化器
# scaler, model = joblib.load('models/saved/svm.pkl')
# X_test = scaler.transform(X_test)

# 预测
predictions = model.predict(X_test)
```

## 实验结果说明

### 性能指标
- **准确率(Accuracy)**: 正确预测的比例
- **精确率(Precision)**: 预测为正的样本中真正为正的比例
- **召回率(Recall)**: 真正为正的样本被预测为正的比例
- **F1值**: 精确率和召回率的调和平均数

### 日志文件格式
所有日志文件采用JSON格式，包含：
- 模型名称和超参数
- 性能指标
- 训练时间
- 时间戳

## 注意事项
1. 首次运行会自动下载MNIST数据集（约11MB）
2. 完整训练流程大约需要10-15分钟（取决于硬件）
3. SVM训练较慢，已优化为使用10000个样本
4. 建议使用GPU加速CNN训练（自动检测）
5. 所有实验数据都有详细日志记录，方便撰写报告

## 常见问题
1. **内存不足**：减小batch_size或SVM训练样本数
2. **训练太慢**：确保使用GPU；SVM已做优化
3. **中文乱码**：确保系统安装了SimHei字体