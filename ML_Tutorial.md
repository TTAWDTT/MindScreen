# 心理健康预测模型 - 原理与使用指南

## 目录
1. [项目概述](#项目概述)
2. [机器学习基本原理](#机器学习基本原理)
3. [三种模型原理详解](#三种模型原理详解)
4. [模型评估指标解读](#模型评估指标解读)
5. [模型使用方法](#模型使用方法)
6. [实验结果分析](#实验结果分析)

---

## 项目概述

本项目利用机器学习技术，根据用户的日常行为数据预测心理健康指标。

### 输入特征（X）
| 特征名 | 说明 | 示例值 |
|--------|------|--------|
| daily_screen_time_min | 每日屏幕使用时长（分钟） | 300 |
| social_media_time_min | 社交媒体使用时长（分钟） | 150 |
| sleep_hours | 睡眠时间（小时） | 7.0 |
| age | 年龄 | 25 |
| gender | 性别（编码后：Female=0, Male=1, Other=2） | 1 |
| physical_activity_min | 身体活动时间（分钟） | 30 |

### 预测目标（Y）
| 目标 | 说明 | 取值范围 |
|------|------|----------|
| stress_level | 压力指数 | 1-10 |
| anxiety_level | 焦虑指数 | 1-5 |

---

## 机器学习基本原理

### 什么是监督学习？

监督学习是机器学习的一种方法，通过已标注的数据（有输入X和对应输出Y）来训练模型。

```
训练过程：
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  输入特征 X  │ --> │   模型学习   │ --> │  预测输出 Y  │
│ (屏幕时长等) │     │  (找规律)    │     │ (压力/焦虑)  │
└─────────────┘     └─────────────┘     └─────────────┘
```

### 回归 vs 分类

- **回归问题**：预测连续数值（如：压力指数 7.5）← 本项目属于此类
- **分类问题**：预测离散类别（如：高压力/低压力）

### 数据划分

```
全部数据 (5000条)
    │
    ├── 训练集 (80% = 4000条) → 用于模型学习
    │
    └── 测试集 (20% = 1000条) → 用于评估模型效果
```

---

## 三种模型原理详解

### 1. 决策树 (Decision Tree)

#### 基本原理
决策树通过一系列"是/否"问题将数据分类，类似于玩"20个问题"游戏。

```
                    [屏幕时间 > 400分钟?]
                    /                    \
                  是                      否
                  /                        \
        [睡眠 < 6小时?]              [运动 < 20分钟?]
        /           \                /            \
      是            否              是             否
      /              \              /               \
   高压力         中压力         中压力           低压力
```

#### 关键参数
```python
DecisionTreeRegressor(
    max_depth=10,        # 树的最大深度，防止过拟合
    min_samples_split=5, # 节点分裂所需最小样本数
    min_samples_leaf=2   # 叶子节点最小样本数
)
```

#### 优缺点
| 优点 | 缺点 |
|------|------|
| ✅ 易于理解和解释 | ❌ 容易过拟合 |
| ✅ 不需要特征标准化 | ❌ 对数据微小变化敏感 |
| ✅ 训练速度快 | ❌ 可能产生偏向性决策 |

---

### 2. 支持向量机 (SVM)

#### 基本原理
SVM 试图找到一个最优的"超平面"来划分数据，对于回归问题（SVR），它在允许一定误差范围内拟合数据。

```
        Y (压力指数)
        │      _______________
        │     /   ε-管道      \    ← 允许的误差范围
        │    /  ○ ○ ○ ○ ○ ○   \
        │   /──────────────────\  ← 回归线
        │  /    ○  ○  ○  ○      \
        │ /________________________\
        └────────────────────────────→ X (特征)
```

#### 关键参数
```python
SVR(
    kernel='rbf',   # 核函数：rbf(高斯核)可处理非线性关系
    C=1.0,          # 正则化参数：越大越严格拟合
    epsilon=0.1     # ε-管道宽度：允许的误差范围
)
```

#### 核函数说明
| 核函数 | 适用场景 |
|--------|----------|
| linear | 线性可分数据 |
| rbf | 非线性数据（最常用） |
| poly | 多项式关系数据 |

#### 优缺点
| 优点 | 缺点 |
|------|------|
| ✅ 在高维空间表现好 | ❌ 需要特征标准化 |
| ✅ 对过拟合有较好控制 | ❌ 大数据集训练慢 |
| ✅ 核函数灵活 | ❌ 参数调优复杂 |

---

### 3. 多层感知机 (MLP)

#### 基本原理
MLP 是一种人工神经网络，由多层神经元组成，通过层层传递和激活函数学习复杂模式。

```
输入层          隐藏层1        隐藏层2        隐藏层3        输出层
(6个特征)       (64神经元)     (32神经元)     (16神经元)     (2个输出)

  ○──────────────○             ○             ○──────────────○ 压力指数
屏幕时长         ╲           ╱ ╲           ╱ 
  ○──────────────○─────────○───○─────────○───○──────────────○ 焦虑指数
社交时长         ╱           ╲ ╱           ╲
  ○──────────────○             ○             ○
睡眠时间
  ○
年龄
  ○
性别
  ○
运动时间
```

#### 关键参数
```python
MLPRegressor(
    hidden_layer_sizes=(64, 32, 16),  # 三个隐藏层的神经元数
    activation='relu',                 # 激活函数
    solver='adam',                     # 优化器
    max_iter=500,                      # 最大迭代次数
    early_stopping=True                # 早停防止过拟合
)
```

#### 激活函数
```
ReLU: f(x) = max(0, x)

        │      ╱
        │     ╱
        │    ╱
   ─────┼───╱─────→
        │  0
        │
```

#### 优缺点
| 优点 | 缺点 |
|------|------|
| ✅ 能学习复杂非线性关系 | ❌ 需要大量数据 |
| ✅ 自动特征提取 | ❌ 训练时间长 |
| ✅ 适用于多种问题 | ❌ "黑盒"模型，难解释 |

---

## 模型评估指标解读

### 核心指标

| 指标 | 公式 | 含义 | 理想值 |
|------|------|------|--------|
| **MSE** | $\frac{1}{n}\sum(y_{真实}-y_{预测})^2$ | 均方误差 | 越小越好 |
| **RMSE** | $\sqrt{MSE}$ | 均方根误差（与原数据同单位） | 越小越好 |
| **MAE** | $\frac{1}{n}\sum\|y_{真实}-y_{预测}\|$ | 平均绝对误差 | 越小越好 |
| **R²** | $1 - \frac{SS_{res}}{SS_{tot}}$ | 决定系数（解释方差比例） | 越接近1越好 |

### R² 分数解读

```
R² = 1.00  → 完美预测 ⭐⭐⭐⭐⭐
R² > 0.90  → 优秀     ⭐⭐⭐⭐
R² > 0.70  → 良好     ⭐⭐⭐
R² > 0.50  → 一般     ⭐⭐
R² < 0.50  → 较差     ⭐
```

---

## 模型使用方法

### 方法一：运行训练脚本

```bash
cd D:\Github\MindScreen
python train_models.py
```

### 方法二：在Python中使用训练好的模型

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor

# 1. 加载并准备数据
df = pd.read_csv('mental_health_social_media_dataset.csv')

# 2. 编码性别
le = LabelEncoder()
df['gender_encoded'] = le.fit_transform(df['gender'])

# 3. 准备特征和目标
X = df[['daily_screen_time_min', 'social_media_time_min', 'sleep_hours', 
        'age', 'gender_encoded', 'physical_activity_min']].values
y = df[['stress_level', 'anxiety_level']].values

# 4. 训练模型（使用表现最好的决策树）
model = MultiOutputRegressor(DecisionTreeRegressor(
    max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42
))
model.fit(X, y)

# 5. 进行预测
def predict_mental_health(screen_time, social_time, sleep, age, gender, exercise):
    """
    预测心理健康指标
    
    参数:
        screen_time: 屏幕使用时长(分钟)
        social_time: 社交媒体时长(分钟)
        sleep: 睡眠时间(小时)
        age: 年龄
        gender: 性别 (0=Female, 1=Male, 2=Other)
        exercise: 运动时间(分钟)
    
    返回:
        (压力指数, 焦虑指数)
    """
    input_data = np.array([[screen_time, social_time, sleep, age, gender, exercise]])
    prediction = model.predict(input_data)[0]
    return prediction[0], prediction[1]

# 使用示例
stress, anxiety = predict_mental_health(
    screen_time=300,   # 5小时屏幕时间
    social_time=150,   # 2.5小时社交媒体
    sleep=7.0,         # 7小时睡眠
    age=25,            # 25岁
    gender=1,          # 男性
    exercise=30        # 30分钟运动
)

print(f"预测结果:")
print(f"  压力指数: {stress:.2f}")
print(f"  焦虑指数: {anxiety:.2f}")
```

### 方法三：保存和加载模型

```python
import joblib

# 保存模型
joblib.dump(model, 'mental_health_model.pkl')
joblib.dump(scaler, 'scaler.pkl')  # 如果使用SVM或MLP需要保存标准化器

# 加载模型
loaded_model = joblib.load('mental_health_model.pkl')
loaded_scaler = joblib.load('scaler.pkl')

# 使用加载的模型预测
prediction = loaded_model.predict(new_data)
```

---

## 实验结果分析

### 模型性能对比

#### 压力指数预测
| 模型 | MSE | RMSE | MAE | R² |
|------|-----|------|-----|-----|
| 🥇 决策树 | 0.0050 | 0.0707 | 0.0065 | **0.9957** |
| 🥈 多层感知机 | 0.0422 | 0.2053 | 0.1377 | 0.9635 |
| 🥉 支持向量机 | 0.0827 | 0.2876 | 0.2139 | 0.9284 |

#### 焦虑指数预测
| 模型 | MSE | RMSE | MAE | R² |
|------|-----|------|-----|-----|
| 🥇 决策树 | 0.0019 | 0.0433 | 0.0030 | **0.9972** |
| 🥈 多层感知机 | 0.0270 | 0.1644 | 0.1039 | 0.9595 |
| 🥉 支持向量机 | 0.0694 | 0.2634 | 0.2019 | 0.8962 |

### 结论与建议

1. **最佳模型**：决策树在本数据集上表现最优，R² > 0.99
2. **原因分析**：
   - 数据特征与目标之间可能存在较强的规则性关系
   - 决策树能很好地捕捉这种规则
3. **使用建议**：
   - 生产环境推荐使用决策树模型
   - 如需更好的泛化能力，可考虑随机森林（多棵决策树集成）

### 注意事项

⚠️ **过拟合警告**：R² 接近 1.0 可能暗示过拟合，建议：
- 使用交叉验证进一步验证
- 在完全独立的新数据上测试
- 考虑使用正则化或集成方法

---

## 产品部署与使用

### 快速开始

```bash
# 1. 训练并保存模型
python save_model.py

# 2. 启动交互式预测
python predict.py
```

### 命令行选项

```bash
python predict.py              # 交互式模式
python predict.py --help       # 查看帮助
python predict.py --demo       # 演示模式
python predict.py --info       # 显示模型信息
```

### 交互式预测示例

```
============================================================
       🧠 MindScreen - 心理健康预测系统 v1.0
============================================================

按回车开始预测 (或输入命令): 

📱 每日屏幕时间(分钟) [300]: 400
💬 社交媒体时间(分钟) [120]: 200
😴 睡眠时间(小时) [7.0]: 6
🎂 年龄 [25]: 28
⚧️ 性别: 1=Female, 2=Male, 3=Other
   请选择 [Male]: 2
🏃 运动时间(分钟) [30]: 15

========================================
           📊 预测结果
========================================

  压力指数: 7.5 / 10
           😟 较高压力

  焦虑指数: 3.2 / 5
           😧 中度焦虑
```

### 在代码中调用

```python
from predict import MentalHealthPredictor

# 加载模型
predictor = MentalHealthPredictor()

# 进行预测
result = predictor.predict(
    screen_time=300,   # 屏幕时间(分钟)
    social_time=150,   # 社交媒体(分钟)
    sleep_hours=7.0,   # 睡眠(小时)
    age=25,            # 年龄
    gender="Male",     # 性别
    exercise=30        # 运动(分钟)
)

print(f"压力指数: {result['stress_level']}")
print(f"焦虑指数: {result['anxiety_level']}")
```

---

## 附录：完整代码结构

```
MindScreen/
├── model/                                   # 模型文件目录
│   ├── mental_health_model.pkl              # 训练好的模型
│   └── config.pkl                           # 模型配置信息
├── mental_health_social_media_dataset.csv   # 原始数据
├── train_models.py                          # 模型对比训练脚本
├── save_model.py                            # 模型保存脚本
├── predict.py                               # 交互式预测脚本 ⭐
├── ML_Tutorial.md                           # 本教程
└── README.md                                # 项目说明
```

### 文件说明

| 文件 | 用途 |
|------|------|
| `save_model.py` | 训练最优模型并保存到 model/ 目录 |
| `predict.py` | **产品入口** - 加载模型进行交互式预测 |
| `train_models.py` | 对比不同模型效果（开发用） |

---

*文档生成时间: 2025-12-05*
