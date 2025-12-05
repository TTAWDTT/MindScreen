# -*- coding: utf-8 -*-
"""
心理健康预测模型训练脚本
使用决策树、支持向量机、多层感知机预测压力指数和焦虑指数
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 加载数据
print("=" * 60)
print("加载数据集...")
df = pd.read_csv('mental_health_social_media_dataset.csv')
print(f"数据集大小: {df.shape[0]} 行, {df.shape[1]} 列")
print(f"\n数据集列名: {list(df.columns)}")

# 特征选择
# 输入特征: 屏幕使用时长, 社交媒体使用时长, 睡眠时间, 年龄, 性别, 身体活动时间
# 输出目标: 压力指数, 焦虑指数

feature_columns = [
    'daily_screen_time_min',   # 屏幕使用时长
    'social_media_time_min',   # 社交媒体使用时长
    'sleep_hours',             # 睡眠时间
    'age',                     # 年龄
    'gender',                  # 性别
    'physical_activity_min'    # 身体活动时间
]

target_columns = [
    'stress_level',   # 压力指数
    'anxiety_level'   # 焦虑指数
]

print(f"\n输入特征: {feature_columns}")
print(f"预测目标: {target_columns}")

# 数据预处理
print("\n" + "=" * 60)
print("数据预处理...")

# 处理性别编码
le = LabelEncoder()
df['gender_encoded'] = le.fit_transform(df['gender'])
print(f"性别编码映射: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# 准备特征矩阵
X = df[['daily_screen_time_min', 'social_media_time_min', 'sleep_hours', 
        'age', 'gender_encoded', 'physical_activity_min']].values
y = df[target_columns].values

print(f"\n特征矩阵形状: {X.shape}")
print(f"目标矩阵形状: {y.shape}")

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n训练集大小: {X_train.shape[0]}")
print(f"测试集大小: {X_test.shape[0]}")

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 定义评估函数
def evaluate_model(model, X_test, y_test, model_name):
    """评估模型性能"""
    y_pred = model.predict(X_test)
    
    results = {}
    for i, target in enumerate(target_columns):
        mse = mean_squared_error(y_test[:, i], y_pred[:, i])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
        r2 = r2_score(y_test[:, i], y_pred[:, i])
        
        results[target] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
    
    return results, y_pred

# 存储所有模型结果
all_results = {}

# ============================================================
# 1. 决策树模型
# ============================================================
print("\n" + "=" * 60)
print("1. 训练决策树模型 (Decision Tree)")
print("=" * 60)

dt_model = DecisionTreeRegressor(
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
dt_model = MultiOutputRegressor(dt_model)
dt_model.fit(X_train, y_train)  # 决策树不需要标准化

dt_results, dt_pred = evaluate_model(dt_model, X_test, y_test, "决策树")
all_results['决策树'] = dt_results

for target, metrics in dt_results.items():
    print(f"\n{target}:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")

# ============================================================
# 2. 支持向量机模型
# ============================================================
print("\n" + "=" * 60)
print("2. 训练支持向量机模型 (SVM)")
print("=" * 60)

svm_model = SVR(
    kernel='rbf',
    C=1.0,
    epsilon=0.1
)
svm_model = MultiOutputRegressor(svm_model)
svm_model.fit(X_train_scaled, y_train)  # SVM需要标准化

svm_results, svm_pred = evaluate_model(svm_model, X_test_scaled, y_test, "SVM")
all_results['支持向量机'] = svm_results

for target, metrics in svm_results.items():
    print(f"\n{target}:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")

# ============================================================
# 3. 多层感知机模型
# ============================================================
print("\n" + "=" * 60)
print("3. 训练多层感知机模型 (MLP)")
print("=" * 60)

mlp_model = MLPRegressor(
    hidden_layer_sizes=(64, 32, 16),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1
)
mlp_model.fit(X_train_scaled, y_train)  # MLP需要标准化

mlp_results, mlp_pred = evaluate_model(mlp_model, X_test_scaled, y_test, "MLP")
all_results['多层感知机'] = mlp_results

for target, metrics in mlp_results.items():
    print(f"\n{target}:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")

# ============================================================
# 模型对比总结
# ============================================================
print("\n" + "=" * 60)
print("模型对比总结")
print("=" * 60)

# 创建对比表格
print("\n【压力指数 (stress_level) 预测效果对比】")
print("-" * 55)
print(f"{'模型':<15} {'MSE':<10} {'RMSE':<10} {'MAE':<10} {'R2':<10}")
print("-" * 55)
for model_name, results in all_results.items():
    metrics = results['stress_level']
    print(f"{model_name:<15} {metrics['MSE']:<10.4f} {metrics['RMSE']:<10.4f} {metrics['MAE']:<10.4f} {metrics['R2']:<10.4f}")

print("\n【焦虑指数 (anxiety_level) 预测效果对比】")
print("-" * 55)
print(f"{'模型':<15} {'MSE':<10} {'RMSE':<10} {'MAE':<10} {'R2':<10}")
print("-" * 55)
for model_name, results in all_results.items():
    metrics = results['anxiety_level']
    print(f"{model_name:<15} {metrics['MSE']:<10.4f} {metrics['RMSE']:<10.4f} {metrics['MAE']:<10.4f} {metrics['R2']:<10.4f}")

# 找出最佳模型
print("\n【最佳模型选择】")
print("-" * 55)

for target in target_columns:
    best_r2 = -float('inf')
    best_model = ""
    for model_name, results in all_results.items():
        if results[target]['R2'] > best_r2:
            best_r2 = results[target]['R2']
            best_model = model_name
    print(f"{target}: 最佳模型是 【{best_model}】, R2 = {best_r2:.4f}")

# ============================================================
# 示例预测
# ============================================================
print("\n" + "=" * 60)
print("示例预测")
print("=" * 60)

# 创建示例数据
example_data = np.array([[
    300,   # 屏幕使用时长(分钟)
    150,   # 社交媒体时长(分钟)
    7.0,   # 睡眠时间(小时)
    25,    # 年龄
    1,     # 性别(0=Female, 1=Male, 2=Other)
    30     # 身体活动时间(分钟)
]])

example_scaled = scaler.transform(example_data)

print("\n输入特征:")
print(f"  屏幕使用时长: 300 分钟")
print(f"  社交媒体时长: 150 分钟")
print(f"  睡眠时间: 7.0 小时")
print(f"  年龄: 25 岁")
print(f"  性别: Male")
print(f"  身体活动时间: 30 分钟")

print("\n各模型预测结果:")
print("-" * 40)
dt_example_pred = dt_model.predict(example_data)[0]
print(f"决策树:     压力指数={dt_example_pred[0]:.2f}, 焦虑指数={dt_example_pred[1]:.2f}")

svm_example_pred = svm_model.predict(example_scaled)[0]
print(f"支持向量机: 压力指数={svm_example_pred[0]:.2f}, 焦虑指数={svm_example_pred[1]:.2f}")

mlp_example_pred = mlp_model.predict(example_scaled)[0]
print(f"多层感知机: 压力指数={mlp_example_pred[0]:.2f}, 焦虑指数={mlp_example_pred[1]:.2f}")

print("\n" + "=" * 60)
print("训练完成!")
print("=" * 60)
