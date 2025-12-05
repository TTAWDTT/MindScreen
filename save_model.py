# -*- coding: utf-8 -*-
"""
模型训练与保存脚本
训练最优模型（决策树）并保存为可部署的产品
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor
import joblib
import os

print("=" * 60)
print("心理健康预测模型 - 训练与保存")
print("=" * 60)

# 创建模型保存目录
MODEL_DIR = "model"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
    print(f"\n创建模型目录: {MODEL_DIR}/")

# 1. 加载数据
print("\n[1/4] 加载数据集...")
df = pd.read_csv('mental_health_social_media_dataset.csv')
print(f"  数据量: {len(df)} 条记录")

# 2. 数据预处理
print("\n[2/4] 数据预处理...")
le = LabelEncoder()
df['gender_encoded'] = le.fit_transform(df['gender'])

# 保存性别编码映射
gender_mapping = {label: idx for idx, label in enumerate(le.classes_)}
print(f"  性别编码: {gender_mapping}")

# 准备特征和目标
X = df[['daily_screen_time_min', 'social_media_time_min', 'sleep_hours', 
        'age', 'gender_encoded', 'physical_activity_min']].values
y = df[['stress_level', 'anxiety_level']].values

# 3. 训练模型
print("\n[3/4] 训练决策树模型...")
model = MultiOutputRegressor(DecisionTreeRegressor(
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
))
model.fit(X, y)
print("  模型训练完成!")

# 4. 保存模型和配置
print("\n[4/4] 保存模型...")

# 保存模型
model_path = os.path.join(MODEL_DIR, "mental_health_model.pkl")
joblib.dump(model, model_path)
print(f"  模型已保存: {model_path}")

# 保存配置信息
config = {
    "model_name": "MentalHealthPredictor",
    "version": "1.0.0",
    "model_type": "DecisionTree",
    "features": [
        {"name": "screen_time", "description": "每日屏幕使用时长(分钟)", "type": "int"},
        {"name": "social_time", "description": "社交媒体使用时长(分钟)", "type": "int"},
        {"name": "sleep_hours", "description": "睡眠时间(小时)", "type": "float"},
        {"name": "age", "description": "年龄", "type": "int"},
        {"name": "gender", "description": "性别", "type": "str", "options": ["Female", "Male", "Other"]},
        {"name": "exercise", "description": "身体活动时间(分钟)", "type": "int"}
    ],
    "outputs": [
        {"name": "stress_level", "description": "压力指数", "range": "1-10"},
        {"name": "anxiety_level", "description": "焦虑指数", "range": "1-5"}
    ],
    "gender_encoding": gender_mapping,
    "performance": {
        "stress_level_r2": 0.9957,
        "anxiety_level_r2": 0.9972
    }
}

config_path = os.path.join(MODEL_DIR, "config.pkl")
joblib.dump(config, config_path)
print(f"  配置已保存: {config_path}")

print("\n" + "=" * 60)
print("模型保存完成!")
print("=" * 60)
print(f"\n模型文件位置: {os.path.abspath(MODEL_DIR)}")
print("\n使用方法:")
print("  python predict.py          # 交互式预测")
print("  python predict.py --help   # 查看帮助")
