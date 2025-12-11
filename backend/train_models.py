"""
MindScreen - 机器学习模型训练脚本
训练多种机器学习模型，选择最优模型进行保存
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import json
import os
import warnings
warnings.filterwarnings('ignore')

# 数据路径
DATA_PATH = '../digital_diet_mental_health.csv'
MODEL_DIR = '../models'

def load_and_preprocess_data():
    """加载并预处理数据"""
    df = pd.read_csv(DATA_PATH)
    
    # 性别编码
    le_gender = LabelEncoder()
    df['gender_encoded'] = le_gender.fit_transform(df['gender'])
    
    # 保存编码器
    joblib.dump(le_gender, os.path.join(MODEL_DIR, 'gender_encoder.pkl'))
    
    # 计算统计信息用于后续分析
    stats = {}
    columns_for_stats = [
        'daily_screen_time_hours', 'work_related_hours', 'entertainment_hours',
        'social_media_hours', 'sleep_duration_hours', 'sleep_quality',
        'weekly_anxiety_score', 'weekly_depression_score'
    ]
    
    for col in columns_for_stats:
        stats[col] = {
            'mean': float(df[col].mean()),
            'std': float(df[col].std()),
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'percentiles': {
                '10': float(df[col].quantile(0.1)),
                '25': float(df[col].quantile(0.25)),
                '50': float(df[col].quantile(0.5)),
                '75': float(df[col].quantile(0.75)),
                '90': float(df[col].quantile(0.9))
            }
        }
    
    # 保存统计信息
    with open(os.path.join(MODEL_DIR, 'data_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    return df, stats

def get_models():
    """获取所有待训练的模型"""
    return {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
        'LightGBM': LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
        'SVR': SVR(kernel='rbf', C=1.0),
        'KNN': KNeighborsRegressor(n_neighbors=5)
    }

def train_and_evaluate(X_train, X_test, y_train, y_test, target_name):
    """训练并评估所有模型，返回最佳模型"""
    models = get_models()
    results = []
    
    print(f"\n{'='*60}")
    print(f"训练目标: {target_name}")
    print(f"{'='*60}")
    
    best_model = None
    best_score = -float('inf')
    best_model_name = None
    
    for name, model in models.items():
        try:
            # 训练模型
            model.fit(X_train, y_train)
            
            # 预测
            y_pred = model.predict(X_test)
            
            # 评估
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # 交叉验证
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            cv_mean = cv_scores.mean()
            
            results.append({
                'model': name,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'cv_r2_mean': cv_mean
            })
            
            print(f"{name:20} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f} | CV-R²: {cv_mean:.4f}")
            
            # 选择最佳模型（基于交叉验证R²）
            if cv_mean > best_score:
                best_score = cv_mean
                best_model = model
                best_model_name = name
                
        except Exception as e:
            print(f"{name:20} | 训练失败: {str(e)}")
    
    print(f"\n最佳模型: {best_model_name} (CV-R²: {best_score:.4f})")
    
    return best_model, best_model_name, results

def calculate_feature_importance(model, feature_names, model_name):
    """计算特征重要性"""
    importance = {}
    
    if hasattr(model, 'feature_importances_'):
        imp = model.feature_importances_
        for i, name in enumerate(feature_names):
            importance[name] = float(imp[i])
    elif hasattr(model, 'coef_'):
        coef = np.abs(model.coef_)
        if len(coef.shape) == 1:
            for i, name in enumerate(feature_names):
                importance[name] = float(coef[i])
        else:
            for i, name in enumerate(feature_names):
                importance[name] = float(coef[0][i])
    else:
        # 对于没有feature_importances_的模型，使用排列重要性或均匀分布
        for name in feature_names:
            importance[name] = 1.0 / len(feature_names)
    
    # 归一化
    total = sum(importance.values())
    if total > 0:
        importance = {k: v/total for k, v in importance.items()}
    
    return importance

def main():
    """主训练流程"""
    print("="*60)
    print("MindScreen - 机器学习模型训练")
    print("="*60)
    
    # 确保模型目录存在
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # 加载数据
    print("\n1. 加载数据...")
    df, stats = load_and_preprocess_data()
    print(f"   数据集大小: {len(df)} 条记录")
    
    # 定义特征和目标
    # 焦虑评分特征 (3-8项 + 年龄 + 性别)
    anxiety_features = ['age', 'gender_encoded', 'daily_screen_time_hours', 'work_related_hours', 
                        'entertainment_hours', 'social_media_hours', 'sleep_duration_hours', 'sleep_quality']
    
    # 抑郁评分特征 (3-8项 + 年龄 + 性别)
    depression_features = ['age', 'gender_encoded', 'daily_screen_time_hours', 'work_related_hours',
                           'entertainment_hours', 'social_media_hours', 'sleep_duration_hours', 'sleep_quality']
    
    # 睡眠质量特征 (3-6项)
    sleep_features = ['daily_screen_time_hours', 'work_related_hours', 'entertainment_hours', 'social_media_hours']
    
    # 准备数据
    X_anxiety = df[anxiety_features].values
    X_depression = df[depression_features].values
    X_sleep = df[sleep_features].values
    
    y_anxiety = df['weekly_anxiety_score'].values
    y_depression = df['weekly_depression_score'].values
    y_sleep = df['sleep_quality'].values
    
    # 标准化
    scaler_anxiety = StandardScaler()
    scaler_depression = StandardScaler()
    scaler_sleep = StandardScaler()
    
    X_anxiety_scaled = scaler_anxiety.fit_transform(X_anxiety)
    X_depression_scaled = scaler_depression.fit_transform(X_depression)
    X_sleep_scaled = scaler_sleep.fit_transform(X_sleep)
    
    # 保存标准化器
    joblib.dump(scaler_anxiety, os.path.join(MODEL_DIR, 'scaler_anxiety.pkl'))
    joblib.dump(scaler_depression, os.path.join(MODEL_DIR, 'scaler_depression.pkl'))
    joblib.dump(scaler_sleep, os.path.join(MODEL_DIR, 'scaler_sleep.pkl'))
    
    # 划分数据集
    X_train_anx, X_test_anx, y_train_anx, y_test_anx = train_test_split(
        X_anxiety_scaled, y_anxiety, test_size=0.2, random_state=42)
    X_train_dep, X_test_dep, y_train_dep, y_test_dep = train_test_split(
        X_depression_scaled, y_depression, test_size=0.2, random_state=42)
    X_train_sleep, X_test_sleep, y_train_sleep, y_test_sleep = train_test_split(
        X_sleep_scaled, y_sleep, test_size=0.2, random_state=42)
    
    # 训练焦虑评分模型
    print("\n2. 训练焦虑评分预测模型...")
    anxiety_model, anxiety_model_name, anxiety_results = train_and_evaluate(
        X_train_anx, X_test_anx, y_train_anx, y_test_anx, "焦虑评分")
    
    # 训练抑郁评分模型
    print("\n3. 训练抑郁评分预测模型...")
    depression_model, depression_model_name, depression_results = train_and_evaluate(
        X_train_dep, X_test_dep, y_train_dep, y_test_dep, "抑郁评分")
    
    # 训练睡眠质量模型
    print("\n4. 训练睡眠质量预测模型...")
    sleep_model, sleep_model_name, sleep_results = train_and_evaluate(
        X_train_sleep, X_test_sleep, y_train_sleep, y_test_sleep, "睡眠质量")
    
    # 重新用全部数据训练最佳模型
    print("\n5. 使用全部数据重新训练最佳模型...")
    anxiety_model.fit(X_anxiety_scaled, y_anxiety)
    depression_model.fit(X_depression_scaled, y_depression)
    sleep_model.fit(X_sleep_scaled, y_sleep)
    
    # 保存模型
    print("\n6. 保存模型...")
    joblib.dump(anxiety_model, os.path.join(MODEL_DIR, 'anxiety_model.pkl'))
    joblib.dump(depression_model, os.path.join(MODEL_DIR, 'depression_model.pkl'))
    joblib.dump(sleep_model, os.path.join(MODEL_DIR, 'sleep_model.pkl'))
    
    # 计算特征重要性
    print("\n7. 计算特征重要性...")
    anxiety_importance = calculate_feature_importance(anxiety_model, anxiety_features, anxiety_model_name)
    depression_importance = calculate_feature_importance(depression_model, depression_features, depression_model_name)
    sleep_importance = calculate_feature_importance(sleep_model, sleep_features, sleep_model_name)
    
    # 保存模型信息
    model_info = {
        'anxiety': {
            'model_name': anxiety_model_name,
            'features': anxiety_features,
            'feature_importance': anxiety_importance,
            'training_results': anxiety_results
        },
        'depression': {
            'model_name': depression_model_name,
            'features': depression_features,
            'feature_importance': depression_importance,
            'training_results': depression_results
        },
        'sleep': {
            'model_name': sleep_model_name,
            'features': sleep_features,
            'feature_importance': sleep_importance,
            'training_results': sleep_results
        }
    }
    
    with open(os.path.join(MODEL_DIR, 'model_info.json'), 'w', encoding='utf-8') as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)
    print(f"\n模型保存位置: {MODEL_DIR}")
    print("保存的文件:")
    print("  - anxiety_model.pkl (焦虑评分预测模型)")
    print("  - depression_model.pkl (抑郁评分预测模型)")
    print("  - sleep_model.pkl (睡眠质量预测模型)")
    print("  - scaler_anxiety.pkl (焦虑模型标准化器)")
    print("  - scaler_depression.pkl (抑郁模型标准化器)")
    print("  - scaler_sleep.pkl (睡眠模型标准化器)")
    print("  - gender_encoder.pkl (性别编码器)")
    print("  - data_stats.json (数据统计信息)")
    print("  - model_info.json (模型信息)")
    
    return model_info

if __name__ == '__main__':
    main()
