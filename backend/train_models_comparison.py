"""
MindScreen - 多模型对比训练脚本
对比多种机器学习算法，选择最佳模型并记录训练结果

支持的模型：
- Logistic Regression
- Random Forest
- XGBoost
- LightGBM (可选)
- SVM (可选)

输出：
- 最佳模型文件 (.pkl)
- 详细训练报告 (model_comparison_report.json)
"""

import json
import os
import time
from typing import Dict, Any, Tuple, List
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, classification_report,
    roc_auc_score, confusion_matrix, precision_score, recall_score
)
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("Warning: XGBoost not installed, skipping XGB models")

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("Warning: LightGBM not installed, skipping LGBM models")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "smmh.csv"))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "models"))

# 列重命名映射
COL_RENAME = {
    'Timestamp': 'timestamp',
    '1. What is your age?': 'age',
    '2. Gender': 'gender',
    '3. Relationship Status': 'relationship',
    '4. Occupation Status': 'occupation',
    '5. What type of organizations are you affiliated with?': 'affiliate_organization',
    '6. Do you use social media?': 'social_media_use',
    '7. What social media platforms do you commonly use?': 'platforms',
    '8. What is the average time you spend on social media every day?': 'avg_time_per_day',
    '9. How often do you find yourself using Social media without a specific purpose?': 'without_purpose',
    '10. How often do you get distracted by Social media when you are busy doing something?': 'distracted',
    "11. Do you feel restless if you haven't used Social media in a while?": 'restless',
    '12. On a scale of 1 to 5, how easily distracted are you?': 'distracted_ease',
    '13. On a scale of 1 to 5, how much are you bothered by worries?': 'worries',
    '14. Do you find it difficult to concentrate on things?': 'concentration',
    '15. On a scale of 1-5, how often do you compare yourself to other successful people through the use of social media?': 'compare_to_others',
    '16. Following the previous question, how do you feel about these comparisons, generally speaking?': 'compare_feelings',
    '17. How often do you look to seek validation from features of social media?': 'validation',
    '18. How often do you feel depressed or down?': 'depressed',
    '19. On a scale of 1 to 5, how frequently does your interest in daily activities fluctuate?': 'daily_activity_flux',
    '20. On a scale of 1 to 5, how often do you face issues regarding sleep?': 'sleeping_issues'
}

LIKERT_COLS = [
    'without_purpose', 'distracted', 'restless', 'distracted_ease', 'worries',
    'concentration', 'compare_to_others', 'compare_feelings', 'validation',
    'depressed', 'daily_activity_flux', 'sleeping_issues'
]

AVG_TIME_ORDER = [
    'Less than an Hour',
    'Between 1 and 2 hours',
    'Between 2 and 3 hours',
    'Between 3 and 4 hours',
    'Between 4 and 5 hours',
    'More than 5 hours'
]


def compute_percentile(val, series: pd.Series):
    if series is None or series.empty or val is None:
        return None
    q = np.nanpercentile(series, np.arange(0, 101))
    pct = np.interp(val, q, np.arange(0, 101))
    return round(float(np.clip(pct, 0, 100)), 1)


def load_and_clean() -> Tuple[pd.DataFrame, List[str]]:
    """加载并清洗数据"""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"数据文件未找到: {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns=COL_RENAME)
    
    # 填充缺失值
    if df['affiliate_organization'].isnull().any():
        mode_val = df['affiliate_organization'].value_counts().index[0]
        df['affiliate_organization'] = df['affiliate_organization'].fillna(mode_val)
    
    # 标准化性别
    df['gender'] = df['gender'].apply(lambda x: x if x in ['Male', 'Female'] else 'other')
    
    # 处理平台数据
    platform_dummies = df['platforms'].fillna('').apply(
        lambda x: [p.strip() for p in str(x).split(',') if p.strip()]
    )
    all_platforms = sorted(set(p for lst in platform_dummies for p in lst))
    
    for p in all_platforms:
        df[f'plat_{p}'] = platform_dummies.apply(lambda lst: 1 if p in lst else 0)
    df['platform_count'] = df[[f'plat_{p}' for p in all_platforms]].sum(axis=1)
    
    # 数字成瘾评分
    q_cols = ['without_purpose', 'distracted', 'restless', 'distracted_ease']
    for c in q_cols:
        if c not in df.columns:
            df[c] = 0
    df['digital_addiction_score'] = df[q_cols].sum(axis=1)
    
    # 时间使用编码
    df['avg_time_per_day'] = pd.Categorical(df['avg_time_per_day'], categories=AVG_TIME_ORDER, ordered=True)
    df['avg_time_ord'] = df['avg_time_per_day'].cat.codes.replace(-1, pd.NA).astype('float')
    df['avg_time_ord'] = df['avg_time_ord'].fillna(df['avg_time_ord'].median())
    
    # 计算风险标签
    df['impact_sum'] = df[LIKERT_COLS].sum(axis=1)
    df['risk'] = np.where(df['impact_sum'] >= 37, 'higher', 'lower')
    
    # 编码分类变量
    df['relationship_enc'] = pd.factorize(df['relationship'].fillna('unknown'))[0]
    df['occupation_enc'] = pd.factorize(df['occupation'].fillna('unknown'))[0]
    
    # 性别独热编码
    gender_dummies = pd.get_dummies(df['gender'].fillna('other'), prefix='gender')
    for c in gender_dummies.columns:
        df[c] = gender_dummies[c]
    
    return df, all_platforms


def get_model_candidates() -> Dict[str, Tuple[Any, Dict]]:
    """获取所有候选模型及其超参数网格"""
    candidates = {
        'logistic_regression': (
            LogisticRegression(max_iter=500, random_state=42),
            {
                'model__C': [0.1, 1.0, 10.0],
                'model__solver': ['lbfgs', 'liblinear']
            }
        ),
        'random_forest': (
            RandomForestClassifier(random_state=42),
            {
                'model__n_estimators': [100, 200],
                'model__max_depth': [6, 12, None],
                'model__min_samples_leaf': [1, 2]
            }
        ),
        'gradient_boosting': (
            GradientBoostingClassifier(random_state=42),
            {
                'model__n_estimators': [100, 150],
                'model__max_depth': [3, 5],
                'model__learning_rate': [0.05, 0.1]
            }
        ),
        'svm': (
            SVC(probability=True, random_state=42),
            {
                'model__C': [0.1, 1.0, 10.0],
                'model__kernel': ['rbf', 'linear']
            }
        )
    }
    
    if HAS_XGB:
        candidates['xgboost'] = (
            XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, verbosity=0),
            {
                'model__n_estimators': [100, 200],
                'model__max_depth': [3, 6],
                'model__learning_rate': [0.05, 0.1]
            }
        )
    
    if HAS_LGBM:
        candidates['lightgbm'] = (
            LGBMClassifier(random_state=42, verbose=-1),
            {
                'model__n_estimators': [100, 200],
                'model__max_depth': [3, 6],
                'model__learning_rate': [0.05, 0.1]
            }
        )
    
    return candidates


def train_and_evaluate(
    X_train, X_test, y_train, y_test,
    preprocessor, model_name, estimator, param_grid,
    is_binary=True
) -> Dict[str, Any]:
    """训练单个模型并评估"""
    start_time = time.time()
    
    pipe = Pipeline([
        ('prep', preprocessor),
        ('model', estimator),
    ])
    
    # 网格搜索
    search = GridSearchCV(
        pipe, param_grid, cv=5,
        scoring='balanced_accuracy',
        n_jobs=-1
    )
    search.fit(X_train, y_train)
    
    best_model = search.best_estimator_
    train_time = time.time() - start_time
    
    # 预测
    y_pred = best_model.predict(X_test)
    y_proba = None
    if hasattr(best_model, 'predict_proba'):
        try:
            y_proba = best_model.predict_proba(X_test)
        except:
            pass
    
    # 计算指标
    metrics = {
        'model_name': model_name,
        'best_params': search.best_params_,
        'train_time_seconds': round(train_time, 2),
        'cv_score': round(search.best_score_, 4),
        'balanced_accuracy': round(balanced_accuracy_score(y_test, y_pred), 4),
        'f1_macro': round(f1_score(y_test, y_pred, average='macro'), 4),
        'precision_macro': round(precision_score(y_test, y_pred, average='macro', zero_division=0), 4),
        'recall_macro': round(recall_score(y_test, y_pred, average='macro', zero_division=0), 4),
    }
    
    # ROC-AUC (仅二分类)
    if is_binary and y_proba is not None:
        try:
            metrics['roc_auc'] = round(roc_auc_score(y_test, y_proba[:, 1]), 4)
        except:
            pass
    
    # 分类报告
    metrics['classification_report'] = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    # 交叉验证分数
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='balanced_accuracy')
    metrics['cv_scores_mean'] = round(cv_scores.mean(), 4)
    metrics['cv_scores_std'] = round(cv_scores.std(), 4)
    
    return metrics, best_model


def select_best_model(results: List[Dict]) -> Tuple[str, Dict]:
    """根据多个指标选择最佳模型"""
    # 综合评分：balanced_accuracy * 0.4 + f1_macro * 0.3 + cv_score * 0.3
    for r in results:
        r['composite_score'] = (
            r['balanced_accuracy'] * 0.4 +
            r['f1_macro'] * 0.3 +
            r['cv_score'] * 0.3
        )
    
    best = max(results, key=lambda x: x['composite_score'])
    return best['model_name'], best


def main():
    print("=" * 60)
    print("MindScreen - 多模型对比训练")
    print("=" * 60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 加载数据
    print("📊 加载数据...")
    df, platforms = load_and_clean()
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # 计算基准值
    baseline_map = {}
    for col in LIKERT_COLS:
        series = df[col]
        mean_val = float(series.mean()) if not series.empty else None
        baseline_map[col] = {
            'baseline_value': mean_val,
            'baseline_percentile': compute_percentile(mean_val, series) if mean_val is not None else None
        }
    
    # 特征列
    plat_cols = [f'plat_{p}' for p in platforms]
    gender_cols = [c for c in df.columns if c.startswith('gender_')]
    feature_cols = ['age', 'relationship_enc', 'occupation_enc', 'avg_time_ord',
                    'platform_count', 'digital_addiction_score'] + plat_cols + gender_cols
    
    # 预处理器
    num_cols = ['age', 'platform_count', 'digital_addiction_score', 'avg_time_ord',
                'relationship_enc', 'occupation_enc']
    cat_cols = plat_cols + gender_cols
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ])
    
    # 获取模型候选
    model_candidates = get_model_candidates()
    
    # ========== 风险模型训练 ==========
    print("\n🎯 训练风险预测模型 (二分类)...")
    print("-" * 40)
    
    X_risk = df[feature_cols]
    y_risk_raw = df['risk']
    risk_mapping = {'higher': 1, 'lower': 0}
    y_risk = y_risk_raw.map(risk_mapping)
    
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_risk, y_risk, test_size=0.2, random_state=42, stratify=y_risk
    )
    
    risk_results = []
    risk_models = {}
    
    for name, (estimator, params) in model_candidates.items():
        print(f"  训练 {name}...")
        metrics, model = train_and_evaluate(
            X_train_r, X_test_r, y_train_r, y_test_r,
            preprocessor, name, estimator, params, is_binary=True
        )
        risk_results.append(metrics)
        risk_models[name] = model
        print(f"    ✓ 平衡准确率: {metrics['balanced_accuracy']:.4f}, F1: {metrics['f1_macro']:.4f}")
    
    best_risk_name, best_risk_metrics = select_best_model(risk_results)
    print(f"\n  🏆 最佳风险模型: {best_risk_name}")
    print(f"     综合得分: {best_risk_metrics['composite_score']:.4f}")
    
    # 保存最佳风险模型
    risk_model = risk_models[best_risk_name]
    risk_model.classes_ = np.array(['lower', 'higher'])
    joblib.dump(risk_model, os.path.join(MODEL_DIR, 'smmh_risk_pipeline_v2.pkl'))
    
    # ========== 抑郁模型训练 ==========
    print("\n🧠 训练抑郁等级模型 (多分类 1-5)...")
    print("-" * 40)
    
    X_dep = df[feature_cols]
    y_dep_raw = df['depressed']
    dep_mapping = {lab: i for i, lab in enumerate(sorted(y_dep_raw.unique()))}
    y_dep = y_dep_raw.map(dep_mapping)
    
    X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
        X_dep, y_dep, test_size=0.2, random_state=42, stratify=y_dep
    )
    
    dep_results = []
    dep_models = {}
    
    for name, (estimator, params) in model_candidates.items():
        print(f"  训练 {name}...")
        metrics, model = train_and_evaluate(
            X_train_d, X_test_d, y_train_d, y_test_d,
            preprocessor, name, estimator, params, is_binary=False
        )
        dep_results.append(metrics)
        dep_models[name] = model
        print(f"    ✓ 平衡准确率: {metrics['balanced_accuracy']:.4f}, F1: {metrics['f1_macro']:.4f}")
    
    best_dep_name, best_dep_metrics = select_best_model(dep_results)
    print(f"\n  🏆 最佳抑郁模型: {best_dep_name}")
    print(f"     综合得分: {best_dep_metrics['composite_score']:.4f}")
    
    # 保存最佳抑郁模型
    dep_model = dep_models[best_dep_name]
    dep_model.classes_ = np.array(sorted(y_dep_raw.unique()))
    joblib.dump(dep_model, os.path.join(MODEL_DIR, 'smmh_depressed_pipeline.pkl'))
    
    # ========== 计算综合评分分布 ==========
    print("\n📈 计算综合评分分布...")
    X_all = df[feature_cols]
    
    risk_proba = risk_model.predict_proba(X_all)
    higher_idx = list(risk_model.classes_).index('higher')
    risk_prob_higher = risk_proba[:, higher_idx]
    
    dep_pred = dep_model.predict(X_all)
    dep_numeric = np.array([float(p) if isinstance(p, (int, float)) else 3.0 for p in dep_pred])
    
    composite_scores = 0.5 * risk_prob_higher + 0.5 * (dep_numeric / 5.0)
    composite_percentiles = np.percentile(composite_scores, np.arange(0, 101))
    
    # ========== 保存模型信息和训练报告 ==========
    model_info = {
        'dataset': 'smmh.csv',
        'n_rows': len(df),
        'features': feature_cols,
        'platforms': platforms,
        'likert_baseline': baseline_map,
        'risk': {
            'model': best_risk_name,
            'params': best_risk_metrics['best_params'],
            'balanced_accuracy': best_risk_metrics['balanced_accuracy'],
            'f1_macro': best_risk_metrics['f1_macro'],
            'roc_auc': best_risk_metrics.get('roc_auc'),
            'report': best_risk_metrics['classification_report']
        },
        'depressed': {
            'model': best_dep_name,
            'params': best_dep_metrics['best_params'],
            'balanced_accuracy': best_dep_metrics['balanced_accuracy'],
            'f1_macro': best_dep_metrics['f1_macro'],
            'report': best_dep_metrics['classification_report']
        },
        'composite_score_distribution': {
            'percentiles': composite_percentiles.tolist(),
            'mean': float(np.mean(composite_scores)),
            'std': float(np.std(composite_scores)),
            'min': float(np.min(composite_scores)),
            'max': float(np.max(composite_scores)),
            'formula': 'composite = 0.5 * P(risk=higher) + 0.5 * (depressed_level / 5)'
        }
    }
    
    with open(os.path.join(MODEL_DIR, 'smmh_model_info.json'), 'w', encoding='utf-8') as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)
    
    # 保存详细训练报告
    training_report = {
        'training_date': datetime.now().isoformat(),
        'dataset_info': {
            'path': DATA_PATH,
            'rows': len(df),
            'features': len(feature_cols)
        },
        'risk_model_comparison': risk_results,
        'depressed_model_comparison': dep_results,
        'best_models': {
            'risk': {
                'name': best_risk_name,
                'reason': f"综合评分最高 ({best_risk_metrics['composite_score']:.4f})，平衡准确率 {best_risk_metrics['balanced_accuracy']:.4f}，F1 {best_risk_metrics['f1_macro']:.4f}"
            },
            'depressed': {
                'name': best_dep_name,
                'reason': f"综合评分最高 ({best_dep_metrics['composite_score']:.4f})，平衡准确率 {best_dep_metrics['balanced_accuracy']:.4f}，F1 {best_dep_metrics['f1_macro']:.4f}"
            }
        },
        'model_selection_criteria': {
            'formula': 'composite_score = balanced_accuracy * 0.4 + f1_macro * 0.3 + cv_score * 0.3',
            'reasoning': '综合考虑测试集准确率、F1分数和交叉验证稳定性，选择泛化能力最强的模型'
        }
    }
    
    with open(os.path.join(MODEL_DIR, 'model_comparison_report.json'), 'w', encoding='utf-8') as f:
        json.dump(training_report, f, ensure_ascii=False, indent=2)
    
    # 打印总结
    print("\n" + "=" * 60)
    print("✅ 训练完成!")
    print("=" * 60)
    print(f"\n📁 输出文件:")
    print(f"   - {os.path.join(MODEL_DIR, 'smmh_risk_pipeline_v2.pkl')}")
    print(f"   - {os.path.join(MODEL_DIR, 'smmh_depressed_pipeline.pkl')}")
    print(f"   - {os.path.join(MODEL_DIR, 'smmh_model_info.json')}")
    print(f"   - {os.path.join(MODEL_DIR, 'model_comparison_report.json')}")
    print(f"\n📊 模型性能:")
    print(f"   风险模型 ({best_risk_name}): 准确率 {best_risk_metrics['balanced_accuracy']:.2%}")
    print(f"   抑郁模型 ({best_dep_name}): 准确率 {best_dep_metrics['balanced_accuracy']:.2%}")
    print(f"\n📈 综合评分: 均值={np.mean(composite_scores):.3f}, 标准差={np.std(composite_scores):.3f}")


if __name__ == '__main__':
    main()
