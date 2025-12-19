"""
MindScreen - 多模型对比训练脚本 (优化版)
对比多种机器学习算法，选择最佳模型并记录训练结果

支持的模型：
- Logistic Regression
- Random Forest
- XGBoost
- LightGBM (可选)
- SVM (可选)

优化特性：
- 异步进度显示，训练过程无卡顿
- 清晰的结构化日志输出
- 实时进度条与耗时统计

输出：
- 最佳模型文件 (.pkl)
- 详细训练报告 (model_comparison_report.json)
"""

import json
import os
import sys
import time
import warnings
import threading
from typing import Dict, Any, Tuple, List, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, classification_report,
    roc_auc_score, precision_score, recall_score
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

# ========== 全局配置 ==========
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# 可选依赖检测
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

# ========== 路径配置 ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "smmh.csv"))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "models"))


# ========== 美化输出工具类 ==========
class ProgressPrinter:
    """训练进度打印器，提供流畅的终端输出体验"""
    
    # 状态图标
    ICONS = {
        'start': '🚀', 'data': '📊', 'train': '🎯', 'model': '🧠',
        'done': '✅', 'best': '🏆', 'file': '📁', 'chart': '📈',
        'time': '⏱️', 'warn': '⚠️', 'info': 'ℹ️'
    }
    
    def __init__(self):
        self.start_time = time.time()
        self._spinner_active = False
        self._spinner_thread: Optional[threading.Thread] = None
    
    def header(self, title: str, char: str = "=", width: int = 65) -> None:
        """打印标题头"""
        print(f"\n{char * width}")
        print(f"  {title}")
        print(f"{char * width}")
    
    def section(self, title: str, icon: str = 'info') -> None:
        """打印章节标题"""
        ico = self.ICONS.get(icon, icon)
        print(f"\n{ico} {title}")
        print("-" * 50)
    
    def step(self, msg: str, indent: int = 2) -> None:
        """打印步骤信息"""
        print(f"{' ' * indent}→ {msg}")
    
    def result(self, name: str, metrics: Dict[str, Any], indent: int = 4) -> None:
        """打印单个模型训练结果"""
        acc = metrics.get('balanced_accuracy', 0)
        f1 = metrics.get('f1_macro', 0)
        t = metrics.get('train_time_seconds', 0)
        
        # 性能等级颜色标记
        grade = "★★★" if acc >= 0.85 else "★★☆" if acc >= 0.75 else "★☆☆"
        print(f"{' ' * indent}✓ {name:20s} | Acc: {acc:.4f} | F1: {f1:.4f} | {t:5.1f}s | {grade}")
    
    def best_model(self, task: str, name: str, score: float) -> None:
        """打印最佳模型信息"""
        print(f"\n  {self.ICONS['best']} 最佳{task}模型: {name}")
        print(f"     综合得分: {score:.4f}")
    
    def summary(self, risk_name: str, risk_acc: float, 
                dep_name: str, dep_acc: float, composite_mean: float, composite_std: float) -> None:
        """打印训练总结"""
        elapsed = time.time() - self.start_time
        print(f"\n{self.ICONS['chart']} 模型性能概览:")
        print(f"   ├─ 风险模型 ({risk_name}): {risk_acc:.2%}")
        print(f"   └─ 抑郁模型 ({dep_name}): {dep_acc:.2%}")
        print(f"\n{self.ICONS['chart']} 综合评分分布: μ={composite_mean:.3f}, σ={composite_std:.3f}")
        print(f"\n{self.ICONS['time']} 总耗时: {elapsed:.1f}s")
    
    def file_output(self, files: List[str]) -> None:
        """打印输出文件列表"""
        print(f"\n{self.ICONS['file']} 输出文件:")
        for f in files:
            print(f"   └─ {f}")
    
    def done(self) -> None:
        """打印完成信息"""
        self.header(f"{self.ICONS['done']} 训练完成!", char="═")


class TrainingSpinner:
    """异步训练进度指示器，确保训练过程无卡顿感"""
    
    SPINNER_CHARS = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
    
    def __init__(self, message: str = "训练中"):
        self.message = message
        self._active = False
        self._thread: Optional[threading.Thread] = None
    
    def _spin(self) -> None:
        """后台旋转动画"""
        idx = 0
        while self._active:
            char = self.SPINNER_CHARS[idx % len(self.SPINNER_CHARS)]
            sys.stdout.write(f"\r  {char} {self.message}...")
            sys.stdout.flush()
            time.sleep(0.1)
            idx += 1
        # 清除spinner行
        sys.stdout.write("\r" + " " * 50 + "\r")
        sys.stdout.flush()
    
    def start(self) -> None:
        """启动spinner"""
        self._active = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
    
    def stop(self) -> None:
        """停止spinner"""
        self._active = False
        if self._thread:
            self._thread.join(timeout=0.5)


# ========== 列重命名映射 ==========
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

# ========== 数据列配置 ==========
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


# ========== 工具函数 ==========
def compute_percentile(val: float, series: pd.Series) -> Optional[float]:
    """计算值在序列中的百分位数"""
    if series is None or series.empty or val is None:
        return None
    q = np.nanpercentile(series, np.arange(0, 101))
    pct = np.interp(val, q, np.arange(0, 101))
    return round(float(np.clip(pct, 0, 100)), 1)


def load_and_clean() -> Tuple[pd.DataFrame, List[str]]:
    """
    加载并清洗数据集
    
    Returns:
        df: 清洗后的DataFrame
        platforms: 所有社交媒体平台列表
    """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"数据文件未找到: {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns=COL_RENAME)
    
    # 填充缺失的组织类型
    if df['affiliate_organization'].isnull().any():
        mode_val = df['affiliate_organization'].value_counts().index[0]
        df['affiliate_organization'] = df['affiliate_organization'].fillna(mode_val)
    
    # 性别标准化
    df['gender'] = df['gender'].apply(lambda x: x if x in ['Male', 'Female'] else 'other')
    
    # 平台特征工程
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
    
    # 时间特征编码
    df['avg_time_per_day'] = pd.Categorical(df['avg_time_per_day'], categories=AVG_TIME_ORDER, ordered=True)
    df['avg_time_ord'] = df['avg_time_per_day'].cat.codes.replace(-1, pd.NA).astype('float')
    df['avg_time_ord'] = df['avg_time_ord'].fillna(df['avg_time_ord'].median())
    
    # 风险标签
    df['impact_sum'] = df[LIKERT_COLS].sum(axis=1)
    df['risk'] = np.where(df['impact_sum'] >= 37, 'higher', 'lower')
    
    # 类别编码
    df['relationship_enc'] = pd.factorize(df['relationship'].fillna('unknown'))[0]
    df['occupation_enc'] = pd.factorize(df['occupation'].fillna('unknown'))[0]
    
    # 性别独热编码
    gender_dummies = pd.get_dummies(df['gender'].fillna('other'), prefix='gender')
    for c in gender_dummies.columns:
        df[c] = gender_dummies[c]
    
    return df, all_platforms


def get_model_candidates() -> Dict[str, Tuple[Any, Dict]]:
    """
    获取所有候选模型及其超参数网格
    
    Returns:
        候选模型字典: {模型名: (估计器实例, 参数网格)}
    """
    candidates = {
        'LogisticRegression': (
            LogisticRegression(max_iter=500, random_state=42),
            {'model__C': [0.1, 1.0, 10.0], 'model__solver': ['lbfgs', 'liblinear']}
        ),
        'RandomForest': (
            RandomForestClassifier(random_state=42, n_jobs=-1),
            {'model__n_estimators': [100, 200], 'model__max_depth': [6, 12, None], 'model__min_samples_leaf': [1, 2]}
        ),
        'GradientBoosting': (
            GradientBoostingClassifier(random_state=42),
            {'model__n_estimators': [100, 150], 'model__max_depth': [3, 5], 'model__learning_rate': [0.05, 0.1]}
        ),
        'SVM': (
            SVC(probability=True, random_state=42),
            {'model__C': [0.1, 1.0, 10.0], 'model__kernel': ['rbf', 'linear']}
        )
    }
    
    if HAS_XGB:
        candidates['XGBoost'] = (
            XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, verbosity=0, n_jobs=-1),
            {'model__n_estimators': [100, 200], 'model__max_depth': [3, 6], 'model__learning_rate': [0.05, 0.1]}
        )
    
    if HAS_LGBM:
        candidates['LightGBM'] = (
            LGBMClassifier(random_state=42, verbose=-1, n_jobs=-1),
            {'model__n_estimators': [100, 200], 'model__max_depth': [3, 6], 'model__learning_rate': [0.05, 0.1]}
        )
    
    return candidates


def train_single_model(
    X_train: pd.DataFrame, X_test: pd.DataFrame, 
    y_train: pd.Series, y_test: pd.Series,
    preprocessor: ColumnTransformer, 
    model_name: str, estimator: Any, param_grid: Dict,
    is_binary: bool = True
) -> Tuple[Dict[str, Any], Any]:
    """
    训练单个模型并评估性能
    
    Args:
        X_train, X_test: 训练/测试特征
        y_train, y_test: 训练/测试标签  
        preprocessor: 数据预处理器
        model_name: 模型名称
        estimator: sklearn估计器
        param_grid: 超参数网格
        is_binary: 是否为二分类任务
        
    Returns:
        metrics: 评估指标字典
        best_model: 训练后的最佳模型
    """
    start_time = time.time()
    
    # 构建Pipeline
    pipe = Pipeline([
        ('prep', preprocessor), 
        ('model', estimator)
    ])
    
    # 网格搜索 (使用并行加速)
    search = GridSearchCV(
        pipe, param_grid, 
        cv=5, 
        scoring='balanced_accuracy', 
        n_jobs=-1,  # 并行加速
        error_score='raise'
    )
    search.fit(X_train, y_train)
    
    best_model = search.best_estimator_
    train_time = time.time() - start_time
    
    # 预测与评估
    y_pred = best_model.predict(X_test)
    y_proba = None
    if hasattr(best_model, "predict_proba"):
        try:
            y_proba = best_model.predict_proba(X_test)
        except Exception:
            pass

    # 构建指标字典
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

    # 二分类额外计算ROC-AUC
    if is_binary and y_proba is not None:
        try:
            metrics['roc_auc'] = round(roc_auc_score(y_test, y_proba[:, 1]), 4)
        except Exception:
            pass
    
    metrics['classification_report'] = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    return metrics, best_model


def train_all_models(
    X_train: pd.DataFrame, X_test: pd.DataFrame,
    y_train: pd.Series, y_test: pd.Series,
    preprocessor: ColumnTransformer,
    model_candidates: Dict[str, Tuple[Any, Dict]],
    is_binary: bool,
    printer: ProgressPrinter
) -> Tuple[List[Dict], Dict[str, Any]]:
    """
    训练所有候选模型 (带进度显示)
    
    Args:
        X_train, X_test, y_train, y_test: 训练测试数据
        preprocessor: 预处理器
        model_candidates: 候选模型字典
        is_binary: 是否二分类
        printer: 进度打印器
        
    Returns:
        results: 所有模型的评估结果列表
        models: 训练好的模型字典
    """
    results = []
    models = {}
    total = len(model_candidates)
    
    for idx, (name, (estimator, params)) in enumerate(model_candidates.items(), 1):
        printer.step(f"[{idx}/{total}] 训练 {name}...")
        
        # 使用spinner显示训练进度
        spinner = TrainingSpinner(f"训练 {name}")
        spinner.start()
        
        try:
            metrics, model = train_single_model(
                X_train, X_test, y_train, y_test,
                preprocessor, name, estimator, params, is_binary
            )
            results.append(metrics)
            models[name] = model
        finally:
            spinner.stop()
        
        # 打印结果
        printer.result(name, metrics)
    
    return results, models


def select_best_model(results: List[Dict]) -> Tuple[str, Dict]:
    """
    根据综合评分选择最佳模型
    
    评分公式: composite = balanced_accuracy * 0.4 + f1_macro * 0.3 + cv_score * 0.3
    
    Args:
        results: 所有模型评估结果
        
    Returns:
        best_name: 最佳模型名称
        best_metrics: 最佳模型指标
    """
    for r in results:
        r['composite_score'] = (
            r.get('balanced_accuracy', 0) * 0.4 +
            r.get('f1_macro', 0) * 0.3 +
            r.get('cv_score', 0) * 0.3
        )
    
    best = max(results, key=lambda x: x.get('composite_score', 0))
    return best.get('model_name', 'unknown'), best


def save_training_report(
    model_info: Dict, training_report: Dict, 
    output_dir: str
) -> List[str]:
    """
    保存模型信息和训练报告
    
    Args:
        model_info: 模型元信息
        training_report: 详细训练报告
        output_dir: 输出目录
        
    Returns:
        saved_files: 保存的文件路径列表
    """
    files = []
    
    info_path = os.path.join(output_dir, 'smmh_model_info.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)
    files.append(info_path)
    
    report_path = os.path.join(output_dir, 'model_comparison_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(training_report, f, ensure_ascii=False, indent=2)
    files.append(report_path)
    
    return files


# ========== 主函数 ==========
def main():
    """主训练流程"""
    printer = ProgressPrinter()
    
    # ===== 初始化 =====
    printer.header("MindScreen - 多模型对比训练", char="═")
    print(f"  开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 检测可用模型
    available = ['LogisticRegression', 'RandomForest', 'GradientBoosting', 'SVM']
    if HAS_XGB:
        available.append('XGBoost')
    if HAS_LGBM:
        available.append('LightGBM')
    print(f"  可用模型: {', '.join(available)}")
    
    # ===== 数据加载 =====
    printer.section("加载与预处理数据", icon='data')
    df, platforms = load_and_clean()
    os.makedirs(MODEL_DIR, exist_ok=True)
    printer.step(f"数据集: {len(df)} 行, {len(platforms)} 个平台")
    
    # 基线统计
    baseline_map = {}
    for col in LIKERT_COLS:
        series = df[col]
        mean_val = float(series.mean()) if not series.empty else None
        baseline_map[col] = {
            'baseline_value': mean_val,
            'baseline_percentile': compute_percentile(mean_val, series) if mean_val is not None else None
        }
    
    # 特征定义
    plat_cols = [f'plat_{p}' for p in platforms]
    gender_cols = [c for c in df.columns if c.startswith('gender_')]
    feature_cols = ['age', 'relationship_enc', 'occupation_enc', 'avg_time_ord',
                    'platform_count', 'digital_addiction_score'] + plat_cols + gender_cols
    
    num_cols = ['age', 'platform_count', 'digital_addiction_score', 'avg_time_ord',
                'relationship_enc', 'occupation_enc']
    cat_cols = plat_cols + gender_cols
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ])
    
    printer.step(f"特征数: {len(feature_cols)} (数值: {len(num_cols)}, 类别: {len(cat_cols)})")
    
    model_candidates = get_model_candidates()
    
    # ===== 风险模型训练 =====
    printer.section("训练风险预测模型 (二分类)", icon='train')
    
    X_risk = df[feature_cols]
    y_risk_raw = df['risk']
    risk_mapping = {'higher': 1, 'lower': 0}
    y_risk = y_risk_raw.map(risk_mapping)
    
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_risk, y_risk, test_size=0.2, random_state=42, stratify=y_risk
    )
    
    risk_results, risk_models = train_all_models(
        X_train_r, X_test_r, y_train_r, y_test_r,
        preprocessor, model_candidates, is_binary=True, printer=printer
    )
    
    best_risk_name, best_risk_metrics = select_best_model(risk_results)
    printer.best_model("风险", best_risk_name, best_risk_metrics.get('composite_score', 0))
    
    risk_model = risk_models[best_risk_name]
    risk_model_path = os.path.join(MODEL_DIR, 'smmh_risk_pipeline_v2.pkl')
    joblib.dump(risk_model, risk_model_path)
    
    # ===== 抑郁模型训练 =====
    printer.section("训练抑郁等级模型 (多分类 1-5)", icon='model')
    
    X_dep = df[feature_cols]
    y_dep_raw = df['depressed']
    dep_mapping = {lab: i for i, lab in enumerate(sorted(y_dep_raw.unique()))}
    y_dep = y_dep_raw.map(dep_mapping)
    
    X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
        X_dep, y_dep, test_size=0.2, random_state=42, stratify=y_dep
    )
    
    dep_results, dep_models = train_all_models(
        X_train_d, X_test_d, y_train_d, y_test_d,
        preprocessor, model_candidates, is_binary=False, printer=printer
    )
    
    best_dep_name, best_dep_metrics = select_best_model(dep_results)
    printer.best_model("抑郁", best_dep_name, best_dep_metrics.get('composite_score', 0))
    
    dep_model = dep_models[best_dep_name]
    dep_model_path = os.path.join(MODEL_DIR, 'smmh_depressed_pipeline.pkl')
    joblib.dump(dep_model, dep_model_path)
    
    # ===== 综合评分计算 =====
    printer.section("计算综合评分分布", icon='chart')
    X_all = df[feature_cols]
    
    risk_proba = risk_model.predict_proba(X_all)
    higher_idx = risk_mapping['higher']
    risk_prob_higher = risk_proba[:, higher_idx]
    
    dep_pred = dep_model.predict(X_all)
    dep_numeric = dep_pred.astype(float) + 1
    
    composite_scores = 0.5 * risk_prob_higher + 0.5 * (dep_numeric / 5.0)
    composite_percentiles = np.percentile(composite_scores, np.arange(0, 101))
    
    printer.step(f"评分范围: [{np.min(composite_scores):.3f}, {np.max(composite_scores):.3f}]")
    printer.step(f"评分分布: μ={np.mean(composite_scores):.3f}, σ={np.std(composite_scores):.3f}")
    
    # ===== 保存报告 =====
    model_info = {
        'dataset': 'smmh.csv',
        'n_rows': len(df),
        'features': feature_cols,
        'platforms': platforms,
        'likert_baseline': baseline_map,
        'risk': {
            'model': best_risk_name,
            'params': best_risk_metrics.get('best_params'),
            'balanced_accuracy': best_risk_metrics.get('balanced_accuracy'),
            'f1_macro': best_risk_metrics.get('f1_macro'),
            'roc_auc': best_risk_metrics.get('roc_auc'),
            'report': best_risk_metrics.get('classification_report')
        },
        'depressed': {
            'model': best_dep_name,
            'params': best_dep_metrics.get('best_params'),
            'balanced_accuracy': best_dep_metrics.get('balanced_accuracy'),
            'f1_macro': best_dep_metrics.get('f1_macro'),
            'report': best_dep_metrics.get('classification_report')
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
                'reason': f"综合评分最高 ({best_risk_metrics.get('composite_score', 0):.4f})"
            },
            'depressed': {
                'name': best_dep_name,
                'reason': f"综合评分最高 ({best_dep_metrics.get('composite_score', 0):.4f})"
            }
        },
        'model_selection_criteria': {
            'formula': 'composite_score = balanced_accuracy * 0.4 + f1_macro * 0.3 + cv_score * 0.3',
            'reasoning': '综合考虑测试集准确率、F1分数和交叉验证稳定性'
        }
    }
    
    report_files = save_training_report(model_info, training_report, MODEL_DIR)
    
    # ===== 输出总结 =====
    all_output_files = [risk_model_path, dep_model_path] + report_files
    printer.file_output(all_output_files)
    
    printer.summary(
        best_risk_name, best_risk_metrics.get('balanced_accuracy', 0),
        best_dep_name, best_dep_metrics.get('balanced_accuracy', 0),
        np.mean(composite_scores), np.std(composite_scores)
    )
    
    printer.done()


if __name__ == '__main__':
    main()
