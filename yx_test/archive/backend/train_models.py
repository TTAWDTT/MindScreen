"""
MindScreen - 机器学习模型训练脚本
训练多种机器学习模型，选择最优模型进行保存
"""

import pandas as pd
import numpy as np
"""
MindScreen - Mental Health & Technology Usage training script (classification focus)
Replaces previous regression pipelines with:
- Mental_Health_Status multi-class classifier
- Stress_Level multi-class classifier
Generates stats and saves pipelines/models under ../models.
"""
import os
import json
import warnings
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score, f1_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.base import clone
import joblib

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "mental_health_and_technology_usage_2024.csv"))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "models"))

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"数据文件未找到: {DATA_PATH}\n请将 mental_health_and_technology_usage_2024.csv 放置在 MindScreen 目录下，或修改 DATA_PATH。"
    )


def create_ohe():
    """Return dense OneHotEncoder; fallback for older sklearn."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_percentile_grid(series: pd.Series, step: float = 0.1) -> Dict[str, Any]:
    quantiles = np.arange(0, 100 + step, step)
    values = np.percentile(series, quantiles)
    return {
        "quantiles": [round(float(q), 1) for q in quantiles],
        "values": [float(v) for v in values],
    }


def compute_stats(df: pd.DataFrame, numeric_cols) -> Dict[str, Any]:
    stats = {}
    for col in numeric_cols:
        stats[col] = {
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "percentiles": {
                "10": float(df[col].quantile(0.1)),
                "25": float(df[col].quantile(0.25)),
                "50": float(df[col].quantile(0.5)),
                "75": float(df[col].quantile(0.75)),
                "90": float(df[col].quantile(0.9)),
            },
            "percentile_grid": build_percentile_grid(df[col]),
        }
    return stats


def build_preprocessor(numeric_cols, cat_cols):
    return ColumnTransformer(
        [
            ("num", StandardScaler(), numeric_cols),
            ("cat", create_ohe(), cat_cols),
        ],
        remainder="drop",
    )


def get_model_spaces():
    return {
        "LogReg": (LogisticRegression(max_iter=500, multi_class="auto"), {
            "model__C": [0.2, 1.0, 3.0],
            "model__penalty": ["l2"],
        }),
        "RandomForest": (RandomForestClassifier(random_state=42, n_jobs=-1, max_depth=10), {
            "model__n_estimators": [150, 300],
            "model__max_depth": [8, 12],
        }),
        "GradientBoosting": (GradientBoostingClassifier(random_state=42), {
            "model__n_estimators": [150, 250],
            "model__learning_rate": [0.05, 0.1],
        }),
    }


def train_classifier(X: pd.DataFrame, y: pd.Series, target_name: str, numeric_cols, cat_cols) -> Tuple[Pipeline, Dict[str, Any]]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    preprocessor = build_preprocessor(numeric_cols, cat_cols)
    spaces = get_model_spaces()

    best_score = -np.inf
    best_pipe = None
    best_name = None
    results = []

    y_majority = y_train.value_counts().idxmax()
    baseline_acc = accuracy_score(y_test, np.full_like(y_test, y_majority))

    for name, (model, grid) in spaces.items():
        pipe = Pipeline([
            ("prep", clone(preprocessor)),
            ("model", clone(model)),
        ])
        search = GridSearchCV(pipe, grid, cv=3, scoring="balanced_accuracy", n_jobs=1)
        search.fit(X_train, y_train)
        best_est = search.best_estimator_
        y_pred = best_est.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        clf_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        results.append({
            "model": name,
            "best_params": search.best_params_,
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "f1_macro": f1_macro,
            "report": clf_report,
        })

        if bal_acc > best_score:
            best_score = bal_acc
            best_pipe = best_est
            best_name = name

    best_metrics = [r for r in results if r["model"] == best_name][0]
    best_metrics["baseline_majority_accuracy"] = baseline_acc

    return best_pipe, best_metrics, results


def main():
    print("=" * 60)
    print("MindScreen - Mental Health & Technology Usage 模型训练")
    print("=" * 60)

    os.makedirs(MODEL_DIR, exist_ok=True)

    print("\n1. 读取数据...")
    df = pd.read_csv(DATA_PATH)
    print(f"   数据量: {len(df)}")

    targets = {
        "mental_health_status": "Mental_Health_Status",
        "stress_level": "Stress_Level",
    }

    numeric_cols = [
        "Age",
        "Technology_Usage_Hours",
        "Social_Media_Usage_Hours",
        "Gaming_Hours",
        "Screen_Time_Hours",
        "Sleep_Hours",
        "Physical_Activity_Hours",
    ]

    cat_cols = [
        "Gender",
        "Support_Systems_Access",
        "Work_Environment_Impact",
        "Online_Support_Usage",
    ]

    df = df.copy()
    eps = 1e-6
    df["social_ratio"] = df["Social_Media_Usage_Hours"] / (df["Technology_Usage_Hours"] + eps)
    df["gaming_ratio"] = df["Gaming_Hours"] / (df["Technology_Usage_Hours"] + eps)
    df["screen_per_sleep"] = df["Screen_Time_Hours"] / (df["Sleep_Hours"] + eps)
    df["low_activity_flag"] = (df["Physical_Activity_Hours"] < 3).astype(int)

    numeric_cols_extended = numeric_cols + ["social_ratio", "gaming_ratio", "screen_per_sleep", "low_activity_flag"]

    features = numeric_cols_extended + cat_cols
    df_features = df[features + list(targets.values())]

    stats = compute_stats(df_features, numeric_cols_extended)
    with open(os.path.join(MODEL_DIR, "data_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    model_info: Dict[str, Any] = {}

    print("\n2. 训练 Mental_Health_Status 分类模型...")
    X_mh = df_features[features]
    y_mh = df_features[targets["mental_health_status"]]
    mh_pipeline, mh_best, mh_all = train_classifier(X_mh, y_mh, "Mental_Health_Status", numeric_cols_extended, cat_cols)
    joblib.dump(mh_pipeline, os.path.join(MODEL_DIR, "mental_health_status_pipeline.pkl"))
    joblib.dump(mh_pipeline.named_steps["model"], os.path.join(MODEL_DIR, "mental_health_status_model.pkl"))
    joblib.dump(mh_pipeline.named_steps["prep"], os.path.join(MODEL_DIR, "mental_health_status_preprocessor.pkl"))

    model_info["mental_health_status"] = {
        "model_name": mh_best["model"],
        "metrics": mh_best,
        "all_results": mh_all,
        "features": features,
    }

    print("\n3. 训练 Stress_Level 分类模型...")
    X_st = df_features[features]
    y_st = df_features[targets["stress_level"]]
    st_pipeline, st_best, st_all = train_classifier(X_st, y_st, "Stress_Level", numeric_cols_extended, cat_cols)
    joblib.dump(st_pipeline, os.path.join(MODEL_DIR, "stress_level_pipeline.pkl"))
    joblib.dump(st_pipeline.named_steps["model"], os.path.join(MODEL_DIR, "stress_level_model.pkl"))
    joblib.dump(st_pipeline.named_steps["prep"], os.path.join(MODEL_DIR, "stress_level_preprocessor.pkl"))

    model_info["stress_level"] = {
        "model_name": st_best["model"],
        "metrics": st_best,
        "all_results": st_all,
        "features": features,
    }

    with open(os.path.join(MODEL_DIR, "model_info.json"), "w", encoding="utf-8") as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)

    print("\n训练完成。模型与统计已保存到 models/ 目录。")
    print("Mental_Health_Status 最佳模型:", mh_best["model"], "| balanced_acc:", f"{mh_best['balanced_accuracy']:.4f}")
    print("Stress_Level 最佳模型:", st_best["model"], "| balanced_acc:", f"{st_best['balanced_accuracy']:.4f}")


if __name__ == "__main__":
    main()
if __name__ == '__main__':
    main()
