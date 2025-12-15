"""评估脚本：对 `smmh.csv` 运行交叉验证，比较模型并输出混淆矩阵与报告。

假设与说明:
- 将第 18 列（关于是否感到抑郁）作为多分类目标 `depressed`（1-5）。
- 将第 9 列（无目的使用社交媒体频率）二值化为 `risk`（>=4 视为 higher，否则 lower）。

输出:
- 在 `models/eval_outputs/` 保存报告（csv）、混淆矩阵图片和 sklearn 的 classification_report。
"""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT.parent / 'smmh.csv'
OUT_DIR = ROOT / 'eval_outputs'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_and_preprocess():
    df = pd.read_csv(DATA_PATH)
    # 显示列名以便调试
    print('Columns:', list(df.columns))

    # 选择或构造特征
    # age -> numeric
    df = df.rename(columns={
        '1. What is your age?': 'age',
        '2. Gender': 'gender',
        '3. Relationship Status': 'relationship',
        '8. What is the average time you spend on social media every day?': 'avg_time_per_day',
        '7. What social media platforms do you commonly use?': 'platforms',
        '9. How often do you find yourself using Social media without a specific purpose?': 'without_purpose',
        '18. How often do you feel depressed or down?': 'depressed'
    })

    # keep columns that exist
    cols = ['age', 'gender', 'relationship', 'avg_time_per_day', 'platforms', 'without_purpose', 'depressed']
    df = df[[c for c in cols if c in df.columns]].copy()

    # target depressed (1-5) as int
    df['depressed'] = pd.to_numeric(df['depressed'], errors='coerce')

    # derive risk binary: without_purpose >=4 -> higher
    df['without_purpose'] = pd.to_numeric(df['without_purpose'], errors='coerce')
    df['risk'] = df['without_purpose'].apply(lambda x: 'higher' if x >= 4 else 'lower')

    # platforms one-hot from string list
    df['platforms'] = df['platforms'].fillna('')
    all_platforms = ['Discord','Facebook','Instagram','Pinterest','Reddit','Snapchat','TikTok','Twitter','YouTube']
    for p in all_platforms:
        df[p] = df['platforms'].str.contains(p, na=False).astype(int)

    # avg_time_per_day mapping
    time_map = {
        'Less than an Hour': 0,
        'Between 1 and 2 hours': 1,
        'Between 2 and 3 hours': 2,
        'Between 3 and 4 hours': 3,
        'Between 4 and 5 hours': 4,
        'More than 5 hours': 5
    }
    if 'avg_time_per_day' in df.columns:
        df['avg_time_per_day'] = df['avg_time_per_day'].map(time_map).fillna(-1)

    # simple encodings for categorical vars
    for c in ['gender', 'relationship']:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
            le = LabelEncoder()
            df[c] = le.fit_transform(df[c].fillna('NA'))

    # age numeric
    if 'age' in df.columns:
        df['age'] = pd.to_numeric(df['age'], errors='coerce')

    # drop rows with missing targets
    df = df.dropna(subset=['depressed', 'risk'])

    return df, all_platforms


def evaluate_task(X, y, labels, task_name, models):
    print(f'Running evaluation for {task_name} on {len(y)} samples')
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {}
    for name, clf in models.items():
        y_pred = cross_val_predict(clf, X, y, cv=skf)
        rpt = classification_report(y, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y, y_pred, labels=labels)
        results[name] = {'report': rpt, 'cm': cm}

        # save classification report
        rpt_df = pd.DataFrame(rpt).transpose()
        rpt_df.to_csv(OUT_DIR / f'{task_name}_{name}_classification_report.csv')

        # plot confusion matrix
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.ylabel('True')
        plt.xlabel('Pred')
        plt.title(f'{task_name} - {name} Confusion Matrix')
        plt.tight_layout()
        plt.savefig(OUT_DIR / f'{task_name}_{name}_confusion_matrix.png')
        plt.close()

    return results


def main():
    df, all_platforms = load_and_preprocess()

    feature_cols = ['age', 'gender', 'relationship', 'avg_time_per_day'] + all_platforms
    X = df[feature_cols].fillna(-1).values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Task depressed (multiclass)
    y_dep = df['depressed'].astype(int)
    dep_labels = sorted(y_dep.unique())

    # Task risk (binary)
    y_risk = df['risk']
    risk_labels = ['lower', 'higher']

    models = {
        'logreg': LogisticRegression(max_iter=200, solver='liblinear', multi_class='auto'),
        'rf': RandomForestClassifier(n_estimators=100, random_state=42)
    }

    # evaluate depressed
    dep_results = evaluate_task(X, y_dep, labels=dep_labels, task_name='depressed', models=models)

    # evaluate risk
    risk_results = evaluate_task(X, y_risk, labels=risk_labels, task_name='risk', models=models)

    # 汇总基本指标到 csv
    summary_rows = []
    for task_name, res in [('depressed', dep_results), ('risk', risk_results)]:
        for mname, info in res.items():
            rpt = info['report']
            if 'macro avg' in rpt:
                summary_rows.append({
                    'task': task_name,
                    'model': mname,
                    'precision_macro': rpt['macro avg']['precision'],
                    'recall_macro': rpt['macro avg']['recall'],
                    'f1_macro': rpt['macro avg']['f1-score']
                })

    pd.DataFrame(summary_rows).to_csv(OUT_DIR / 'evaluation_summary.csv', index=False)
    print('Evaluation finished. Outputs saved to', OUT_DIR)


if __name__ == '__main__':
    main()
