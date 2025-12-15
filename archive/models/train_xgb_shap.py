"""训练 XGBoost 并生成 SHAP 可解释性图。

遵循用户要求：
- 目标：Q18 二分类（4/5 -> 1，高风险；1-3 -> 0，低风险）
- 仅使用“行为特征”：age, gender, relationship, occupation, avg_time(序数), platform_count, 平台 one-hot, digital_addiction_score = Q9+Q10+Q11+Q12
- 丢弃后果类特征 Q13-Q20（含 Q18 目标）避免数据泄露
- 模型：XGBoost，使用 scale_pos_weight 处理不平衡

输出保存在 `models/eval_outputs/`。
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

import xgboost as xgb
import shap

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT.parent / 'smmh.csv'
OUT_DIR = ROOT / 'eval_outputs'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_preprocess():
    df = pd.read_csv(DATA_PATH)

    # 映射并简化列名
    col_map = {
        '1. What is your age?': 'age',
        '2. Gender': 'gender',
        '3. Relationship Status': 'relationship',
        '4. Occupation Status': 'occupation',
        '7. What social media platforms do you commonly use?': 'platforms',
        '8. What is the average time you spend on social media every day?': 'avg_time',
        '9. How often do you find yourself using Social media without a specific purpose?': 'q9',
        '10. How often do you get distracted by Social media when you are busy doing something?': 'q10',
        "11. Do you feel restless if you haven't used Social media in a while?": 'q11',
        '12. On a scale of 1 to 5, how easily distracted are you?': 'q12',
        '18. How often do you feel depressed or down?': 'q18'
    }

    df = df.rename(columns=col_map)

    # Target 二分类
    df['q18'] = pd.to_numeric(df['q18'], errors='coerce')
    df = df.dropna(subset=['q18'])
    df['target'] = df['q18'].apply(lambda x: 1 if x >= 4 else 0)

    # 行为特征构造
    # age
    df['age'] = pd.to_numeric(df['age'], errors='coerce')

    # gender 清洗为 Male/Female/Other
    def clean_gender(g):
        if pd.isna(g):
            return 'Other'
        s = str(g).strip().lower()
        if 'male' in s:
            return 'Male'
        if 'female' in s:
            return 'Female'
        return 'Other'

    df['gender'] = df['gender'].apply(clean_gender)

    # relationship, occupation -> label encode
    for c in ['relationship', 'occupation']:
        if c in df.columns:
            df[c] = df[c].fillna('Unknown').astype(str)
            le = LabelEncoder()
            df[c+'_enc'] = le.fit_transform(df[c])

    # avg_time 序数映射
    time_map = {
        'Less than an Hour': 0,
        'Between 1 and 2 hours': 1,
        'Between 2 and 3 hours': 2,
        'Between 3 and 4 hours': 3,
        'Between 4 and 5 hours': 4,
        'More than 5 hours': 5
    }
    df['avg_time_ord'] = df['avg_time'].map(time_map).fillna(-1)

    # platforms -> count + one-hot for main platforms
    main_platforms = ['Discord','Facebook','Instagram','Pinterest','Reddit','Snapchat','TikTok','Twitter','YouTube']
    df['platforms'] = df['platforms'].fillna('')
    df['platform_count'] = df['platforms'].apply(lambda s: 0 if pd.isna(s) or s=='' else len([p for p in str(s).split(',') if p.strip()!='']))
    for p in main_platforms:
        df['plat_'+p] = df['platforms'].str.contains(p, na=False).astype(int)

    # Digital Addiction Score = q9+q10+q11+q12
    for q in ['q9','q10','q11','q12']:
        df[q] = pd.to_numeric(df[q], errors='coerce').fillna(0)
    df['digital_addiction_score'] = df[['q9','q10','q11','q12']].sum(axis=1)

    # 仅保留行为特征
    feature_cols = ['age','gender','relationship_enc','occupation_enc','avg_time_ord','platform_count','digital_addiction_score'] + [f'plat_{p}' for p in main_platforms]

    # gender one-hot
    df = pd.get_dummies(df, columns=['gender'], prefix='gender')
    # ensure relationship_enc and occupation_enc exist
    for c in ['relationship_enc','occupation_enc']:
        if c not in df.columns:
            df[c] = 0

    # final X, y
    X = df.reindex(columns=[c for c in feature_cols if c in df.columns] + [col for col in df.columns if col.startswith('gender_')]).fillna(-1)
    y = df['target'].astype(int)

    return X, y


def train_and_explain(X, y):
    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # compute scale_pos_weight = negatives / positives on training
    n_pos = (y_train==1).sum()
    n_neg = (y_train==0).sum()
    scale_pos_weight = n_neg / max(1, n_pos)

    print('Train size:', len(y_train), 'Pos:', n_pos, 'Neg:', n_neg, 'scale_pos_weight:', scale_pos_weight)

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        scale_pos_weight=scale_pos_weight
    )

    model.fit(X_train, y_train)

    # predict on test
    y_pred = model.predict(X_test)
    rpt = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    rpt_df = pd.DataFrame(rpt).transpose()
    rpt_df.to_csv(OUT_DIR / 'xgb_classification_report.csv')

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Pred')
    plt.ylabel('True')
    plt.title('XGBoost Confusion Matrix')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'xgb_confusion_matrix.png')
    plt.close()

    # SHAP explainability
    explainer = shap.TreeExplainer(model)
    # convert to numpy for SHAP
    X_test_np = X_test.values
    shap_values = explainer.shap_values(X_test_np)

    # summary plot
    plt.figure()
    try:
        shap.summary_plot(shap_values, X_test, show=False)
    except Exception:
        # fallback: try passing numpy
        shap.summary_plot(shap_values, X_test_np, feature_names=X_test.columns, show=False)
    plt.savefig(OUT_DIR / 'shap_summary.png', bbox_inches='tight')
    plt.close()

    # dependence plots for digital_addiction_score and avg_time_ord if present
    for feat in ['digital_addiction_score','avg_time_ord']:
        if feat in X_test.columns:
            plt.figure()
            try:
                shap.dependence_plot(feat, shap_values, X_test, display_features=X_test, show=False)
            except Exception:
                shap.dependence_plot(feat, shap_values, X_test_np, feature_names=X_test.columns, show=False)
            plt.savefig(OUT_DIR / f'shap_dependence_{feat}.png', bbox_inches='tight')
            plt.close()

    # Save basic metrics summary
    summary = {
        'n_train': len(y_train),
        'n_test': len(y_test),
        'train_pos': int(n_pos),
        'train_neg': int(n_neg),
        'scale_pos_weight': float(scale_pos_weight)
    }
    pd.Series(summary).to_csv(OUT_DIR / 'xgb_training_summary.csv')

    # Analyze direction: compute mean SHAP for features
    # shap_values for binary classifier: shape (n_samples, n_features)
    try:
        mean_shap = np.mean(shap_values, axis=0)
    except Exception:
        mean_shap = np.mean(shap_values[1], axis=0) if isinstance(shap_values, list) else np.mean(shap_values, axis=0)

    feat_effect = pd.Series(mean_shap, index=X_test.columns).sort_values(key=abs, ascending=False)
    feat_effect.to_csv(OUT_DIR / 'feature_shap_mean_effect.csv')

    return model, rpt_df, cm, feat_effect


def main():
    X, y = load_preprocess()
    model, rpt_df, cm, feat_effect = train_and_explain(X, y)
    print('Saved outputs to', OUT_DIR)

    # print short analysis for requested features
    for f in ['digital_addiction_score','avg_time_ord']:
        if f in feat_effect.index:
            val = feat_effect.loc[f]
            direction = '正向（增加风险）' if val > 0 else '负向（降低风险）' if val < 0 else '无明显影响'
            print(f'{f}: mean SHAP={val:.4f} -> {direction}')


if __name__ == '__main__':
    main()
