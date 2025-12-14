"""PR 曲线与 SHAP 数值化分析脚本

功能：
- 重新训练与 `train_xgb_shap.py` 相同的 XGBoost 模型（确保本地已安装 xgboost/shap）
- 计算并保存 Precision-Recall 曲线与 ROC 曲线，输出 AUC/平均精度
- 为给定召回目标（0.8 和 0.9）推荐阈值（若可行）
- 基于 SHAP 计算每个 `digital_addiction_score` 的平均 SHAP，并找到接近 0 的交叉点，给出分段建议

运行示例（你之前运行训练的命令，使用正斜杠或转义反斜杠）：
C:/Users/yixiao/AppData/Local/Programs/Python/Python312/python.exe d:/HuaweiMoveData/Users/yixiao/Desktop/Study_Diary/MindScreen/models/pr_shap_analysis.py
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score, roc_curve

import xgboost as xgb
import shap

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT.parent / 'smmh.csv'
OUT_DIR = ROOT / 'eval_outputs'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_preprocess():
    df = pd.read_csv(DATA_PATH)
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
    df['q18'] = pd.to_numeric(df['q18'], errors='coerce')
    df = df.dropna(subset=['q18']).copy()
    df['target'] = df['q18'].apply(lambda x: 1 if x >= 4 else 0)

    df['age'] = pd.to_numeric(df['age'], errors='coerce')

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
    for c in ['relationship', 'occupation']:
        if c in df.columns:
            df[c] = df[c].fillna('Unknown').astype(str)
            le = LabelEncoder()
            df[c+'_enc'] = le.fit_transform(df[c])

    time_map = {
        'Less than an Hour': 0,
        'Between 1 and 2 hours': 1,
        'Between 2 and 3 hours': 2,
        'Between 3 and 4 hours': 3,
        'Between 4 and 5 hours': 4,
        'More than 5 hours': 5
    }
    df['avg_time_ord'] = df['avg_time'].map(time_map).fillna(-1)

    main_platforms = ['Discord','Facebook','Instagram','Pinterest','Reddit','Snapchat','TikTok','Twitter','YouTube']
    df['platforms'] = df['platforms'].fillna('')
    df['platform_count'] = df['platforms'].apply(lambda s: 0 if pd.isna(s) or s=='' else len([p for p in str(s).split(',') if p.strip()!='']))
    for p in main_platforms:
        df['plat_'+p] = df['platforms'].str.contains(p, na=False).astype(int)

    for q in ['q9','q10','q11','q12']:
        df[q] = pd.to_numeric(df[q], errors='coerce').fillna(0)
    df['digital_addiction_score'] = df[['q9','q10','q11','q12']].sum(axis=1)

    feature_cols = ['age','avg_time_ord','platform_count','digital_addiction_score'] + [f'plat_{p}' for p in main_platforms]
    df = pd.get_dummies(df, columns=['gender'], prefix='gender')
    X = df.reindex(columns=[c for c in feature_cols if c in df.columns] + [col for col in df.columns if col.startswith('gender_')]).fillna(-1)
    y = df['target'].astype(int)
    return df, X, y


def train_model(X_train, y_train):
    n_pos = (y_train==1).sum()
    n_neg = (y_train==0).sum()
    scale_pos_weight = n_neg / max(1, n_pos)
    model = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, use_label_encoder=False, eval_metric='logloss', random_state=42, scale_pos_weight=scale_pos_weight)
    model.fit(X_train, y_train)
    return model, scale_pos_weight


def recommend_thresholds(y_true, y_scores, recalls_target=[0.8,0.9]):
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    # thresholds returned are for precision/recall pairs excluding last point; align
    thr_df = pd.DataFrame({'precision':precision[:-1],'recall':recall[:-1],'threshold':np.append(thresholds, np.nan)[:-1]})
    recs = {}
    for r in recalls_target:
        cand = thr_df[thr_df['recall']>=r]
        if len(cand)==0:
            recs[r] = None
        else:
            # choose threshold with max precision among those achieving recall>=r
            best = cand.loc[cand['precision'].idxmax()]
            recs[r] = {'threshold': float(best['threshold']), 'precision': float(best['precision']), 'recall': float(best['recall'])}
    return ap, recs, precision, recall, thresholds


def main():
    df, X, y = load_preprocess()
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    model, spw = train_model(X_train, y_train)

    y_scores = model.predict_proba(X_test)[:,1]
    ap, recs, precision, recall, thresholds = recommend_thresholds(y_test, y_scores, [0.8,0.9])

    # save PR and ROC plots
    plt.figure()
    plt.plot(recall, precision, label=f'AP={ap:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend()
    plt.savefig(OUT_DIR / 'pr_curve.png')
    plt.close()

    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = roc_auc_score(y_test, y_scores)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC={roc_auc:.3f}')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC curve')
    plt.legend()
    plt.savefig(OUT_DIR / 'roc_curve.png')
    plt.close()

    # save recommendation
    rec_df = pd.DataFrame.from_dict({k:v for k,v in recs.items() if v is not None}, orient='index')
    rec_df.to_csv(OUT_DIR / 'threshold_recommendations.csv')

    # SHAP dependence numeric for digital_addiction_score
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    # shap_values shape (n_samples, n_features)
    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    if 'digital_addiction_score' in X_test.columns:
        das = X_test['digital_addiction_score'].values
        df_sh = pd.DataFrame({'digital_addiction_score': das, 'shap': sv[:, list(X_test.columns).index('digital_addiction_score')]})
        mean_shap_by_score = df_sh.groupby('digital_addiction_score')['shap'].mean().reset_index()
        mean_shap_by_score.to_csv(OUT_DIR / 'mean_shap_by_digital_addiction_score.csv', index=False)

        # find crossing where mean shap crosses zero (approx)
        crosses = mean_shap_by_score.sort_values('digital_addiction_score')
        cross_point = None
        for i in range(len(crosses)-1):
            a = crosses['shap'].iloc[i]
            b = crosses['shap'].iloc[i+1]
            if a<=0 and b>=0:
                cross_point = (crosses['digital_addiction_score'].iloc[i], crosses['digital_addiction_score'].iloc[i+1])
                break
        with open(OUT_DIR / 'das_segmentation_suggestion.txt','w',encoding='utf-8') as f:
            f.write(f'AP={ap:.4f}\nROC_AUC={roc_auc:.4f}\nscale_pos_weight={spw}\n')
            f.write('threshold_recommendations:\n')
            f.write(rec_df.to_csv(index=True))
            if cross_point is not None:
                f.write(f'Estimated digital_addiction_score crossing interval: {cross_point}\n')
                low, high = cross_point
                f.write(f'Suggested segmentation: low <={low}, mid {low+1}-{high-1}, high >={high}\n')
            else:
                f.write('No clear crossing found; consider quantile-based segmentation.\n')

    print('Analysis finished. Outputs saved to', OUT_DIR)


if __name__ == '__main__':
    main()
