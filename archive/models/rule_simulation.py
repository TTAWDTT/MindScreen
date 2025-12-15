"""生产规则仿真：基于 OOF（分层交叉）模型预测与 Digital Addiction Score，评估告警量与精度/召回。

规则（已实现）：
- Strong alert: `digital_addiction_score >= 15` OR `model_score >= 0.38266`
- Early-warning: `digital_addiction_score in {13,14}` OR (0.26493 <= `model_score` < 0.38266)
- Low-priority: `digital_addiction_score <= 12` AND `model_score < 0.26493`

输出：
- `models/eval_outputs/rule_simulation_summary.csv`（规则命中计数与指标）
- `models/eval_outputs/rule_simulation_details.csv`（每个样本的规则标签与真实目标、model_score、das）

运行：
C:/Users/yixiao/AppData/Local/Programs/Python/Python312/python.exe d:/HuaweiMoveData/Users/yixiao/Desktop/Study_Diary/MindScreen/models/rule_simulation.py
"""
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score
import xgboost as xgb

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT.parent / 'smmh.csv'
OUT = ROOT / 'eval_outputs'
OUT.mkdir(exist_ok=True)


def preprocess():
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


def oof_predict_proba(X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    probs = np.zeros(len(y))
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        n_pos = (y_train==1).sum()
        n_neg = (y_train==0).sum()
        spw = float(n_neg) / max(1, int(n_pos))
        model = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, use_label_encoder=False, eval_metric='logloss', random_state=42, scale_pos_weight=spw)
        model.fit(X_train, y_train)
        probs[test_idx] = model.predict_proba(X_test)[:,1]
    return probs


def apply_rules(df, probs, thr_low=0.26493003964424133, thr_mid=0.3826602101325989):
    df = df.copy()
    df['model_score'] = probs
    # rules
    df['strong_alert'] = ((df['digital_addiction_score'] >= 15) | (df['model_score'] >= thr_mid)).astype(int)
    df['early_warning'] = (((df['digital_addiction_score'].isin([13,14])) | ((df['model_score'] >= thr_low) & (df['model_score'] < thr_mid)))).astype(int)
    df['low_priority'] = ((df['digital_addiction_score'] <= 12) & (df['model_score'] < thr_low)).astype(int)
    return df


def apply_candidate_rules(df, probs, thr_low=0.26493003964424133, thr_mid=0.3826602101325989):
    """添加候选规则 A/B/C 并返回带标记的 DataFrame"""
    df = df.copy()
    df['model_score'] = probs
    # Candidate A: Strong = (DAS >= 16) OR (model_score >= thr_mid)
    df['A_strong'] = ((df['digital_addiction_score'] >= 16) | (df['model_score'] >= thr_mid)).astype(int)
    # Candidate B: Strong = (DAS >= 15) AND (model_score >= thr_mid)
    df['B_strong'] = ((df['digital_addiction_score'] >= 15) & (df['model_score'] >= thr_mid)).astype(int)
    # Candidate C: Early-warning = (DAS in {13,14}) AND (avg_time_ord >= 3)
    df['C_early'] = ((df['digital_addiction_score'].isin([13,14])) & (df['avg_time_ord'] >= 3)).astype(int)
    return df


def summarize(df):
    total = len(df)
    summaries = []
    for col in ['strong_alert','early_warning','low_priority']:
        preds = df[col]
        prec = precision_score(df['target'], preds, zero_division=0)
        rec = recall_score(df['target'], preds, zero_division=0)
        f1 = f1_score(df['target'], preds, zero_division=0)
        count = int(preds.sum())
        per_1000 = count/total*1000
        summaries.append({'rule': col, 'count': count, 'per_1000': per_1000, 'precision': prec, 'recall': rec, 'f1': f1, 'total': total})
    df_sum = pd.DataFrame(summaries)
    df_sum.to_csv(OUT / 'rule_simulation_summary.csv', index=False)
    df.to_csv(OUT / 'rule_simulation_details.csv', index=False)
    return df_sum


def summarize_candidates(df):
    total = len(df)
    cand_cols = ['A_strong','B_strong','C_early']
    summaries = []
    for col in cand_cols:
        preds = df[col]
        prec = precision_score(df['target'], preds, zero_division=0)
        rec = recall_score(df['target'], preds, zero_division=0)
        f1 = f1_score(df['target'], preds, zero_division=0)
        count = int(preds.sum())
        per_1000 = count/total*1000
        summaries.append({'rule': col, 'count': count, 'per_1000': per_1000, 'precision': prec, 'recall': rec, 'f1': f1, 'total': total})
    df_sum = pd.DataFrame(summaries)
    df_sum.to_csv(OUT / 'rule_simulation_candidates_summary.csv', index=False)
    # also save details with candidate flags
    df.to_csv(OUT / 'rule_simulation_with_candidates_details.csv', index=False)
    return df_sum


def main():
    df, X, y = preprocess()
    probs = oof_predict_proba(X, y, n_splits=5)
    df_rules = apply_rules(df, probs)
    summary = summarize(df_rules)
    # apply candidate rules A/B/C and summarize
    df_cand = apply_candidate_rules(df, probs)
    cand_summary = summarize_candidates(df_cand)
    print('Simulation saved to', OUT)
    print(summary)
    print('Candidate rules summary:')
    print(cand_summary)


if __name__ == '__main__':
    main()
