"""Run stratified 5-fold OOF CV for risk (Q18 binary) and a quick depressed->binary experiment.

Outputs saved under models/eval_outputs/oof_cv/
"""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, roc_auc_score
import xgboost as xgb
import json

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT.parent / 'smmh.csv'
OUT_DIR = ROOT / 'eval_outputs' / 'oof_cv'
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

    # target binary for risk
    df['q18'] = pd.to_numeric(df['q18'], errors='coerce')
    df = df.dropna(subset=['q18'])
    df['target'] = df['q18'].apply(lambda x: 1 if x >= 4 else 0)

    # features
    df['age'] = pd.to_numeric(df['age'], errors='coerce').fillna(df['age'].median())

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
            df[c+'_enc'] = pd.Categorical(df[c]).codes
        else:
            df[c+'_enc'] = 0

    time_map = {
        'Less than an Hour': 0,
        'Between 1 and 2 hours': 1,
        'Between 2 and 3 hours': 2,
        'Between 3 and 4 hours': 3,
        'Between 4 and 5 hours': 4,
        'More than 5 hours': 5
    }
    df['avg_time_ord'] = df['avg_time'].map(time_map).fillna(2)

    main_platforms = ['Discord','Facebook','Instagram','Pinterest','Reddit','Snapchat','TikTok','Twitter','YouTube']
    df['platforms'] = df.get('platforms','').fillna('')
    df['platform_count'] = df['platforms'].apply(lambda s: 0 if pd.isna(s) or s=='' else len([p for p in str(s).split(',') if p.strip()!='']))
    for p in main_platforms:
        df['plat_'+p] = df['platforms'].str.contains(p, na=False).astype(int)

    for q in ['q9','q10','q11','q12']:
        df[q] = pd.to_numeric(df.get(q,0), errors='coerce').fillna(0)
    df['digital_addiction_score'] = df[['q9','q10','q11','q12']].sum(axis=1)

    # one-hot gender
    df = pd.get_dummies(df, columns=['gender'], prefix='gender')

    feature_cols = ['age','relationship_enc','occupation_enc','avg_time_ord','platform_count','digital_addiction_score'] + [f'plat_{p}' for p in main_platforms] + [c for c in df.columns if c.startswith('gender_')]
    X = df.reindex(columns=feature_cols).fillna(0)
    y = df['target'].astype(int)
    return X, y


def run_oof_xgb(X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_pred = np.zeros(len(X), dtype=int)
    oof_prob = np.zeros(len(X), dtype=float)
    oof_true = y.values

    fold_reports = {}
    feat_shap_means = []

    for i, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        n_pos = int((y_train==1).sum())
        n_neg = int((y_train==0).sum())
        spw = n_neg / max(1, n_pos)

        model = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, use_label_encoder=False, eval_metric='logloss', random_state=42, scale_pos_weight=spw)
        model.fit(X_train, y_train)

        prob = model.predict_proba(X_test)[:,1]
        pred = (prob >= 0.5).astype(int)

        oof_pred[test_idx] = pred
        oof_prob[test_idx] = prob

        rpt = classification_report(y_test, pred, output_dict=True, zero_division=0)
        fold_reports[f'fold_{i}'] = rpt

        # SHAP mean for fold
        try:
            import shap
            expl = shap.TreeExplainer(model)
            sv = expl.shap_values(X_test)
            if isinstance(sv, list):
                mv = np.mean(sv[1], axis=0)
            else:
                mv = np.mean(sv, axis=0)
            feat_shap_means.append(mv)
        except Exception:
            feat_shap_means.append(np.zeros(X.shape[1]))

    # overall
    precision, recall, f1, _ = precision_recall_fscore_support(oof_true, oof_pred, average='binary', zero_division=0)
    acc = accuracy_score(oof_true, oof_pred)
    try:
        auc = roc_auc_score(oof_true, oof_prob)
    except Exception:
        auc = None

    overall = {
        'accuracy': float(acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'roc_auc': float(auc) if auc is not None else None
    }

    # save OOF table
    oof_df = pd.DataFrame({'true': oof_true, 'pred': oof_pred, 'prob': oof_prob})
    oof_df.to_csv(OUT_DIR / 'oof_predictions.csv', index=False)
    with open(OUT_DIR / 'oof_fold_reports.json','w',encoding='utf-8') as f:
        json.dump(fold_reports, f, ensure_ascii=False, indent=2)
    with open(OUT_DIR / 'oof_overall.json','w',encoding='utf-8') as f:
        json.dump(overall, f, ensure_ascii=False, indent=2)

    # feature-level SHAP mean average across folds
    feat_means = np.mean(np.vstack(feat_shap_means), axis=0)
    feat_series = pd.Series(feat_means, index=X.columns)
    feat_series.to_csv(OUT_DIR / 'oof_feature_shap_mean.csv')

    return overall, fold_reports, oof_df


def depressed_binary_experiment():
    # Convert depressed Likert to binary (>=4 high)
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns={
        '18. How often do you feel depressed or down?': 'q18'
    })
    df['q18'] = pd.to_numeric(df['q18'], errors='coerce')
    df = df.dropna(subset=['q18'])
    df['dep_bin'] = df['q18'].apply(lambda x: 1 if x >= 4 else 0)

    # Use same feature set as risk preprocessing
    X, _ = load_preprocess()
    y = df['dep_bin'].loc[X.index].astype(int)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_predict

    clf = RandomForestClassifier(random_state=42, n_estimators=200)
    preds = cross_val_predict(clf, X, y, cv=5)
    acc = accuracy_score(y, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y, preds, average='binary', zero_division=0)
    rpt = classification_report(y, preds, output_dict=True, zero_division=0)
    with open(OUT_DIR / 'depressed_binary_report.json','w',encoding='utf-8') as f:
        json.dump({'accuracy':acc,'precision':prec,'recall':rec,'f1':f1,'report':rpt}, f, ensure_ascii=False, indent=2)
    return {'accuracy':acc,'precision':prec,'recall':rec,'f1':f1}


if __name__ == '__main__':
    X, y = load_preprocess()
    overall, folds, oof = run_oof_xgb(X, y, n_splits=5)
    print('OOF overall:', overall)
    dep_res = depressed_binary_experiment()
    print('Depressed binary experiment:', dep_res)
    print('Saved outputs under', OUT_DIR)
