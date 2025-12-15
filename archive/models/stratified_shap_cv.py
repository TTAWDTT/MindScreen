"""分层交叉验证下的 SHAP 稳定性与阈值评估脚本

功能：
- 使用 StratifiedKFold 对数据做 5 折交叉验证
- 每折训练 XGBoost（scale_pos_weight 基于该折训练集计算），
  记录每折在给定阈值下的 precision/recall/f1（用于 0.26493 和 0.38266）
- 每折计算 test-fold 上 `digital_addiction_score` 的平均 SHAP 值并保存
- 聚合所有折的 mean SHAP，计算均值与标准差，统计 12->13 转折在多少折内出现

输出（保存到 `models/eval_outputs/stratified_shap_cv/`）：
- per_fold_mean_shap_fold{i}.csv
- aggregated_mean_shap_by_score.csv
- threshold_fold_metrics.csv
- crossing_summary.txt
"""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score
import xgboost as xgb
import shap

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT.parent / 'smmh.csv'
BASE_OUT = ROOT / 'eval_outputs' / 'stratified_shap_cv'
BASE_OUT.mkdir(parents=True, exist_ok=True)


def load_features():
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

    # preprocess features per spec
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

    # features order
    feature_cols = ['age','avg_time_ord','platform_count','digital_addiction_score'] + [f'plat_{p}' for p in main_platforms]
    df = pd.get_dummies(df, columns=['gender'], prefix='gender')
    X = df.reindex(columns=[c for c in feature_cols if c in df.columns] + [col for col in df.columns if col.startswith('gender_')]).fillna(-1)
    y = df['target'].astype(int)
    return df, X, y


def evaluate_cv(X, y, n_splits=5, thresholds=[0.26493003964424133, 0.3826602101325989]):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []
    per_fold_mean_shaps = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        n_pos = (y_train==1).sum()
        n_neg = (y_train==0).sum()
        spw = float(n_neg) / max(1, int(n_pos))

        model = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, use_label_encoder=False, eval_metric='logloss', random_state=42, scale_pos_weight=spw)
        model.fit(X_train, y_train)

        probs = model.predict_proba(X_test)[:,1]
        preds_by_thr = {thr: (probs >= thr).astype(int) for thr in thresholds}

        for thr, preds in preds_by_thr.items():
            prec = precision_score(y_test, preds, zero_division=0)
            rec = recall_score(y_test, preds, zero_division=0)
            f1 = f1_score(y_test, preds, zero_division=0)
            fold_results.append({'fold': fold, 'threshold': thr, 'precision': prec, 'recall': rec, 'f1': f1, 'n_test': len(y_test)})

        # SHAP on test fold
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        if isinstance(shap_values, list):
            sv = shap_values[1]
        else:
            sv = shap_values

        # index of digital_addiction_score
        if 'digital_addiction_score' in X_test.columns:
            idx = list(X_test.columns).index('digital_addiction_score')
            das = X_test['digital_addiction_score'].values
            shap_das = sv[:, idx]
            df_sh = pd.DataFrame({'digital_addiction_score': das, 'shap': shap_das})
            mean_sh = df_sh.groupby('digital_addiction_score')['shap'].mean().reset_index()
            mean_sh.to_csv(BASE_OUT / f'per_fold_mean_shap_fold{fold}.csv', index=False)
            per_fold_mean_shaps.append((fold, mean_sh))

    return fold_results, per_fold_mean_shaps


def aggregate_and_report(fold_results, per_fold_mean_shaps):
    # save fold-level threshold metrics
    df_fr = pd.DataFrame(fold_results)
    df_fr.to_csv(BASE_OUT / 'threshold_fold_metrics.csv', index=False)

    # aggregate mean shap by digital_addiction_score across folds
    all_scores = {}
    for fold, dfm in per_fold_mean_shaps:
        for _, row in dfm.iterrows():
            score = int(row['digital_addiction_score'])
            all_scores.setdefault(score, []).append(float(row['shap']))

    agg = [{'digital_addiction_score': s, 'mean_shap_mean': np.mean(v), 'mean_shap_std': np.std(v), 'n_folds': len(v)} for s, v in sorted(all_scores.items())]
    df_agg = pd.DataFrame(agg)
    df_agg.to_csv(BASE_OUT / 'aggregated_mean_shap_by_score.csv', index=False)

    # detect crossing per fold and overall
    fold_crossings = []
    for fold, dfm in per_fold_mean_shaps:
        crosses = dfm.sort_values('digital_addiction_score')
        cross_point = None
        for i in range(len(crosses)-1):
            a = crosses['shap'].iloc[i]
            b = crosses['shap'].iloc[i+1]
            if a<=0 and b>=0:
                cross_point = (int(crosses['digital_addiction_score'].iloc[i]), int(crosses['digital_addiction_score'].iloc[i+1]))
                break
        fold_crossings.append({'fold': fold, 'cross_point': cross_point})

    # how many folds have crossing at 12->13
    count_12_13 = sum(1 for fc in fold_crossings if fc['cross_point'] == (12,13))

    with open(BASE_OUT / 'crossing_summary.txt','w',encoding='utf-8') as f:
        f.write('Per-fold crossings:\n')
        for fc in fold_crossings:
            f.write(str(fc) + '\n')
        f.write(f'Folds with crossing at (12,13): {count_12_13} / {len(fold_crossings)}\n')

    return df_fr, df_agg, fold_crossings


def main():
    df, X, y = load_features()
    fold_results, per_fold_mean_shaps = evaluate_cv(X, y, n_splits=5)
    df_fr, df_agg, fold_crossings = aggregate_and_report(fold_results, per_fold_mean_shaps)
    print('Saved outputs to', BASE_OUT)


if __name__ == '__main__':
    main()
