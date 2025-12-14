"""Comprehensive optimization: tune XGBoost, compute OOF with best params, SHAP-guided feature selection retrain, threshold recommendations and capacity sweep.
Saves outputs to models/eval_outputs/opt_results/
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import balanced_accuracy_score, precision_recall_curve, precision_recall_fscore_support
import xgboost as xgb

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT.parent / 'smmh.csv'
OUT = ROOT / 'eval_outputs' / 'opt_results'
OUT.mkdir(parents=True, exist_ok=True)

# reuse existing preprocessing from oof_xgb_cv
from oof_xgb_cv import load_preprocess


def cv_score_for_params(X, y, params, n_splits=5, early_stop_rounds=10):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[test_idx]
        # further split train to have an eval_set for early stopping
        X_tr2, X_es, y_tr2, y_es = train_test_split(X_tr, y_tr, test_size=0.15, stratify=y_tr, random_state=42)
        n_pos = int((y_tr2==1).sum())
        n_neg = int((y_tr2==0).sum())
        spw = params.get('scale_pos_weight', max(1.0, n_neg / max(1, n_pos)))
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, **params)
        try:
            model.fit(X_tr2, y_tr2, eval_set=[(X_es, y_es)], early_stopping_rounds=early_stop_rounds, verbose=False)
        except Exception:
            model.fit(X_tr2, y_tr2)
        y_pred = model.predict(X_val)
        scores.append(balanced_accuracy_score(y_val, y_pred))
    return float(np.mean(scores))


def tune_grid(X, y):
    grid = []
    for n in [100, 200]:
        for md in [3,5]:
            for lr in [0.05, 0.1]:
                grid.append({'n_estimators': n, 'max_depth': md, 'learning_rate': lr})
    best = None
    best_score = -1
    results = []
    for params in grid:
        print('Evaluating', params)
        score = cv_score_for_params(X, y, params)
        params['score'] = score
        results.append(dict(params))
        if score > best_score:
            best_score = score
            best = dict(params)
    pd.DataFrame(results).to_csv(OUT / 'tuning_grid_results.csv', index=False)
    with open(OUT / 'tuning_best.json','w',encoding='utf-8') as f:
        json.dump({'best': best, 'best_score': best_score}, f, ensure_ascii=False, indent=2)
    return best


def oof_with_params(X, y, params, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_pred = np.zeros(len(X), dtype=int)
    oof_prob = np.zeros(len(X), dtype=float)
    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        n_pos = int((y_tr==1).sum())
        n_neg = int((y_tr==0).sum())
        spw = params.get('scale_pos_weight', max(1.0, n_neg / max(1, n_pos)))
        clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, scale_pos_weight=spw, **{k:v for k,v in params.items() if k!='score'})
        clf.fit(X_tr, y_tr)
        prob = clf.predict_proba(X_te)[:,1]
        pred = (prob >= 0.5).astype(int)
        oof_prob[test_idx] = prob
        oof_pred[test_idx] = pred
    # metrics
    prec, rec, f1, _ = precision_recall_fscore_support(y, oof_pred, average='binary', zero_division=0)
    bal = balanced_accuracy_score(y, oof_pred)
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y, oof_prob)
    except Exception:
        auc = None
    out = {'accuracy': float((oof_pred==y.values).mean()), 'balanced_accuracy': float(bal), 'precision': float(prec), 'recall': float(rec), 'f1': float(f1), 'roc_auc': float(auc) if auc is not None else None}
    pd.DataFrame({'true': y.values, 'pred': oof_pred, 'prob': oof_prob}).to_csv(OUT / 'oof_with_tuned_params.csv', index=False)
    with open(OUT / 'oof_with_tuned_params_metrics.json','w',encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return out


def train_final_and_shap(X, y, params, top_k=10):
    # train on full data
    n_pos = int((y==1).sum())
    n_neg = int((y==0).sum())
    spw = params.get('scale_pos_weight', max(1.0, n_neg / max(1, n_pos)))
    clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, scale_pos_weight=spw, **{k:v for k,v in params.items() if k!='score'})
    clf.fit(X, y)
    try:
        import shap
        expl = shap.TreeExplainer(clf)
        sv = expl.shap_values(X)
        if isinstance(sv, list):
            mv = np.mean(np.abs(sv[1]), axis=0)
        else:
            mv = np.mean(np.abs(sv), axis=0)
        feat_imp = pd.Series(mv, index=X.columns).sort_values(ascending=False)
        feat_imp.to_csv(OUT / 'final_shap_feature_importance.csv')
        top_features = feat_imp.head(top_k).index.tolist()
    except Exception:
        feat_imp = pd.Series(np.zeros(X.shape[1]), index=X.columns)
        feat_imp.to_csv(OUT / 'final_shap_feature_importance.csv')
        top_features = list(X.columns)[:top_k]
    # retrain OOF with top features
    X_sub = X[top_features]
    oof_res = oof_with_params(X_sub, y, params)
    with open(OUT / 'feature_selection_top.json','w',encoding='utf-8') as f:
        json.dump({'top_features': top_features}, f, ensure_ascii=False, indent=2)
    return feat_imp, top_features, oof_res


def threshold_and_capacity(oof_csv):
    df = pd.read_csv(oof_csv)
    y = df['true'].values
    prob = df['prob'].values
    precisions, recalls, thresholds = precision_recall_curve(y, prob)
    # for target recalls 0.8 and 0.9 find smallest threshold achieving recall >= target
    targets = [0.8, 0.9]
    rec_map = {}
    for t in targets:
        idx = np.where(recalls >= t)[0]
        if len(idx)==0:
            rec_map[str(t)] = None
        else:
            th = thresholds[idx[-1]-1] if idx[-1]>0 and len(thresholds)>0 else 0.0
            rec_map[str(t)] = float(th)
    # capacity sweep: sort by prob desc and compute TP in top k
    order = np.argsort(-prob)
    sorted_true = y[order]
    caps = list(range(50, min(501,len(y))+1, 50))
    sweep = []
    for cap in caps:
        top = sorted_true[:cap]
        tp = int((top==1).sum())
        precision = tp / cap
        recall = tp / int(y.sum()) if y.sum()>0 else None
        sweep.append({'cap':cap,'alerts':cap,'true_positives':tp,'precision':precision,'recall':recall})
    pd.DataFrame(sweep).to_csv(OUT / 'capacity_sweep.csv', index=False)
    with open(OUT / 'threshold_recommendations.json','w',encoding='utf-8') as f:
        json.dump({'recall_thresholds': rec_map}, f, ensure_ascii=False, indent=2)
    return rec_map, sweep


def main():
    X, y = load_preprocess()
    print('Starting tuning...')
    best = tune_grid(X, y)
    print('Best found:', best)
    print('Running OOF with tuned params...')
    oof_metrics = oof_with_params(X, y, best)
    print('OOF metrics:', oof_metrics)
    print('Training final model & SHAP feature selection...')
    feat_imp, top_feats, oof_res2 = train_final_and_shap(X, y, best, top_k=10)
    print('Top features:', top_feats)
    rec_map, sweep = threshold_and_capacity(OUT / 'oof_with_tuned_params.csv')
    print('Thresholds for recalls:', rec_map)
    print('Done. Results in', OUT)

if __name__ == '__main__':
    main()
