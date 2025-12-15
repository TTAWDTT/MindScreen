"""Do extra tuning with simple minority upsampling and compare to scale_pos_weight approach."""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils import resample
import xgboost as xgb
import json

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT.parent / 'smmh.csv'
OUT = ROOT / 'eval_outputs' / 'opt_results'
OUT.mkdir(parents=True, exist_ok=True)

# load data via oof_xgb_cv preprocessing
from oof_xgb_cv import load_preprocess
X, y = load_preprocess()

param_grid = [
    {'n_estimators':100,'max_depth':3,'learning_rate':0.05},
    {'n_estimators':200,'max_depth':5,'learning_rate':0.05},
]

results = []
for params in param_grid:
    # approach 1: scale_pos_weight
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores_spw = []
    for tr, te in skf.split(X,y):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]
        n_pos = int((ytr==1).sum()); n_neg = int((ytr==0).sum())
        spw = max(1.0, n_neg / max(1, n_pos))
        clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, scale_pos_weight=spw, **params)
        clf.fit(Xtr,ytr)
        scores_spw.append(balanced_accuracy_score(yte, clf.predict(Xte)))
    mean_spw = float(np.mean(scores_spw))

    # approach 2: upsample minority in train folds
    scores_up = []
    for tr, te in skf.split(X,y):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]
        # concatenate and upsample minority
        df_tr = pd.concat([Xtr, ytr.rename('target')], axis=1)
        # majority and minority
        maj = df_tr[df_tr['target']==0]
        mino = df_tr[df_tr['target']==1]
        mino_up = resample(mino, replace=True, n_samples=len(maj), random_state=42)
        df_up = pd.concat([maj, mino_up])
        Xu = df_up.drop(columns=['target']); yu = df_up['target']
        clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, **params)
        clf.fit(Xu, yu)
        scores_up.append(balanced_accuracy_score(yte, clf.predict(Xte)))
    mean_up = float(np.mean(scores_up))

    results.append({'params':params,'scale_pos_weight_bal_acc':mean_spw,'upsample_bal_acc':mean_up})

pd.DataFrame(results).to_csv(OUT / 'extra_tuning_upsample_results.csv', index=False)
with open(OUT / 'extra_tuning_upsample.json','w',encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print('Saved extra tuning results to', OUT)
