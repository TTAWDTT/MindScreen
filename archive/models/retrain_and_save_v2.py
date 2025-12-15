"""Train final XGBoost on full data with tuned params and top features, save pipeline v2 and update smmh_model_info.json"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import joblib
import xgboost as xgb

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT.parent / 'smmh.csv'
OPT_DIR = ROOT / 'eval_outputs' / 'opt_results'
MODEL_DIR = ROOT

# load best params and top features
with open(OPT_DIR / 'tuning_best.json','r',encoding='utf-8') as f:
    best = json.load(f).get('best', {})
with open(OPT_DIR / 'feature_selection_top.json','r',encoding='utf-8') as f:
    top = json.load(f).get('top_features', [])

print('Best params:', best)
print('Top features:', top)

# read data and preprocess similar to earlier pipelines
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
df = df.dropna(subset=['q18'])
df['target'] = df['q18'].apply(lambda x: 1 if x>=4 else 0)

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

for c in ['relationship','occupation']:
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

# ensure top features exist
for f in top:
    if f not in df.columns:
        df[f] = 0

X = df[top]
y = df['target'].astype(int)

# build pipeline that selects top features then scales then model
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

selector = ColumnTransformer([('sel','passthrough', top)], remainder='drop')
clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, **{k:best[k] for k in best if k!='score'})

pipe = Pipeline([
    ('select', selector),
    ('scaler', StandardScaler()),
    ('model', clf)
])

pipe.fit(X, y)

# set human-readable classes
orig_labels = ['lower','higher']
try:
    pipe.named_steps['model'].classes_ = np.array(orig_labels)
    pipe.classes_ = np.array(orig_labels)
except Exception:
    pass

# save pipeline
v2_name = MODEL_DIR / 'smmh_risk_pipeline_v2.pkl'
joblib.dump(pipe, v2_name)
print('Saved pipeline to', v2_name)

# update smmh_model_info.json with v2 metadata
info_path = MODEL_DIR / 'smmh_model_info.json'
info = {}
if info_path.exists():
    info = json.loads(info_path.read_text(encoding='utf-8'))

info['risk_v2'] = {
    'model': 'xgb',
    'params': {k:best[k] for k in best if k!='score'},
    'top_features': top,
    'artifact': str(v2_name.name)
}
info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
print('Updated', info_path)
