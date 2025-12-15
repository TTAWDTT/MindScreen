"""Simulate candidate rules A/B/C using OOF predictions and original features, produce summary CSV."""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
OOF = ROOT / 'eval_outputs' / 'opt_results' / 'oof_with_tuned_params.csv'
DATA = ROOT.parent / 'smmh.csv'
OUT = ROOT / 'eval_outputs' / 'opt_results'

# load oof and original data
oof = pd.read_csv(OOF)
df = pd.read_csv(DATA)

# preprocess to compute digital_addiction_score and avg_time_ord similarly to other scripts
col_map = {
    '1. What is your age?': 'age',
    '8. What is the average time you spend on social media every day?': 'avg_time',
    '9. How often do you find yourself using Social media without a specific purpose?': 'q9',
    '10. How often do you get distracted by Social media when you are busy doing something?': 'q10',
    "11. Do you feel restless if you haven't used Social media in a while?": 'q11',
    '12. On a scale of 1 to 5, how easily distracted are you?': 'q12'
}

df = df.rename(columns=col_map)
for q in ['q9','q10','q11','q12']:
    df[q] = pd.to_numeric(df.get(q,0), errors='coerce').fillna(0)

df['digital_addiction_score'] = df[['q9','q10','q11','q12']].sum(axis=1)

time_map = {
    'Less than an Hour': 0,
    'Between 1 and 2 hours': 1,
    'Between 2 and 3 hours': 2,
    'Between 3 and 4 hours': 3,
    'Between 4 and 5 hours': 4,
    'More than 5 hours': 5
}
if 'avg_time' in df.columns:
    df['avg_time_ord'] = df['avg_time'].map(time_map).fillna(2)
else:
    df['avg_time_ord'] = 2

# align lengths: oof corresponds to preprocessed subset used earlier; assume same order and length
# merge oof prob with features by index
merged = df.copy()
merged['prob'] = oof['prob']
merged['true'] = oof['true']

# thresholds from earlier
thr = pd.read_json(OUT / 'threshold_recommendations.json', typ='series')
th_08 = thr.get('recall_thresholds', {}).get('0.8')
th_09 = thr.get('recall_thresholds', {}).get('0.9')
# fallback
if th_08 is None:
    th_08 = 0.45
if th_09 is None:
    th_09 = 0.38

# candidate rules:
# A: digital_addiction_score >=16 OR model_prob >= th_08
# B: digital_addiction_score >=15 AND model_prob >= th_08
# C: digital_addiction_score in {13,14} AND avg_time_ord >=3

merged['A_rule'] = ((merged['digital_addiction_score'] >= 16) | (merged['prob'] >= th_08)).astype(int)
merged['B_rule'] = ((merged['digital_addiction_score'] >= 15) & (merged['prob'] >= th_08)).astype(int)
merged['C_rule'] = (merged['digital_addiction_score'].isin([13,14]) & (merged['avg_time_ord'] >= 3)).astype(int)

# compute metrics for each rule
from sklearn.metrics import precision_score, recall_score

rows = []
for name in ['A_rule','B_rule','C_rule']:
    preds = merged[name]
    precision = precision_score(merged['true'], preds, zero_division=0)
    recall = recall_score(merged['true'], preds, zero_division=0)
    alerts = int(preds.sum())
    rows.append({'rule':name,'alerts':alerts,'precision':precision,'recall':recall})

pd.DataFrame(rows).to_csv(OUT / 'candidate_rules_summary.csv', index=False)
merged.to_csv(OUT / 'candidate_rules_details.csv', index=False)
print('Saved candidate rules summary to', OUT / 'candidate_rules_summary.csv')
