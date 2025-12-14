import pandas as pd
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent
O = ROOT / 'eval_outputs' / 'oof_cv'
fn = O / 'oof_predictions.csv'

df = pd.read_csv(fn)
true = df['true']
pred = df['pred']
prob = df['prob']

bal = balanced_accuracy_score(true, pred)
rep = classification_report(true, pred, output_dict=True)
cm = confusion_matrix(true, pred)

print('balanced_accuracy:', bal)
print('accuracy:', rep['accuracy'])
print('precision (pos):', rep['1']['precision'])
print('recall (pos):', rep['1']['recall'])
print('f1 (pos):', rep['1']['f1-score'])
print('confusion_matrix:\n', cm)

with open(O / 'oof_balanced_report.json','w',encoding='utf-8') as f:
    import json
    json.dump({'balanced_accuracy':bal, 'report':rep, 'confusion_matrix':cm.tolist()}, f, ensure_ascii=False, indent=2)
