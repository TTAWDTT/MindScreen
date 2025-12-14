"""模拟告警排序与限额，生成对比报告。

优先级（从高到低）:
 1) B_strong
 2) A_strong AND C_early
 3) A_strong (其余)
 4) C_early (其余)
 5) early_warning
 6) low_priority

在同一优先级内按 `model_score` 降序，再按 `digital_addiction_score` 降序排序。
对不同的每天告警上限（caps）模拟并输出 precision/recall/f1/告警量。
生成文件到 `models/eval_outputs/alert_ordering_*`。
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
IN_FILE = ROOT / 'eval_outputs' / 'rule_simulation_with_candidates_details.csv'
OUT_DIR = ROOT / 'eval_outputs' / 'alert_ordering_outputs'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_details():
    df = pd.read_csv(IN_FILE, parse_dates=False)
    # ensure needed cols exist
    for c in ['B_strong','A_strong','C_early','early_warning','low_priority','model_score','digital_addiction_score','target','avg_time_ord']:
        if c not in df.columns:
            df[c] = 0
    return df


def assign_priority(df):
    df = df.copy()
    # compute priority integer (smaller is higher priority)
    # 1:B_strong
    # 2:A_strong & C_early
    # 3:A_strong only
    # 4:C_early only
    # 5:early_warning
    # 6:low_priority
    def get_pr(row):
        if row['B_strong'] == 1:
            return 1
        if row['A_strong'] == 1 and row['C_early'] == 1:
            return 2
        if row['A_strong'] == 1:
            return 3
        if row['C_early'] == 1:
            return 4
        if row['early_warning'] == 1:
            return 5
        if row['low_priority'] == 1:
            return 6
        return 7

    df['priority'] = df.apply(get_pr, axis=1)
    # tie-breaker: model_score desc, digital_addiction_score desc
    df = df.sort_values(['priority','model_score','digital_addiction_score'], ascending=[True, False, False]).reset_index(drop=True)
    return df


def simulate_caps(df, caps=[50,100,200,300,481]):
    rows = []
    total = len(df)
    for cap in caps:
        sel = df.head(cap)
        tp = int(((sel['target']==1)).sum())
        count = len(sel)
        prec = tp / count if count>0 else 0.0
        rec = tp / df['target'].sum() if df['target'].sum()>0 else 0.0
        # f1
        f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
        rows.append({'cap': cap, 'alerts': count, 'true_positives': tp, 'precision': prec, 'recall': rec, 'f1': f1, 'per_1000': count/total*1000})
    return pd.DataFrame(rows)


def plot_metrics(df_metrics):
    plt.figure(figsize=(6,4))
    plt.plot(df_metrics['cap'], df_metrics['precision'], marker='o', label='precision')
    plt.plot(df_metrics['cap'], df_metrics['recall'], marker='o', label='recall')
    plt.plot(df_metrics['cap'], df_metrics['f1'], marker='o', label='f1')
    plt.xlabel('daily alert cap')
    plt.ylabel('metric')
    plt.title('Alert metrics vs cap')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'metrics_vs_cap.png')
    plt.close()


def save_report(df_sorted, df_metrics):
    df_sorted.to_csv(OUT_DIR / 'alert_ordered_details.csv', index=False)
    df_metrics.to_csv(OUT_DIR / 'alert_ordering_simulation.csv', index=False)

    # simple HTML report
    html = ['<html><head><meta charset="utf-8"><title>Alert Ordering Report</title></head><body>']
    html.append('<h2>Alert Ordering Simulation</h2>')
    html.append('<h3>Metrics vs Cap</h3>')
    html.append(f'<img src="metrics_vs_cap.png" alt="metrics">')
    html.append('<h3>Summary Table</h3>')
    html.append(df_metrics.to_html(index=False))
    html.append('<h3>Top 50 Ordered Alerts</h3>')
    html.append(df_sorted.head(50).to_html(index=False))
    html.append('</body></html>')
    with open(OUT_DIR / 'alert_ordering_report.html','w',encoding='utf-8') as f:
        f.write('\n'.join(html))


def main():
    df = load_details()
    df_sorted = assign_priority(df)
    caps = [50,100,200,300,481]
    df_metrics = simulate_caps(df_sorted, caps)
    plot_metrics(df_metrics)
    save_report(df_sorted, df_metrics)
    print('Saved alert ordering outputs to', OUT_DIR)


if __name__ == '__main__':
    main()
