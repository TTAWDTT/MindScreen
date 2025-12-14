"""Train models on smmh.csv (Social Media & Mental Health survey).

Targets
- risk: binary (high/low) based on impact_sum >= 37 (sum of 12 Likert questions)
- depressed: 1-5 Likert treated as multiclass

Features (no label leakage):
- age (numeric)
- gender, relationship, avg_time_per_day (categorical)
- platforms one-hot (split by comma)
- platform_sum (count of platforms used)

Outputs (saved to ../models):
- smmh_risk_pipeline.pkl
- smmh_depressed_pipeline.pkl
- smmh_model_info.json (metrics, features)

Usage:
    python train_smmh_models.py
"""
import json
import os
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "smmh.csv"))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "models"))

COL_RENAME = {
    'Timestamp': 'timestamp',
    '1. What is your age?': 'age',
    '2. Gender': 'gender',
    '3. Relationship Status': 'relationship',
    '4. Occupation Status': 'occupation',
    '5. What type of organizations are you affiliated with?': 'affiliate_organization',
    '6. Do you use social media?': 'social_media_use',
    '7. What social media platforms do you commonly use?': 'platforms',
    '8. What is the average time you spend on social media every day?': 'avg_time_per_day',
    '9. How often do you find yourself using Social media without a specific purpose?': 'without_purpose',
    '10. How often do you get distracted by Social media when you are busy doing something?': 'distracted',
    "11. Do you feel restless if you haven't used Social media in a while?": 'restless',
    '12. On a scale of 1 to 5, how easily distracted are you?': 'distracted_ease',
    '13. On a scale of 1 to 5, how much are you bothered by worries?': 'worries',
    '14. Do you find it difficult to concentrate on things?': 'concentration',
    '15. On a scale of 1-5, how often do you compare yourself to other successful people through the use of social media?': 'compare_to_others',
    '16. Following the previous question, how do you feel about these comparisons, generally speaking?': 'compare_feelings',
    '17. How often do you look to seek validation from features of social media?': 'validation',
    '18. How often do you feel depressed or down?': 'depressed',
    '19. On a scale of 1 to 5, how frequently does your interest in daily activities fluctuate?': 'daily_activity_flux',
    '20. On a scale of 1 to 5, how often do you face issues regarding sleep?': 'sleeping_issues'
}

LIKERT_COLS = [
    'without_purpose', 'distracted', 'restless', 'distracted_ease', 'worries',
    'concentration', 'compare_to_others', 'compare_feelings', 'validation',
    'depressed', 'daily_activity_flux', 'sleeping_issues'
]

AVG_TIME_ORDER = [
    'Less than an Hour',
    'Between 1 and 2 hours',
    'Between 2 and 3 hours',
    'Between 3 and 4 hours',
    'Between 4 and 5 hours',
    'More than 5 hours'
]


def compute_percentile(val, series: pd.Series):
    if series is None or series.empty or val is None:
        return None
    q = np.nanpercentile(series, np.arange(0, 101))
    pct = np.interp(val, q, np.arange(0, 101))
    pct = float(np.clip(pct, 0, 100))
    return round(pct, 1)


def load_and_clean() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns=COL_RENAME)

    # Fill affiliate org missing with mode
    if df['affiliate_organization'].isnull().any():
        mode_val = df['affiliate_organization'].value_counts().index[0]
        df['affiliate_organization'] = df['affiliate_organization'].fillna(mode_val)

    # Normalize gender
    df['gender'] = df['gender'].apply(lambda x: x if x in ['Male', 'Female'] else 'other')

    # avg_time_per_day to ordered category; keep raw for encoding
    df['avg_time_per_day'] = pd.Categorical(df['avg_time_per_day'], categories=AVG_TIME_ORDER, ordered=True)

    # Split platform list into set
    platform_dummies = df['platforms'].fillna('').apply(lambda x: [p.strip() for p in str(x).split(',') if p.strip()])
    all_platforms = sorted(set(p for lst in platform_dummies for p in lst))
    for p in all_platforms:
        df[p] = platform_dummies.apply(lambda lst: 1 if p in lst else 0)
    df['platform_sum'] = df[all_platforms].sum(axis=1)

    # Digital addiction score (engineered): sum of Q9-Q12 (behavioral features)
    q_cols = ['without_purpose', 'distracted', 'restless', 'distracted_ease']
    for c in q_cols:
        if c not in df.columns:
            df[c] = 0
    df['digital_addiction_score'] = df[q_cols].sum(axis=1)

    # avg_time_per_day ordinal numeric encoding for models
    try:
        df['avg_time_per_day'] = pd.Categorical(df['avg_time_per_day'], categories=AVG_TIME_ORDER, ordered=True)
        df['avg_time_ord'] = df['avg_time_per_day'].cat.codes.replace(-1, pd.NA).astype('float')
        if df['avg_time_ord'].isnull().any():
            median_ord = int(df['avg_time_ord'].median(skipna=True) if not df['avg_time_ord'].isnull().all() else 2)
            df['avg_time_ord'] = df['avg_time_ord'].fillna(median_ord)
    except Exception:
        df['avg_time_ord'] = 2

    # impact sum for label construction
    df['impact_sum'] = df[LIKERT_COLS].sum(axis=1)
    df['risk'] = np.where(df['impact_sum'] >= 37, 'higher', 'lower')

    # --- Backwards-compatibility columns ---
    # Some analysis scripts expect prefixed platform columns (e.g. 'plat_Reddit'),
    # encoded relationship/occupation columns, and a 'platform_count' column.
    # Create those here so different modules share a common DataFrame schema.
    for p in all_platforms:
        plat_col = f'plat_{p}'
        if p in df.columns:
            df[plat_col] = df[p].fillna(0).astype(int)
        else:
            df[plat_col] = platform_dummies.apply(lambda lst: 1 if p in lst else 0)

    # platform_count kept for compatibility (some scripts use this name)
    df['platform_count'] = df.get('platform_sum', df[all_platforms].sum(axis=1))

    # simple label-encodings for relationship and occupation used in analyses
    df['relationship_enc'] = pd.factorize(df['relationship'].fillna('unknown'))[0]
    df['occupation_enc'] = pd.factorize(df['occupation'].fillna('unknown'))[0]

    # gender one-hot columns for downstream scripts that expect gender_Male etc.
    gender_dummies = pd.get_dummies(df['gender'].fillna('other'), prefix='gender')
    for c in gender_dummies.columns:
        df[c] = gender_dummies[c]

    # ensure digital_addiction_score exists (some callers rely on it)
    if 'digital_addiction_score' not in df.columns:
        df['digital_addiction_score'] = df[q_cols].sum(axis=1)

    return df, all_platforms


def make_preprocessor(num_cols, cat_cols):
    return ColumnTransformer(
        [
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
        ]
    )


def train_model(X: pd.DataFrame, y: pd.Series, preprocessor, model_space: Dict[str, Tuple[Any, Dict[str, Any]]]):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    best = None
    best_name = None
    best_score = -np.inf
    best_metrics: Dict[str, Any] = {}

    for name, (est, grid) in model_space.items():
        # Allow per-model adjustments (e.g., XGBoost scale_pos_weight based on train class balance)
        grid_to_use = grid.copy()
        if name == 'xgb':
            try:
                counts = y_train.value_counts()
                n_pos = int(counts.get('higher', counts.min()))
                n_neg = int(counts.sum() - n_pos)
                spw = max(1.0, float(n_neg / max(1, n_pos)))
            except Exception:
                spw = 1.0
            grid_to_use = dict(grid)
            grid_to_use['model__scale_pos_weight'] = [spw]

        pipe = Pipeline([
            ('prep', preprocessor),
            ('model', est),
        ])
        search = GridSearchCV(pipe, grid_to_use, cv=3, scoring='balanced_accuracy', n_jobs=-1)
        search.fit(X_train, y_train)
        candidate = search.best_estimator_
        y_pred = candidate.predict(X_test)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        if bal_acc > best_score:
            best_score = bal_acc
            best = candidate
            best_name = name
            best_metrics = {
                'model': name,
                'params': search.best_params_,
                'balanced_accuracy': float(bal_acc),
                'f1_macro': float(f1_macro),
                'report': classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            }
    return best, best_metrics


def main():
    print("Loading and cleaning data...")
    df, platforms = load_and_clean()
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Baseline ("normal") values for each Likert item: mean raw score and its percentile within training data
    baseline_map = {}
    for col in LIKERT_COLS:
        series = df[col]
        mean_val = float(series.mean()) if not series.empty else None
        baseline_map[col] = {
            'baseline_value': mean_val,
            'baseline_percentile': compute_percentile(mean_val, series) if mean_val is not None else None
        }

    # Use behavior-only features (no consequence items Q13-Q20). Add engineered features.
    # Use compatible column names (plat_*, relationship_enc, occupation_enc, platform_count) 
    # to match other analysis scripts in models/
    plat_cols = [f'plat_{p}' for p in platforms]
    gender_cols = [c for c in df.columns if c.startswith('gender_')]
    
    feature_cols = ['age', 'relationship_enc', 'occupation_enc', 'avg_time_ord', 
                    'platform_count', 'digital_addiction_score'] + plat_cols + gender_cols

    # risk target
    X_risk = df[feature_cols]
    y_risk_raw = df['risk']
    # map labels to integers for consistent training (e.g., 'lower'/'higher' -> 0/1)
    unique_risk = list(y_risk_raw.unique())
    risk_mapping = {lab: i for i, lab in enumerate(sorted(unique_risk))}
    y_risk = y_risk_raw.map(risk_mapping)
    num_cols = ['age', 'platform_count', 'digital_addiction_score', 'avg_time_ord', 
                'relationship_enc', 'occupation_enc']
    cat_cols = plat_cols + gender_cols
    preprocessor = make_preprocessor(num_cols, cat_cols)

    model_space_risk = {
        'logreg': (LogisticRegression(max_iter=300), {
            'model__C': [1.0]
        }),
        'rf': (RandomForestClassifier(random_state=42, n_estimators=160), {
            'model__max_depth': [12, None],
            'model__min_samples_leaf': [1],
        })
    }

    # Add XGBoost candidate with a small grid; we'll set scale_pos_weight based on train split inside train_model
    model_space_risk['xgb'] = (XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, verbosity=0), {
        'model__n_estimators': [100, 200],
        'model__max_depth': [3, 6],
        'model__learning_rate': [0.05, 0.1],
    })

    print("Training risk model...")
    risk_model, risk_metrics = train_model(X_risk, y_risk, preprocessor, model_space_risk)
    # attach original class labels so API can present human-readable labels
    try:
        orig_labels = sorted(unique_risk)
        # set classes_ on model and pipeline for downstream code expecting human labels
        if hasattr(risk_model, 'named_steps') and 'model' in risk_model.named_steps:
            risk_model.named_steps['model'].classes_ = np.array(orig_labels)
        risk_model.classes_ = np.array(orig_labels)
    except Exception:
        pass
    joblib.dump(risk_model, os.path.join(MODEL_DIR, 'smmh_risk_pipeline.pkl'))

    # depressed target (multiclass 1-5)
    X_dep = df[feature_cols]
    y_dep_raw = df['depressed']
    unique_dep = list(y_dep_raw.unique())
    dep_mapping = {lab: i for i, lab in enumerate(sorted(unique_dep))}
    y_dep = y_dep_raw.map(dep_mapping)
    model_space_dep = {
        'logreg': (LogisticRegression(max_iter=400), {
            'model__C': [1.0]
        }),
        'rf': (RandomForestClassifier(random_state=42, n_estimators=180), {
            'model__max_depth': [12, None],
            'model__min_samples_leaf': [1],
        })
    }
    print("Training depressed model...")
    dep_model, dep_metrics = train_model(X_dep, y_dep, preprocessor, model_space_dep)
    try:
        orig_dep_labels = sorted(unique_dep)
        if hasattr(dep_model, 'named_steps') and 'model' in dep_model.named_steps:
            dep_model.named_steps['model'].classes_ = np.array(orig_dep_labels)
        dep_model.classes_ = np.array(orig_dep_labels)
    except Exception:
        pass
    joblib.dump(dep_model, os.path.join(MODEL_DIR, 'smmh_depressed_pipeline.pkl'))

    # Compute composite mental health score distribution for all training samples
    print("Computing composite score distribution...")
    X_all = df[feature_cols]
    
    # Get risk probabilities (prob of 'higher' risk)
    risk_proba = risk_model.predict_proba(X_all)
    # Find index of 'higher' class
    try:
        higher_idx = list(risk_model.classes_).index('higher')
    except (ValueError, AttributeError):
        higher_idx = 1  # fallback
    risk_prob_higher = risk_proba[:, higher_idx]
    
    # Get depressed predictions (1-5 scale)
    dep_pred = dep_model.predict(X_all)
    # Map back to numeric if needed
    dep_numeric = []
    for pred in dep_pred:
        try:
            dep_numeric.append(float(pred))
        except (ValueError, TypeError):
            dep_numeric.append(3.0)  # default middle value
    dep_numeric = np.array(dep_numeric)
    
    # Composite score formula: weighted average of risk probability and normalized depression level
    # Score range: 0-1, higher means worse mental health
    composite_scores = 0.5 * risk_prob_higher + 0.5 * (dep_numeric / 5.0)
    
    # Save score distribution as percentiles for ranking
    composite_percentiles = np.percentile(composite_scores, np.arange(0, 101))
    
    model_info = {
        'dataset': 'smmh.csv',
        'n_rows': len(df),
        'features': feature_cols,
        'platforms': platforms,
        'likert_baseline': baseline_map,
        'risk': risk_metrics,
        'depressed': dep_metrics,
        'composite_score_distribution': {
            'percentiles': composite_percentiles.tolist(),
            'mean': float(np.mean(composite_scores)),
            'std': float(np.std(composite_scores)),
            'min': float(np.min(composite_scores)),
            'max': float(np.max(composite_scores)),
            'formula': 'composite = 0.5 * P(risk=higher) + 0.5 * (depressed_level / 5)'
        }
    }
    with open(os.path.join(MODEL_DIR, 'smmh_model_info.json'), 'w', encoding='utf-8') as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)

    print("Done. Artifacts saved to models/:")
    print(" - smmh_risk_pipeline.pkl")
    print(" - smmh_depressed_pipeline.pkl")
    print(" - smmh_model_info.json")
    print(f"Composite score stats: mean={np.mean(composite_scores):.3f}, std={np.std(composite_scores):.3f}")


if __name__ == '__main__':
    main()
