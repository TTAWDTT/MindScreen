"""
MindScreen - Flask API 后端服务（smmh 版）
适配 smmh.csv 训练得到的 risk / depressed 模型
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import json
import os

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'models'))
DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', 'smmh.csv'))

required_files = [
    'smmh_risk_pipeline.pkl',
    'smmh_depressed_pipeline.pkl',
    'smmh_model_info.json'
]

print(f"加载模型... (模型目录: {MODEL_DIR})")
missing = [f for f in required_files if not os.path.exists(os.path.join(MODEL_DIR, f))]
if missing:
    raise FileNotFoundError(
        f"缺少模型文件: {missing}\n预期位于: {MODEL_DIR}\n请先运行 backend/train_smmh_models.py 生成模型或复制 models/ 文件夹。"
    )

# Prefer v2 risk pipeline if available
risk_v2_path = os.path.join(MODEL_DIR, 'smmh_risk_pipeline_v2.pkl')
if os.path.exists(risk_v2_path):
    risk_pipeline = joblib.load(risk_v2_path)
    RISK_PIPE_VERSION = 'v2'
else:
    risk_pipeline = joblib.load(os.path.join(MODEL_DIR, 'smmh_risk_pipeline.pkl'))
    RISK_PIPE_VERSION = 'v1'

dep_pipeline = joblib.load(os.path.join(MODEL_DIR, 'smmh_depressed_pipeline.pkl'))

with open(os.path.join(MODEL_DIR, 'smmh_model_info.json'), 'r', encoding='utf-8') as f:
    model_info = json.load(f)

PLATFORMS = model_info.get('platforms', [])
FEATURES = model_info.get('features', [])

# Load reference data for percentile computation (Likert questions)
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


def compute_percentile(val, series: pd.Series):
    try:
        if series is None or series.empty or val is None:
            return None
        # Clamp within observed range, then interpolate on percentiles 0-100
        q = np.nanpercentile(series, np.arange(0, 101))
        pct = np.interp(val, q, np.arange(0, 101))
        pct = float(np.clip(pct, 0, 100))
        return round(pct, 1)
    except Exception:
        return None

baseline_map = model_info.get('likert_baseline', {})

Q_LABELS = {
    'q9': 'Q9 无目的刷社交媒体频率',
    'q10': 'Q10 忙碌时被社交媒体分心频率',
    'q11': 'Q11 一段时间不用社交媒体的焦躁感',
    'q12': 'Q12 容易分心程度',
    'q13': 'Q13 被担忧困扰程度',
    'q14': 'Q14 难以集中注意力频率',
    'q15': 'Q15 与他人比较频率',
    'q16': 'Q16 对比较的感受',
    'q17': 'Q17 寻求社交媒体认可频率',
    'q18': 'Q18 感到沮丧频率',
    'q19': 'Q19 对日常活动兴趣波动频率',
    'q20': 'Q20 睡眠问题频率'
}

try:
    df_ref = pd.read_csv(DATA_PATH)
    df_ref = df_ref.rename(columns=COL_RENAME)
    if not baseline_map:
        # Fallback: compute baseline from training data if not saved in model_info
        for col in LIKERT_COLS:
            if col in df_ref.columns:
                series = df_ref[col]
                raw_mean = float(series.mean()) if series.notnull().any() else None
                pct_mean = compute_percentile(raw_mean, series) if raw_mean is not None else None
                baseline_map[col] = {
                    'baseline_value': raw_mean,
                    'baseline_percentile': pct_mean
                }
            else:
                baseline_map[col] = {
                    'baseline_value': None,
                    'baseline_percentile': None
                }
except Exception:
    df_ref = pd.DataFrame()
    baseline_map = baseline_map or {}

DEFAULTS = {
    'age': 25.0,
    'gender': 'Male',
    'relationship': 'Single',
    'occupation': 'University Student',
    'avg_time_per_day': 'Between 2 and 3 hours'
}

AVG_TIME_ORDER = [
    'Less than an Hour',
    'Between 1 and 2 hours',
    'Between 2 and 3 hours',
    'Between 3 and 4 hours',
    'Between 4 and 5 hours',
    'More than 5 hours'
]

# Simple label encoding mappings (consistent with training)
RELATIONSHIP_MAPPING = {'Single': 0, 'In a relationship': 1, 'Married': 2, 'Divorced': 3}
OCCUPATION_MAPPING = {'University Student': 0, 'School Student': 1, 'Salaried Worker': 2, 'Retired': 3}


def build_feature_row(data: dict):
    """Build feature row matching training schema with plat_*, encodings, and engineered features."""
    age = float(data.get('age', DEFAULTS['age']))
    gender = data.get('gender', DEFAULTS['gender'])
    relationship = data.get('relationship', DEFAULTS['relationship'])
    occupation = data.get('occupation', DEFAULTS['occupation'])
    avg_time = data.get('avg_time_per_day', DEFAULTS['avg_time_per_day'])

    # Platform flags (plat_* prefix to match training)
    platforms_input = data.get('platforms', [])
    if isinstance(platforms_input, str):
        platforms_input = [p.strip() for p in platforms_input.split(',') if p.strip()]
    
    plat_flags = {f'plat_{p}': (1 if p in platforms_input else 0) for p in PLATFORMS}
    platform_count = sum(plat_flags.values())

    # Encode relationship and occupation
    relationship_enc = RELATIONSHIP_MAPPING.get(relationship, 0)
    occupation_enc = OCCUPATION_MAPPING.get(occupation, 0)

    # Encode avg_time_per_day as ordinal
    try:
        avg_time_ord = float(AVG_TIME_ORDER.index(avg_time))
    except (ValueError, AttributeError):
        avg_time_ord = 2.0  # default to middle

    # Get survey responses for digital_addiction_score (Q9-Q12)
    survey = data.get('survey', {}) or {}
    q9 = int(survey.get('q9', 0))
    q10 = int(survey.get('q10', 0))
    q11 = int(survey.get('q11', 0))
    q12 = int(survey.get('q12', 0))
    digital_addiction_score = q9 + q10 + q11 + q12

    # Gender one-hot (match training: gender_Male, gender_Female, gender_other)
    gender_Male = 1 if gender == 'Male' else 0
    gender_Female = 1 if gender == 'Female' else 0
    gender_other = 1 if gender not in ['Male', 'Female'] else 0

    row = {
        'age': age,
        'relationship_enc': relationship_enc,
        'occupation_enc': occupation_enc,
        'avg_time_ord': avg_time_ord,
        'platform_count': platform_count,
        'digital_addiction_score': digital_addiction_score,
        **plat_flags,
        'gender_Male': gender_Male,
        'gender_Female': gender_Female,
        'gender_other': gender_other
    }

    # Order according to FEATURES from model_info
    ordered_row = {feat: row.get(feat, 0) for feat in FEATURES}
    return ordered_row

def predict_with_pipeline(pipe, row):
    df = pd.DataFrame([row])
    pred = pipe.predict(df)[0]
    proba = []
    if hasattr(pipe, 'predict_proba'):
        try:
            probs = pipe.predict_proba(df)[0]
            labels = pipe.classes_
            proba = [
                {'label': str(lbl), 'probability': float(prob)}
                for lbl, prob in sorted(zip(labels, probs), key=lambda x: x[1], reverse=True)
            ]
        except Exception:
            proba = []
    return str(pred), proba


#---------- 以上为工具函数 ----------


#---------- 以下为路由部分 ----------

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'message': 'MindScreen API is running'})


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json or {}
        feature_row = build_feature_row(data)

        risk_pred, risk_proba = predict_with_pipeline(risk_pipeline, feature_row)
        dep_pred, dep_proba = predict_with_pipeline(dep_pipeline, feature_row)

        # Percentile analysis for Likert questions (Q9-Q20) if provided
        survey = data.get('survey', {}) or {}
        percentile_list = []
        if not df_ref.empty:
            survey_map = {
                'q9': 'without_purpose',
                'q10': 'distracted',
                'q11': 'restless',
                'q12': 'distracted_ease',
                'q13': 'worries',
                'q14': 'concentration',
                'q15': 'compare_to_others',
                'q16': 'compare_feelings',
                'q17': 'validation',
                'q18': 'depressed',
                'q19': 'daily_activity_flux',
                'q20': 'sleeping_issues'
            }
            for qid, col in survey_map.items():
                try:
                    val = survey.get(qid)
                    if val is not None:
                        val = float(val)
                    pct = compute_percentile(val, df_ref.get(col))
                    baseline = baseline_map.get(col, {})
                    baseline_pct = baseline.get('baseline_percentile', baseline.get('pct'))
                    baseline_val = baseline.get('baseline_value', baseline.get('raw'))
                    percentile_list.append({
                        'id': qid,
                        'label': Q_LABELS.get(qid, qid),
                        'value': val,
                        'percentile': pct,
                        'baseline_percentile': baseline_pct,
                        'baseline_value': baseline_val
                    })
                except Exception:
                    continue

        # Compute composite mental health score and percentile ranking
        composite_score = None
        composite_percentile = None
        composite_rank_text = None
        
        composite_dist = model_info.get('composite_score_distribution', {})
        if composite_dist and risk_proba and dep_proba:
            try:
                # Get probability of 'higher' risk
                # Model may return numeric labels (0/1) or string labels (higher/lower)
                risk_prob_higher = 0.0
                
                # Debug: print all risk probabilities
                print(f"[DEBUG] Risk probabilities: {risk_proba}")
                
                # Try different label formats
                for item in risk_proba:
                    label = str(item['label']).lower()
                    # Check for 'higher', '1', or if it's the second item (assuming sorted by probability)
                    if 'higher' in label or label == '1':
                        risk_prob_higher = item['probability']
                        break
                
                # If still 0, check if we need to look at the actual prediction
                if risk_prob_higher == 0.0 and len(risk_proba) >= 2:
                    # If risk_pred indicates higher risk, get that probability
                    if str(risk_pred).lower() in ['higher', '1']:
                        # Find the matching probability
                        for item in risk_proba:
                            if str(item['label']) == str(risk_pred):
                                risk_prob_higher = item['probability']
                                break
                    else:
                        # If prediction is 'lower' or '0', use 1 - P(lower)
                        for item in risk_proba:
                            if str(item['label']) == str(risk_pred):
                                risk_prob_higher = 1.0 - item['probability']
                                break
                
                # Get depressed level (convert to numeric 1-5)
                dep_level = 3.0  # default
                
                # Debug: print depression prediction
                print(f"[DEBUG] Depression pred: {dep_pred}, probs: {dep_proba}")
                
                try:
                    # Try to convert prediction to float directly
                    dep_level = float(dep_pred)
                except (ValueError, TypeError):
                    # If it's a string number, extract it
                    dep_str = str(dep_pred).strip()
                    for digit in ['1', '2', '3', '4', '5']:
                        if digit in dep_str:
                            dep_level = float(digit)
                            break
                
                # Debug logging
                print(f"[DEBUG] Composite calculation: risk_higher={risk_prob_higher:.4f}, dep_level={dep_level}")
                
                # Calculate composite score using same formula as training
                composite_score = 0.5 * risk_prob_higher + 0.5 * (dep_level / 5.0)
                
                print(f"[DEBUG] Raw composite score: {composite_score:.4f}")
                
                # Compare with training distribution to get percentile
                percentiles_array = np.array(composite_dist.get('percentiles', []))
                if len(percentiles_array) > 0:
                    # Find percentile by interpolation and clamp to 0-100
                    composite_percentile = float(np.interp(
                        composite_score,
                        percentiles_array,
                        np.arange(0, 101)
                    ))
                    composite_percentile = np.clip(composite_percentile, 0, 100)
                    
                    # Generate ranking text (percentile=分数在训练集中的位置，高分=差)
                    # 分数越高越差，所以百分位高意味着状态差
                    if composite_percentile >= 90:
                        composite_rank_text = "心理风险高于90%用户，建议重点关注心理健康"
                    elif composite_percentile >= 75:
                        composite_rank_text = "心理风险高于75%用户，心理压力较大"
                    elif composite_percentile >= 50:
                        composite_rank_text = "处于中等水平，需要适度调节"
                    elif composite_percentile >= 25:
                        composite_rank_text = "心理风险低于50%用户，状态相对良好"
                    elif composite_percentile >= 10:
                        composite_rank_text = "心理风险低于75%用户，状态良好"
                    else:
                        composite_rank_text = "心理风险低于90%用户，心理状态较佳"
                        
            except Exception as e:
                print(f"Composite score calculation error: {e}")

        response = {
            'predictions': {
                'risk': risk_pred,
                'risk_probs': risk_proba,
                'depressed': dep_pred,
                'depressed_probs': dep_proba
            },
            'composite_score': {
                'score': composite_score,
                'percentile': composite_percentile,
                'rank_description': composite_rank_text,
                'formula': composite_dist.get('formula', 'N/A')
            } if composite_score is not None else None,
            'input_summary': feature_row,
            'model_meta': {
                'risk_model': model_info['risk']['model'],
                'risk_model_version': RISK_PIPE_VERSION,
                'depressed_model': model_info['depressed']['model'],
                'features': FEATURES,
                'platforms': PLATFORMS
            },
            'percentiles': percentile_list
        }

        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/stats', methods=['GET'])
def get_stats():
    return jsonify({'model_info': model_info})


@app.route('/api/training-report', methods=['GET'])
def get_training_report():
    """获取模型训练对比报告"""
    report_path = os.path.join(MODEL_DIR, 'model_comparison_report.json')
    if os.path.exists(report_path):
        with open(report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
        return jsonify(report)
    else:
        # 返回基础信息
        return jsonify({
            'message': '详细训练报告不可用',
            'basic_info': {
                'risk_model': model_info.get('risk', {}).get('model', 'unknown'),
                'depressed_model': model_info.get('depressed', {}).get('model', 'unknown'),
                'dataset_rows': model_info.get('n_rows', 'unknown'),
                'features': model_info.get('features', [])
            }
        })


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("MindScreen API 服务启动 (smmh)")
    print("=" * 50)
    print("访问地址: http://localhost:5000")
    print("API 端点:")
    print("  - GET  /api/health          - 健康检查")
    print("  - POST /api/predict         - 预测分析")
    print("  - GET  /api/stats           - 获取模型信息")
    print("  - GET  /api/training-report - 获取训练报告")
    print("=" * 50 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
