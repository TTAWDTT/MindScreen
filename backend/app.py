"""
MindScreen - Flask API 后端服务
提供预测接口和数据分析功能
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import json
import os

app = Flask(__name__)
CORS(app)

# 模型路径
MODEL_DIR = '../models'

# 加载模型和工具
print("加载模型...")
anxiety_model = joblib.load(os.path.join(MODEL_DIR, 'anxiety_model.pkl'))
depression_model = joblib.load(os.path.join(MODEL_DIR, 'depression_model.pkl'))
sleep_model = joblib.load(os.path.join(MODEL_DIR, 'sleep_model.pkl'))

scaler_anxiety = joblib.load(os.path.join(MODEL_DIR, 'scaler_anxiety.pkl'))
scaler_depression = joblib.load(os.path.join(MODEL_DIR, 'scaler_depression.pkl'))
scaler_sleep = joblib.load(os.path.join(MODEL_DIR, 'scaler_sleep.pkl'))

gender_encoder = joblib.load(os.path.join(MODEL_DIR, 'gender_encoder.pkl'))

with open(os.path.join(MODEL_DIR, 'data_stats.json'), 'r', encoding='utf-8') as f:
    data_stats = json.load(f)

with open(os.path.join(MODEL_DIR, 'model_info.json'), 'r', encoding='utf-8') as f:
    model_info = json.load(f)

print("模型加载完成！")

# 特征名称映射
FEATURE_NAMES = {
    'age': '年龄',
    'gender_encoded': '性别',
    'daily_screen_time_hours': '每日总屏幕时间',
    'work_related_hours': '工作相关时间',
    'entertainment_hours': '娱乐时间',
    'social_media_hours': '社交媒体时间',
    'sleep_duration_hours': '睡眠时长',
    'sleep_quality': '睡眠质量'
}

def get_percentile(value, stat_key):
    """计算数值在人群中的百分位"""
    stats = data_stats.get(stat_key, {})
    percentiles = stats.get('percentiles', {})
    
    if value <= percentiles.get('10', 0):
        return 10, "前10%（最低）"
    elif value <= percentiles.get('25', 0):
        return 25, "前25%（较低）"
    elif value <= percentiles.get('50', 0):
        return 50, "前50%（中等偏低）"
    elif value <= percentiles.get('75', 0):
        return 75, "前75%（中等偏高）"
    elif value <= percentiles.get('90', 0):
        return 90, "前90%（较高）"
    else:
        return 100, "前100%（最高）"

def analyze_causes(prediction_type, value, features, importance):
    """分析导致问题的原因"""
    causes = []
    suggestions = []
    
    # 判断是否存在问题
    if prediction_type == 'anxiety':
        stats = data_stats.get('weekly_anxiety_score', {})
        threshold = stats.get('percentiles', {}).get('75', 15)
        is_high = value > threshold
        score_type = "焦虑评分"
    elif prediction_type == 'depression':
        stats = data_stats.get('weekly_depression_score', {})
        threshold = stats.get('percentiles', {}).get('75', 15)
        is_high = value > threshold
        score_type = "抑郁评分"
    else:  # sleep
        stats = data_stats.get('sleep_quality', {})
        threshold = stats.get('percentiles', {}).get('25', 4)
        is_high = value < threshold  # 睡眠质量低分为问题
        score_type = "睡眠质量"
    
    if not is_high:
        return {
            'has_issue': False,
            'message': f"您的{score_type}处于正常范围内，请继续保持良好的生活习惯！",
            'causes': [],
            'suggestions': []
        }
    
    # 分析主要影响因素
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    top_factors = sorted_importance[:3]  # 取前3个重要因素
    
    # 生成原因分析
    age_gender_only = True
    for factor, imp in top_factors:
        if factor == 'age':
            causes.append(f"您的年龄（{features.get('age', 0)}岁）是一个影响因素")
        elif factor == 'gender_encoded':
            causes.append(f"您的性别是一个影响因素")
        elif factor == 'daily_screen_time_hours':
            age_gender_only = False
            screen_time = features.get('daily_screen_time_hours', 0)
            pct, desc = get_percentile(screen_time, 'daily_screen_time_hours')
            if pct >= 75:
                causes.append(f"您的每日屏幕使用时间（{screen_time}小时）较高，处于人群{desc}")
                suggestions.append("建议减少每日屏幕使用时间，尝试安排更多户外活动")
        elif factor == 'work_related_hours':
            age_gender_only = False
            work_time = features.get('work_related_hours', 0)
            pct, desc = get_percentile(work_time, 'work_related_hours')
            if pct >= 75:
                causes.append(f"您的工作相关屏幕时间（{work_time}小时）较高，处于人群{desc}")
                suggestions.append("建议优化工作效率，适当减少连续工作时间，增加休息间隔")
        elif factor == 'entertainment_hours':
            age_gender_only = False
            ent_time = features.get('entertainment_hours', 0)
            pct, desc = get_percentile(ent_time, 'entertainment_hours')
            if pct >= 75:
                causes.append(f"您的娱乐时间（{ent_time}小时）较高，处于人群{desc}")
                suggestions.append("建议控制娱乐时间，尝试更健康的娱乐方式如运动")
        elif factor == 'social_media_hours':
            age_gender_only = False
            social_time = features.get('social_media_hours', 0)
            pct, desc = get_percentile(social_time, 'social_media_hours')
            if pct >= 75:
                causes.append(f"您在社交媒体上花费的时间（{social_time}小时）较高，处于人群{desc}")
                suggestions.append("建议减少社交媒体使用时间，增加面对面社交活动")
        elif factor == 'sleep_duration_hours':
            age_gender_only = False
            sleep_dur = features.get('sleep_duration_hours', 0)
            pct, desc = get_percentile(sleep_dur, 'sleep_duration_hours')
            if pct <= 25:
                causes.append(f"您的睡眠时长（{sleep_dur}小时）较短，处于人群{desc}")
                suggestions.append("建议保证每天7-9小时的睡眠时间")
        elif factor == 'sleep_quality':
            age_gender_only = False
            sleep_qual = features.get('sleep_quality', 0)
            pct, desc = get_percentile(sleep_qual, 'sleep_quality')
            if pct <= 25:
                causes.append(f"您的睡眠质量评分（{sleep_qual}）较低，处于人群{desc}")
                suggestions.append("建议改善睡眠环境，保持规律作息，睡前避免使用电子设备")
    
    if age_gender_only and len(causes) > 0:
        return {
            'has_issue': True,
            'message': f"您的{score_type}偏高，主要与您的年龄和性别相关，这属于该人群的正常现象。建议保持健康的生活方式，定期进行心理健康评估。",
            'causes': causes,
            'suggestions': ["保持规律作息", "适当运动", "如有需要可咨询专业人士"]
        }
    
    return {
        'has_issue': True,
        'message': f"您的{score_type}偏高，以下是可能的原因：",
        'causes': causes if causes else ["多种因素综合影响"],
        'suggestions': suggestions if suggestions else ["建议保持健康的生活方式，合理安排屏幕使用时间"]
    }

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({'status': 'ok', 'message': 'MindScreen API is running'})

@app.route('/api/predict', methods=['POST'])
def predict():
    """预测接口"""
    try:
        data = request.json
        
        # 提取输入特征
        age = float(data.get('age', 25))
        gender = data.get('gender', 'Male')
        daily_screen_time = float(data.get('daily_screen_time_hours', 6))
        work_related = float(data.get('work_related_hours', 2))
        entertainment = float(data.get('entertainment_hours', 2))
        social_media = float(data.get('social_media_hours', 2))
        sleep_duration = float(data.get('sleep_duration_hours', 7))
        sleep_quality = float(data.get('sleep_quality', 5))
        
        # 性别编码
        try:
            gender_encoded = gender_encoder.transform([gender])[0]
        except:
            gender_encoded = 1  # 默认Male
        
        # 准备特征
        features_dict = {
            'age': age,
            'gender_encoded': gender_encoded,
            'daily_screen_time_hours': daily_screen_time,
            'work_related_hours': work_related,
            'entertainment_hours': entertainment,
            'social_media_hours': social_media,
            'sleep_duration_hours': sleep_duration,
            'sleep_quality': sleep_quality
        }
        
        # 焦虑评分预测
        X_anxiety = np.array([[age, gender_encoded, daily_screen_time, work_related, 
                              entertainment, social_media, sleep_duration, sleep_quality]])
        X_anxiety_scaled = scaler_anxiety.transform(X_anxiety)
        anxiety_score = float(anxiety_model.predict(X_anxiety_scaled)[0])
        anxiety_score = max(0, min(20, anxiety_score))  # 限制范围
        
        # 抑郁评分预测
        X_depression = np.array([[age, gender_encoded, daily_screen_time, work_related,
                                 entertainment, social_media, sleep_duration, sleep_quality]])
        X_depression_scaled = scaler_depression.transform(X_depression)
        depression_score = float(depression_model.predict(X_depression_scaled)[0])
        depression_score = max(0, min(20, depression_score))  # 限制范围
        
        # 睡眠质量预测
        X_sleep = np.array([[daily_screen_time, work_related, entertainment, social_media]])
        X_sleep_scaled = scaler_sleep.transform(X_sleep)
        predicted_sleep = float(sleep_model.predict(X_sleep_scaled)[0])
        predicted_sleep = max(1, min(10, predicted_sleep))  # 限制范围
        
        # 计算各指标的统计区间
        percentile_info = {}
        for key in ['daily_screen_time_hours', 'work_related_hours', 'entertainment_hours',
                    'social_media_hours', 'sleep_duration_hours', 'sleep_quality']:
            value = features_dict.get(key, 0)
            pct, desc = get_percentile(value, key)
            percentile_info[key] = {
                'value': value,
                'percentile': pct,
                'description': desc,
                'stats': data_stats.get(key, {})
            }
        
        # 分析原因
        anxiety_analysis = analyze_causes('anxiety', anxiety_score, features_dict, 
                                         model_info['anxiety']['feature_importance'])
        depression_analysis = analyze_causes('depression', depression_score, features_dict,
                                            model_info['depression']['feature_importance'])
        sleep_analysis = analyze_causes('sleep', predicted_sleep, features_dict,
                                       model_info['sleep']['feature_importance'])
        
        # 构建响应
        response = {
            'predictions': {
                'anxiety_score': round(anxiety_score, 2),
                'depression_score': round(depression_score, 2),
                'predicted_sleep_quality': round(predicted_sleep, 2),
                'actual_sleep_quality': sleep_quality
            },
            'percentile_analysis': percentile_info,
            'cause_analysis': {
                'anxiety': anxiety_analysis,
                'depression': depression_analysis,
                'sleep': sleep_analysis
            },
            'input_summary': {
                'age': age,
                'gender': gender,
                'daily_screen_time_hours': daily_screen_time,
                'work_related_hours': work_related,
                'entertainment_hours': entertainment,
                'social_media_hours': social_media,
                'sleep_duration_hours': sleep_duration,
                'sleep_quality': sleep_quality
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """获取统计数据"""
    return jsonify({
        'data_stats': data_stats,
        'model_info': {
            'anxiety_model': model_info['anxiety']['model_name'],
            'depression_model': model_info['depression']['model_name'],
            'sleep_model': model_info['sleep']['model_name'],
            'feature_importance': {
                'anxiety': model_info['anxiety']['feature_importance'],
                'depression': model_info['depression']['feature_importance'],
                'sleep': model_info['sleep']['feature_importance']
            }
        }
    })

@app.route('/api/percentile', methods=['POST'])
def calculate_percentile():
    """计算单个值的百分位"""
    try:
        data = request.json
        metric = data.get('metric')
        value = float(data.get('value', 0))
        
        if metric not in data_stats:
            return jsonify({'error': f'Unknown metric: {metric}'}), 400
        
        pct, desc = get_percentile(value, metric)
        return jsonify({
            'metric': metric,
            'value': value,
            'percentile': pct,
            'description': desc,
            'stats': data_stats[metric]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    print("\n" + "="*50)
    print("MindScreen API 服务启动")
    print("="*50)
    print("访问地址: http://localhost:5000")
    print("API 端点:")
    print("  - GET  /api/health    - 健康检查")
    print("  - POST /api/predict   - 预测分析")
    print("  - GET  /api/stats     - 获取统计数据")
    print("  - POST /api/percentile - 计算百分位")
    print("="*50 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
