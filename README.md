# MindScreen - 心理健康智能评估系统

基于机器学习的心理健康评估系统，通过分析用户的屏幕使用习惯和睡眠模式来预测焦虑评分、抑郁评分和睡眠质量。

## 项目结构

```
MindScreen/
├── backend/                    # 后端服务
│   ├── app.py                  # Flask API 服务
│   ├── train_models.py         # 模型训练脚本
│   └── requirements.txt        # Python 依赖
├── frontend/                   # 前端应用 (React + Vite)
│   ├── src/
│   │   ├── App.jsx             # 主应用组件
│   │   ├── main.jsx            # 入口文件
│   │   └── index.css           # Spotify风格样式
│   ├── index.html
│   ├── vite.config.js
│   └── package.json
├── models/                     # 训练好的模型
│   ├── anxiety_model.pkl       # 焦虑评分预测模型
│   ├── depression_model.pkl    # 抑郁评分预测模型
│   ├── sleep_model.pkl         # 睡眠质量预测模型
│   ├── scaler_*.pkl            # 标准化器
│   ├── gender_encoder.pkl      # 性别编码器
│   ├── data_stats.json         # 数据统计信息
│   └── model_info.json         # 模型信息
└── digital_diet_mental_health.csv  # 原始数据集
```

## 功能特性

### 输入参数
1. 年龄
2. 性别
3. 每日总屏幕时间
4. 工作相关时间
5. 娱乐时间
6. 社交媒体时间
7. 睡眠时长
8. 睡眠质量

### 预测输出
- 焦虑评分 (0-20)
- 抑郁评分 (0-20)
- 睡眠质量 (1-10)

### 分析功能
- 各指标在人群中的百分位排名
- 可视化图表（柱状图、雷达图、饼图）
- 问题原因分析与改善建议

## 快速开始

### 1. 安装依赖

**后端:**
```bash
cd backend
pip install -r requirements.txt
```

**前端:**
```bash
cd frontend
npm install
```

### 2. 训练模型（可选，已包含训练好的模型）

```bash
cd backend
python train_models.py
```

### 3. 启动服务

**启动后端 API:**
```bash
cd backend
python app.py
```
后端服务运行在: http://localhost:5000

**启动前端开发服务器:**
```bash
cd frontend
npm run dev
```
前端服务运行在: http://localhost:3000

## API 接口

### 健康检查
```
GET /api/health
```

### 预测分析
```
POST /api/predict
Content-Type: application/json

{
  "age": 25,
  "gender": "Male",
  "daily_screen_time_hours": 6,
  "work_related_hours": 3,
  "entertainment_hours": 2,
  "social_media_hours": 2,
  "sleep_duration_hours": 7,
  "sleep_quality": 5
}
```

响应示例:
```json
{
  "predictions": {
    "anxiety_score": 10.5,
    "depression_score": 9.2,
    "predicted_sleep_quality": 5.8
  },
  "percentile_analysis": {
    "daily_screen_time_hours": {
      "value": 6,
      "percentile": 50,
      "description": "前50%（中等偏低）"
    }
  },
  "cause_analysis": {
    "anxiety": {
      "has_issue": false,
      "message": "您的焦虑评分处于正常范围内"
    }
  }
}
```

### 获取统计数据
```
GET /api/stats
```

### 计算百分位
```
POST /api/percentile
Content-Type: application/json

{
  "metric": "daily_screen_time_hours",
  "value": 8
}
```

## 机器学习模型

本系统测试了以下10种机器学习算法：
- Linear Regression
- Ridge Regression
- Lasso Regression
- ElasticNet
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM
- SVR
- KNN

系统自动选择交叉验证 R² 分数最高的模型进行保存和部署。

## 技术栈

- **后端:** Python, Flask, scikit-learn, XGBoost, LightGBM
- **前端:** React 19, Vite, Recharts, Axios
- **样式:** Spotify 风格深色主题

## 注意事项

⚠️ 本系统仅供参考，不能替代专业医疗诊断。如有心理健康问题，请咨询专业心理健康服务机构。
