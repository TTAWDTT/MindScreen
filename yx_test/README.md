# MindScreen - 心理健康智能评估系统

MindScreen 是一个基于机器学习的心理健康评估系统，旨在通过分析用户的社交媒体使用习惯、睡眠模式及个人基本信息，预测用户的心理健康风险（焦虑、抑郁倾向）和睡眠质量，并提供个性化的改善建议。

## 🚀 项目特性

- **多维度评估**: 结合社交媒体使用时长、平台偏好、睡眠质量等多维度数据。
- **智能预测**:
  - **心理风险等级**: 基于 XGBoost/RandomForest 的二分类模型。
  - **抑郁程度评分**: 1-5 级多分类预测。
  - **综合健康评分**: 结合风险概率与抑郁等级的加权评分。
- **可视化报告**: 提供直观的雷达图、柱状图展示用户在人群中的百分位排名。
- **个性化建议**: 根据评估结果生成针对性的心理健康改善建议。

## 📂 项目结构

```
MindScreen/
├── backend/                    # Flask 后端服务
│   ├── app.py                  # API 核心入口
│   ├── train_smmh_models.py    # 模型训练脚本 (基于 smmh.csv)
│   └── requirements.txt        # Python 依赖列表
├── frontend/                   # React + Vite 前端应用
│   ├── src/                    # 源代码
│   └── ...
├── models/                     # 预训练模型仓库
│   ├── smmh_risk_pipeline.pkl      # 风险预测模型
│   ├── smmh_depressed_pipeline.pkl # 抑郁预测模型
│   └── smmh_model_info.json        # 模型元数据
├── smmh.csv                    # 训练数据集 (Social Media & Mental Health)
└── archive/                    # 归档的历史文件
```

## 🛠️ 快速开始

### 1. 环境准备

确保您的系统已安装：
- Python 3.8+
- Node.js 16+

### 2. 后端设置

```bash
cd backend

# 安装依赖
pip install -r requirements.txt

# (可选) 重新训练模型
# python train_smmh_models.py

# 启动 API 服务
python app.py
```
后端服务将在 `http://localhost:5000` 启动。

### 3. 前端设置

```bash
cd frontend

# 安装依赖
npm install

# 启动开发服务器
npm run dev
```
前端应用将在 `http://localhost:3000` (或终端提示的其他端口) 启动。

## 📊 模型说明

本项目使用 `smmh.csv` 数据集进行训练，主要包含以下模型：

1.  **Risk Model (风险模型)**:
    -   **目标**: 预测用户是否处于高心理健康风险状态。
    -   **算法**: 集成学习 (Ensemble Learning)。
    -   **输入**: 年龄、性别、社交媒体使用时长、平台类型、各类心理量表评分。

2.  **Depression Model (抑郁模型)**:
    -   **目标**: 预测用户的抑郁倾向等级 (1-5)。
    -   **算法**: 多分类器。

## 📝 注意事项

- 本系统提供的评估结果仅供参考，不能替代专业的医疗诊断。
- 如有严重的心理困扰，请及时寻求专业心理医生的帮助。

## 📄 许可证

MIT License
