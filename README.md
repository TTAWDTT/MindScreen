# MindScreen - 心理健康智能评估系统

MindScreen 是一个基于机器学习的心理健康评估系统，通过分析用户的社交媒体使用习惯，预测心理健康风险并提供个性化改善建议。

## ✨ 项目特性

- **智能问卷系统**: 支持快速评估(30秒)和完整评估(2分钟)两种模式
- **多维度分析**: 数字成瘾、情绪状态、睡眠质量、社交比较等多维指标
- **机器学习预测**:
  - 心理风险等级预测 (XGBoost/RandomForest)
  - 抑郁倾向评分 (1-5级多分类)
  - 综合健康评分与人群百分位排名
- **可视化报告**: 雷达图、柱状图展示用户在人群中的位置
- **个性化建议**: 根据评估结果生成针对性的改善建议
- **模型透明度**: 可查看模型训练原理和对比结果

## 📂 项目结构

```
MindScreen/
├── backend/                        # Flask 后端服务
│   ├── app.py                      # API 核心入口
│   ├── train_smmh_models.py        # 基础模型训练脚本
│   ├── train_models_comparison.py  # 多模型对比训练脚本
│   └── requirements.txt            # Python 依赖
├── frontend/                       # React + Vite 前端应用
│   ├── src/
│   │   ├── components/             # 可复用组件
│   │   │   ├── QuestionnaireComponents.jsx  # 问卷UI组件
│   │   │   ├── ResultsVisualization.jsx     # 结果可视化组件
│   │   │   └── ModelInfoModal.jsx           # 模型说明弹窗
│   │   ├── config/
│   │   │   └── questions.js        # 问卷配置文件 (集中管理题目)
│   │   ├── pages/
│   │   │   └── ResultsPage.jsx     # 结果展示页面
│   │   ├── utils/
│   │   │   └── advice.js           # 建议生成工具
│   │   ├── App.jsx                 # 主应用组件
│   │   └── index.css               # 全局样式 (浅色玻璃风格)
│   └── ...
├── models/                         # 预训练模型
│   ├── smmh_risk_pipeline_v2.pkl   # 风险预测模型
│   ├── smmh_depressed_pipeline.pkl # 抑郁预测模型
│   ├── smmh_model_info.json        # 模型元数据
│   └── model_comparison_report.json # 模型对比报告
├── smmh.csv                        # 训练数据集
└── README.md
```

## 🛠️ 快速开始

### 1. 环境要求

- Python 3.8+
- Node.js 16+

### 2. 后端设置

```bash
cd backend

# 安装依赖
pip install -r requirements.txt

# (可选) 重新训练模型并对比多种算法
python train_models_comparison.py

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

前端应用将在 `http://localhost:3000` 启动。

## 📋 问卷配置

问卷题目集中管理在 `frontend/src/config/questions.js`，支持：

- **基础信息题目**: 年龄、性别、感情状态、职业
- **社交媒体使用**: 平台选择、日均使用时长
- **Likert量表题目**: 1-5分的心理状态评估

修改此文件即可调整问卷内容，无需修改组件代码。

```javascript
// 示例：添加新题目
export const LIKERT_QUESTIONS = [
  {
    id: 'q_new',
    field: 'q_new',
    label: '新题目标签',
    fullLabel: '完整的问题描述',
    type: 'likert',
    category: 'mental',
    lowLabel: '从不',
    highLabel: '总是'
  },
  // ...
];
```

## 📊 模型说明

### 数据来源
使用 Social Media and Mental Health (SMMH) 数据集，包含 481 条问卷数据。

### 模型架构

| 模型 | 任务 | 算法 | 准确率 |
|------|------|------|--------|
| 风险模型 | 二分类 (高/低风险) | XGBoost | ~83.7% |
| 抑郁模型 | 多分类 (1-5级) | Logistic Regression | ~35.7% |

### 模型选择标准

运行 `train_models_comparison.py` 会自动对比以下算法：
- Logistic Regression
- Random Forest
- Gradient Boosting
- SVM
- XGBoost (可选)
- LightGBM (可选)

根据综合评分选择最佳模型：
```
综合评分 = 平衡准确率 × 0.4 + F1分数 × 0.3 + 交叉验证分数 × 0.3
```

## 🎨 UI 特性

- **浅色玻璃风格**: 现代化毛玻璃效果，柔和的渐变背景
- **响应式设计**: 支持桌面和移动设备
- **交互式可视化**: 悬浮查看详细分析
- **打印友好**: 支持导出/打印评估报告

## 📝 API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/health` | 健康检查 |
| POST | `/api/predict` | 预测分析 |
| GET | `/api/stats` | 获取模型信息 |
| GET | `/api/training-report` | 获取训练对比报告 |

## ⚠️ 注意事项

- 本系统提供的评估结果仅供参考，不能替代专业的医疗诊断
- 如有严重的心理困扰，请及时寻求专业心理医生的帮助

## 🚀 未来功能建议

1. **历史记录追踪**: 保存用户历次评估结果，展示心理健康变化趋势
2. **社区匿名分享**: 用户可匿名分享改善经验
3. **智能提醒**: 根据使用习惯推送减少屏幕时间提醒
4. **多语言支持**: 国际化多语言界面
5. **移动端APP**: React Native 原生应用
6. **API开放**: 提供第三方集成接口

## 📄 许可证

MIT License
