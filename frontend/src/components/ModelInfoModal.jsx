/**
 * 模型说明弹窗组件
 */
import React, { useState, useEffect } from 'react';
import axios from 'axios';

const API_BASE = '/api';

export function ModelInfoModal({ isOpen, onClose }) {
  const [modelInfo, setModelInfo] = useState(null);
  const [loading, setLoading] = useState(false);
  
  useEffect(() => {
    if (isOpen && !modelInfo) {
      setLoading(true);
      axios.get(`${API_BASE}/stats`)
        .then(res => setModelInfo(res.data.model_info))
        .catch(console.error)
        .finally(() => setLoading(false));
    }
  }, [isOpen]);
  
  if (!isOpen) return null;
  
  const riskMetrics = modelInfo?.risk || {};
  const depMetrics = modelInfo?.depressed || {};
  
  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={e => e.stopPropagation()}>
        <button className="modal-close" onClick={onClose}>×</button>
        
        <div className="modal-header">
          <h2>🔬 预测模型原理</h2>
          <p className="modal-subtitle">了解我们如何进行心理健康评估</p>
        </div>
        
        {loading ? (
          <div className="modal-loading">加载中...</div>
        ) : (
          <div className="modal-body">
            {/* 数据来源 */}
            <section className="info-section">
              <h3>📊 数据来源</h3>
              <p>
                本系统使用 <strong>Social Media and Mental Health (SMMH)</strong> 数据集进行训练，
                该数据集包含 {modelInfo?.n_rows || 481} 条真实问卷调查数据，
                涵盖社交媒体使用习惯与心理健康状态的多维度信息。
              </p>
            </section>
            
            {/* 模型架构 */}
            <section className="info-section">
              <h3>🤖 模型架构</h3>
              <div className="model-cards">
                <div className="model-card">
                  <h4>风险预测模型</h4>
                  <div className="model-badge">{riskMetrics.model?.toUpperCase() || 'XGBoost'}</div>
                  <p>预测用户是否处于较高心理健康风险状态（二分类）</p>
                  <ul className="model-metrics">
                    <li>
                      <span>平衡准确率</span>
                      <strong>{(riskMetrics.balanced_accuracy * 100)?.toFixed(1) || 83.7}%</strong>
                    </li>
                    <li>
                      <span>F1-Macro</span>
                      <strong>{(riskMetrics.f1_macro * 100)?.toFixed(1) || 83.5}%</strong>
                    </li>
                  </ul>
                </div>
                
                <div className="model-card">
                  <h4>抑郁等级模型</h4>
                  <div className="model-badge">{depMetrics.model?.toUpperCase() || 'LogReg'}</div>
                  <p>预测用户的抑郁倾向等级（1-5级多分类）</p>
                  <ul className="model-metrics">
                    <li>
                      <span>平衡准确率</span>
                      <strong>{(depMetrics.balanced_accuracy * 100)?.toFixed(1) || 35.7}%</strong>
                    </li>
                    <li>
                      <span>F1-Macro</span>
                      <strong>{(depMetrics.f1_macro * 100)?.toFixed(1) || 35.6}%</strong>
                    </li>
                  </ul>
                </div>
              </div>
            </section>
            
            {/* 特征工程 */}
            <section className="info-section">
              <h3>⚙️ 特征工程</h3>
              <p>模型使用以下特征进行预测：</p>
              <div className="feature-tags">
                {(modelInfo?.features || []).slice(0, 12).map((f, i) => (
                  <span key={i} className="feature-tag">{translateFeature(f)}</span>
                ))}
              </div>
            </section>
            
            {/* 综合评分 */}
            <section className="info-section">
              <h3>📈 综合评分计算</h3>
              <div className="formula-box">
                <code>{modelInfo?.composite_score_distribution?.formula || 'composite = 0.5 × P(风险=高) + 0.5 × (抑郁等级/5)'}</code>
              </div>
              <p>
                综合评分结合风险概率和抑郁等级，范围0-1，分数越高表示心理健康风险越高。
                您的得分将与训练数据中的分布进行比较，得出百分位排名。
              </p>
            </section>
            
            {/* 模型选择理由 */}
            <section className="info-section">
              <h3>🎯 模型选择理由</h3>
              <div className="reason-list">
                <div className="reason-item">
                  <span className="reason-icon">✅</span>
                  <div>
                    <strong>XGBoost (风险模型)</strong>
                    <p>在不平衡数据集上表现优秀，支持自动处理缺失值，抗过拟合能力强</p>
                  </div>
                </div>
                <div className="reason-item">
                  <span className="reason-icon">✅</span>
                  <div>
                    <strong>Logistic Regression (抑郁模型)</strong>
                    <p>对多分类问题有良好的概率校准，可解释性强，计算效率高</p>
                  </div>
                </div>
                <div className="reason-item">
                  <span className="reason-icon">✅</span>
                  <div>
                    <strong>交叉验证</strong>
                    <p>使用3折交叉验证 + GridSearch超参数搜索，确保模型泛化能力</p>
                  </div>
                </div>
              </div>
            </section>
            
            {/* 免责声明 */}
            <section className="info-section disclaimer">
              <h3>⚠️ 重要声明</h3>
              <p>
                本系统提供的评估结果仅供参考，不能替代专业的医疗诊断。
                如果您正在经历严重的心理困扰，请及时寻求专业心理医生的帮助。
              </p>
            </section>
          </div>
        )}
      </div>
    </div>
  );
}

// 特征名称翻译
function translateFeature(feature) {
  const map = {
    'age': '年龄',
    'relationship_enc': '感情状态',
    'occupation_enc': '职业',
    'avg_time_ord': '日均使用时长',
    'platform_count': '平台数量',
    'digital_addiction_score': '数字成瘾评分',
    'gender_Male': '性别(男)',
    'gender_Female': '性别(女)',
    'gender_other': '性别(其他)'
  };
  
  if (feature.startsWith('plat_')) {
    return feature.replace('plat_', '') + '使用';
  }
  
  return map[feature] || feature;
}

export default ModelInfoModal;
