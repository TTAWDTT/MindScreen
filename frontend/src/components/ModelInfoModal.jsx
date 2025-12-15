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
      <div className="modal-content glass-card" onClick={e => e.stopPropagation()}>
        <button className="modal-close" onClick={onClose}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <line x1="18" y1="6" x2="6" y2="18"/>
            <line x1="6" y1="6" x2="18" y2="18"/>
          </svg>
        </button>
        
        <div className="modal-header">
          <div className="modal-icon-wrapper">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <circle cx="12" cy="12" r="10"/>
              <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/>
              <line x1="12" y1="17" x2="12.01" y2="17"/>
            </svg>
          </div>
          <h2>预测模型原理</h2>
          <p className="modal-subtitle">了解我们如何进行心理健康评估</p>
        </div>
        
        {loading ? (
          <div className="modal-loading">
            <span className="loading-spinner"></span>
            加载中...
          </div>
        ) : (
          <div className="modal-body">
            {/* 数据来源 */}
            <section className="info-section glass-inner">
              <div className="info-header">
                <div className="info-icon-wrapper">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                    <ellipse cx="12" cy="5" rx="9" ry="3"/>
                    <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/>
                    <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>
                  </svg>
                </div>
                <h3>数据来源</h3>
              </div>
              <p>
                本系统使用 <strong>Social Media and Mental Health (SMMH)</strong> 数据集进行训练，
                该数据集包含 {modelInfo?.n_rows || 481} 条真实问卷调查数据，
                涵盖社交媒体使用习惯与心理健康状态的多维度信息。
              </p>
            </section>
            
            {/* 模型架构 */}
            <section className="info-section">
              <div className="info-header">
                <div className="info-icon-wrapper">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                    <rect x="3" y="3" width="7" height="7"/>
                    <rect x="14" y="3" width="7" height="7"/>
                    <rect x="14" y="14" width="7" height="7"/>
                    <rect x="3" y="14" width="7" height="7"/>
                  </svg>
                </div>
                <h3>模型架构</h3>
              </div>
              <div className="model-cards">
                <div className="model-card glass-inner">
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
                
                <div className="model-card glass-inner">
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
            <section className="info-section glass-inner">
              <div className="info-header">
                <div className="info-icon-wrapper">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                    <circle cx="12" cy="12" r="3"/>
                    <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/>
                  </svg>
                </div>
                <h3>特征工程</h3>
              </div>
              <p>模型使用以下特征进行预测：</p>
              <div className="feature-tags">
                {(modelInfo?.features || []).slice(0, 12).map((f, i) => (
                  <span key={i} className="feature-tag">{translateFeature(f)}</span>
                ))}
              </div>
            </section>
            
            {/* 综合评分 */}
            <section className="info-section glass-inner">
              <div className="info-header">
                <div className="info-icon-wrapper">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                    <line x1="18" y1="20" x2="18" y2="10"/>
                    <line x1="12" y1="20" x2="12" y2="4"/>
                    <line x1="6" y1="20" x2="6" y2="14"/>
                  </svg>
                </div>
                <h3>综合评分计算</h3>
              </div>
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
              <div className="info-header">
                <div className="info-icon-wrapper">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                    <circle cx="12" cy="12" r="10"/>
                    <polyline points="12 6 12 12 16 14"/>
                  </svg>
                </div>
                <h3>模型选择理由</h3>
              </div>
              <div className="reason-list">
                <div className="reason-item glass-inner">
                  <div className="reason-icon-wrapper">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <polyline points="20 6 9 17 4 12"/>
                    </svg>
                  </div>
                  <div>
                    <strong>XGBoost (风险模型)</strong>
                    <p>在不平衡数据集上表现优秀，支持自动处理缺失值，抗过拟合能力强</p>
                  </div>
                </div>
                <div className="reason-item glass-inner">
                  <div className="reason-icon-wrapper">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <polyline points="20 6 9 17 4 12"/>
                    </svg>
                  </div>
                  <div>
                    <strong>Logistic Regression (抑郁模型)</strong>
                    <p>对多分类问题有良好的概率校准，可解释性强，计算效率高</p>
                  </div>
                </div>
                <div className="reason-item glass-inner">
                  <div className="reason-icon-wrapper">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <polyline points="20 6 9 17 4 12"/>
                    </svg>
                  </div>
                  <div>
                    <strong>交叉验证</strong>
                    <p>使用3折交叉验证 + GridSearch超参数搜索，确保模型泛化能力</p>
                  </div>
                </div>
              </div>
            </section>
            
            {/* 免责声明 */}
            <section className="info-section disclaimer glass-inner">
              <div className="info-header">
                <div className="info-icon-wrapper warning">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                    <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
                    <line x1="12" y1="9" x2="12" y2="13"/>
                    <line x1="12" y1="17" x2="12.01" y2="17"/>
                  </svg>
                </div>
                <h3>重要声明</h3>
              </div>
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
