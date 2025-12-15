/**
 * 结果页面组件
 */
import React, { useState } from 'react';
import {
  IndicatorCard,
  RadarChartComponent,
  DistributionChart,
  RiskPieChart,
  CompositeScoreDisplay
} from '../components/ResultsVisualization';
import { generateAdvice, calculateKeyIndicators, generateSummary, HELP_RESOURCES } from '../utils/advice';

export function ResultsPage({ results, surveyAnswers, onRetry }) {
  const [hoveredIndicator, setHoveredIndicator] = useState(null);
  
  if (!results) return null;
  
  const predictions = results.predictions || {};
  const composite = results.composite_score;
  const percentiles = results.percentiles || [];
  
  // 计算关键指标
  const keyIndicators = calculateKeyIndicators(surveyAnswers, percentiles);
  
  // 生成建议
  const advice = generateAdvice(results, surveyAnswers);
  
  // 生成综合评价
  const summary = generateSummary(results);
  
  // 翻译标签
  const translateLabel = (label) => {
    const map = { higher: '较高风险', lower: '较低风险' };
    return map[String(label)] || String(label);
  };
  
  return (
    <div className="results-page">
      {/* 综合评分卡片 */}
      {composite && (
        <div className="results-hero glass-card" style={{ borderColor: summary?.color }}>
          <div className="hero-content">
            <div className="hero-icon-wrapper" style={{ background: `${summary?.color}15` }}>
              <svg className="hero-icon" viewBox="0 0 24 24" fill="none" stroke={summary?.color} strokeWidth="1.5">
                {summary?.level === 'good' ? (
                  <><circle cx="12" cy="12" r="10"/><path d="M8 12l3 3 5-6"/></>
                ) : summary?.level === 'critical' ? (
                  <><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><circle cx="12" cy="16" r="1" fill={summary?.color}/></>
                ) : (
                  <><circle cx="12" cy="12" r="10"/><path d="M8 14s1.5 2 4 2 4-2 4-2"/><line x1="9" y1="9" x2="9.01" y2="9"/><line x1="15" y1="9" x2="15.01" y2="9"/></>
                )}
              </svg>
            </div>
            <div className="hero-info">
              <h2 className="hero-title">{summary?.title || '评估完成'}</h2>
              <p className="hero-message">{summary?.message}</p>
            </div>
            <CompositeScoreDisplay
              score={composite.score}
              percentile={composite.percentile}
              description={composite.rank_description}
            />
          </div>
        </div>
      )}
      
      {/* 关键指标区域 */}
      <section className="results-section glass-card">
        <h3 className="section-title">
          关键指标分析
        </h3>
        <p className="section-desc">将鼠标悬停在指标上查看详细分析</p>
        
        <div className="indicators-grid">
          {keyIndicators.map((indicator, idx) => (
            <IndicatorCard
              key={indicator.id}
              indicator={indicator}
              onHover={setHoveredIndicator}
            />
          ))}
        </div>
        
        {/* 悬浮时显示雷达图 */}
        {hoveredIndicator && (
          <div className="hover-chart">
            <RadarChartComponent
              data={keyIndicators}
              title="各指标综合分布"
            />
          </div>
        )}
      </section>
      
      {/* 预测结果 */}
      <section className="results-section glass-card">
        <h3 className="section-title">
          预测结果
        </h3>
        
        <div className="prediction-cards">
          <div className="prediction-card glass-inner">
            <div className="card-header">
              <div className="card-icon-wrapper">
                <svg className="card-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2z"/>
                  <path d="M12 6v6l4 2"/>
                </svg>
              </div>
              <h4>心理健康风险</h4>
            </div>
            <div 
              className={`prediction-result ${predictions.risk === 'higher' ? 'high' : 'low'}`}
            >
              {translateLabel(predictions.risk)}
            </div>
            <RiskPieChart
              probabilities={predictions.risk_probs}
              title=""
            />
          </div>
          
          <div className="prediction-card glass-inner">
            <div className="card-header">
              <div className="card-icon-wrapper">
                <svg className="card-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <circle cx="12" cy="12" r="10"/>
                  <path d="M8 15s1.5 2 4 2 4-2 4-2"/>
                  <line x1="9" y1="9" x2="9.01" y2="9"/>
                  <line x1="15" y1="9" x2="15.01" y2="9"/>
                </svg>
              </div>
              <h4>抑郁倾向等级</h4>
            </div>
            <div className="prediction-result level">
              <span className="level-number">{predictions.depressed}</span>
              <span className="level-max">/5</span>
            </div>
            <RiskPieChart
              probabilities={predictions.depressed_probs}
              title=""
            />
          </div>
        </div>
      </section>
      
      {/* 详细数据分布 */}
      {percentiles.length > 0 && (
        <section className="results-section glass-card">
          <h3 className="section-title">
            各项指标在人群中的位置
          </h3>
          <DistributionChart
            percentiles={percentiles}
            title=""
          />
        </section>
      )}
      
      {/* 个性化建议 */}
      <section className="results-section advice-section glass-card">
        <h3 className="section-title">
          个性化建议
        </h3>
        
        <div className="advice-list">
          {advice.map((item, idx) => (
            <div 
              key={idx} 
              className={`advice-card glass-inner priority-${item.priority || 'medium'}`}
            >
              <div className="advice-icon-wrapper">
                <span className="advice-icon-dot" style={{ 
                  background: item.priority === 'critical' ? 'var(--danger)' : 
                              item.priority === 'high' ? 'var(--warning)' : 'var(--primary)'
                }}></span>
              </div>
              <div className="advice-content">
                <h4>{item.title}</h4>
                <p>{item.content}</p>
              </div>
              {item.priority === 'critical' && (
                <span className="priority-badge">重要</span>
              )}
            </div>
          ))}
        </div>
        
        {/* 如果风险较高，显示求助资源 */}
        {(predictions.risk === 'higher' || Number(predictions.depressed) >= 4) && (
          <div className="help-resources glass-inner">
            <h4>如需帮助，请联系：</h4>
            <div className="resource-list">
              {HELP_RESOURCES.map((resource, idx) => (
                <div key={idx} className="resource-item">
                  <div className="resource-icon-wrapper">
                    <svg className="resource-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                      <path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z"/>
                    </svg>
                  </div>
                  <div>
                    <strong>{resource.name}</strong>
                    <span className="resource-number">{resource.number}</span>
                    <span className="resource-available">{resource.available}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </section>
      
      {/* 操作按钮 */}
      <div className="results-actions">
        <button className="btn-secondary glass-btn" onClick={onRetry}>
          重新测试
        </button>
        <button className="btn-primary" onClick={() => window.print()}>
          导出报告
        </button>
      </div>
    </div>
  );
}

export default ResultsPage;
