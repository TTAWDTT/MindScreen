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
        <div className="results-hero" style={{ borderColor: summary?.color }}>
          <div className="hero-content">
            <div className="hero-icon">
              {summary?.level === 'good' ? '✨' : summary?.level === 'critical' ? '⚠️' : '📊'}
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
      <section className="results-section">
        <h3 className="section-title">
          <span className="title-icon">📊</span>
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
      <section className="results-section">
        <h3 className="section-title">
          <span className="title-icon">🎯</span>
          预测结果
        </h3>
        
        <div className="prediction-cards">
          <div className="prediction-card">
            <div className="card-header">
              <span className="card-icon">🧩</span>
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
          
          <div className="prediction-card">
            <div className="card-header">
              <span className="card-icon">💭</span>
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
        <section className="results-section">
          <h3 className="section-title">
            <span className="title-icon">📈</span>
            各项指标在人群中的位置
          </h3>
          <DistributionChart
            percentiles={percentiles}
            title=""
          />
        </section>
      )}
      
      {/* 个性化建议 */}
      <section className="results-section advice-section">
        <h3 className="section-title">
          <span className="title-icon">💡</span>
          个性化建议
        </h3>
        
        <div className="advice-list">
          {advice.map((item, idx) => (
            <div 
              key={idx} 
              className={`advice-card priority-${item.priority || 'medium'}`}
            >
              <span className="advice-icon">{item.icon}</span>
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
          <div className="help-resources">
            <h4>🆘 如需帮助，请联系：</h4>
            <div className="resource-list">
              {HELP_RESOURCES.map((resource, idx) => (
                <div key={idx} className="resource-item">
                  <span className="resource-icon">{resource.icon}</span>
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
        <button className="btn-secondary" onClick={onRetry}>
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
