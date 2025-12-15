/**
 * 结果可视化组件
 */
import React, { useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, RadarChart, Radar, PolarGrid,
  PolarAngleAxis, PolarRadiusAxis, Legend
} from 'recharts';

// 图表颜色
const COLORS = {
  primary: '#6366f1',
  secondary: '#8b5cf6',
  success: '#10b981',
  warning: '#f59e0b',
  danger: '#ef4444',
  info: '#3b82f6',
  chart: ['#6366f1', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981']
};

// 百分位条形图
export function PercentileBar({ label, value, baseline = 50, color = COLORS.primary }) {
  const isAboveBaseline = value > baseline;
  
  return (
    <div className="percentile-bar-container">
      <div className="percentile-bar-header">
        <span className="percentile-label">{label}</span>
        <span className={`percentile-value ${isAboveBaseline ? 'high' : 'low'}`}>
          {value?.toFixed(1)}%
        </span>
      </div>
      <div className="percentile-bar-track">
        {/* 基准线 */}
        <div 
          className="percentile-baseline" 
          style={{ left: `${baseline}%` }}
        />
        {/* 当前值 */}
        <div 
          className="percentile-bar-fill"
          style={{ 
            width: `${value}%`,
            backgroundColor: isAboveBaseline ? COLORS.warning : COLORS.success
          }}
        />
      </div>
      <div className="percentile-bar-labels">
        <span>低</span>
        <span>人群平均</span>
        <span>高</span>
      </div>
    </div>
  );
}

// 关键指标卡片
export function IndicatorCard({ indicator, onHover }) {
  const [isHovered, setIsHovered] = useState(false);
  
  const statusColors = {
    low: COLORS.success,
    medium: COLORS.warning,
    high: COLORS.danger
  };
  
  return (
    <div 
      className={`indicator-card ${indicator.status}`}
      onMouseEnter={() => { setIsHovered(true); onHover?.(indicator); }}
      onMouseLeave={() => { setIsHovered(false); onHover?.(null); }}
    >
      <div className="indicator-header">
        <span className="indicator-icon" style={{ color: indicator.color }}>
          {indicator.status === 'high' ? '⚠️' : indicator.status === 'low' ? '✅' : '📊'}
        </span>
        <h4 className="indicator-title">{indicator.label}</h4>
      </div>
      <div className="indicator-content">
        <div className="indicator-score">
          <span className="score-value">{indicator.score?.toFixed(1)}</span>
          <span className="score-label">/5</span>
        </div>
        <div className="indicator-percentile">
          <div 
            className="percentile-mini-bar"
            style={{ 
              background: `linear-gradient(to right, ${statusColors[indicator.status]} ${indicator.percentile}%, rgba(255,255,255,0.1) ${indicator.percentile}%)`
            }}
          />
          <span className="percentile-text">
            {indicator.statusLabel} · 超过{indicator.percentile?.toFixed(0)}%人群
          </span>
        </div>
      </div>
      
      {/* 悬浮详情 */}
      {isHovered && (
        <div className="indicator-tooltip">
          <h5>详细分析</h5>
          <p>您的{indicator.label}得分为 {indicator.score?.toFixed(1)}/5</p>
          <p>在人群中处于第 {indicator.percentile?.toFixed(0)} 百分位</p>
          <p className="tooltip-hint">
            {indicator.percentile >= 75 
              ? '建议重点关注此项指标' 
              : indicator.percentile >= 50
              ? '处于平均水平，可适当改善'
              : '表现良好，继续保持'}
          </p>
        </div>
      )}
    </div>
  );
}

// 雷达图组件
export function RadarChartComponent({ data, title }) {
  const chartData = data.map(d => ({
    subject: d.label,
    value: d.percentile || 50,
    fullMark: 100
  }));
  
  return (
    <div className="chart-container">
      {title && <h3 className="chart-title">{title}</h3>}
      <ResponsiveContainer width="100%" height={300}>
        <RadarChart data={chartData}>
          <PolarGrid stroke="rgba(255,255,255,0.1)" />
          <PolarAngleAxis 
            dataKey="subject" 
            tick={{ fill: 'rgba(255,255,255,0.7)', fontSize: 11 }}
          />
          <PolarRadiusAxis 
            angle={30} 
            domain={[0, 100]} 
            tick={{ fill: 'rgba(255,255,255,0.5)' }}
          />
          <Radar
            name="您的得分"
            dataKey="value"
            stroke={COLORS.primary}
            fill={COLORS.primary}
            fillOpacity={0.3}
          />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}

// 分布对比图
export function DistributionChart({ percentiles, title }) {
  const chartData = percentiles?.map(p => ({
    name: p.label?.replace(/Q\d+\s*/, '') || p.id,
    您的得分: p.percentile || 0,
    人群平均: p.baseline_percentile || 50
  })) || [];
  
  return (
    <div className="chart-container">
      {title && <h3 className="chart-title">{title}</h3>}
      <ResponsiveContainer width="100%" height={350}>
        <BarChart 
          data={chartData} 
          layout="vertical"
          margin={{ left: 10, right: 30 }}
        >
          <XAxis type="number" domain={[0, 100]} tick={{ fill: 'rgba(255,255,255,0.6)' }} />
          <YAxis 
            type="category" 
            dataKey="name" 
            width={140}
            tick={{ fill: 'rgba(255,255,255,0.8)', fontSize: 11 }}
          />
          <Tooltip 
            contentStyle={{ 
              background: 'rgba(30,30,46,0.95)', 
              border: '1px solid rgba(255,255,255,0.1)',
              borderRadius: 8
            }}
            formatter={(value) => `${value.toFixed(1)}%`}
          />
          <Legend />
          <Bar dataKey="人群平均" fill="rgba(255,255,255,0.15)" radius={[0, 4, 4, 0]} />
          <Bar dataKey="您的得分" fill={COLORS.primary} radius={[0, 4, 4, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// 风险概率饼图
export function RiskPieChart({ probabilities, title }) {
  const data = probabilities?.map((p, idx) => ({
    name: translateLabel(p.label),
    value: Math.round(p.probability * 1000) / 10
  })) || [];
  
  return (
    <div className="chart-container">
      {title && <h3 className="chart-title">{title}</h3>}
      <ResponsiveContainer width="100%" height={250}>
        <PieChart>
          <Pie
            data={data}
            dataKey="value"
            nameKey="name"
            cx="50%"
            cy="50%"
            innerRadius={50}
            outerRadius={80}
            paddingAngle={3}
            label={({ name, value }) => `${name}: ${value}%`}
          >
            {data.map((entry, idx) => (
              <Cell key={idx} fill={COLORS.chart[idx % COLORS.chart.length]} />
            ))}
          </Pie>
          <Tooltip formatter={(v) => `${v}%`} />
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
}

// 综合评分展示
export function CompositeScoreDisplay({ score, percentile, description }) {
  const getScoreColor = (pct) => {
    if (pct >= 75) return COLORS.danger;
    if (pct >= 50) return COLORS.warning;
    if (pct >= 25) return COLORS.info;
    return COLORS.success;
  };
  
  const color = getScoreColor(percentile || 50);
  
  return (
    <div className="composite-score-display">
      <div className="score-circle" style={{ borderColor: color }}>
        <span className="score-number" style={{ color }}>
          {(score * 100).toFixed(0)}
        </span>
        <span className="score-max">/100</span>
      </div>
      <div className="score-info">
        <div className="score-percentile">
          {percentile >= 50 
            ? `心理风险超过 ${percentile.toFixed(0)}% 用户`
            : `心理状态优于 ${(100 - percentile).toFixed(0)}% 用户`
          }
        </div>
        {description && (
          <div className="score-description">{description}</div>
        )}
      </div>
    </div>
  );
}

// 标签翻译
function translateLabel(label) {
  const map = {
    higher: '较高风险',
    lower: '较低风险',
    '1': '等级1',
    '2': '等级2', 
    '3': '等级3',
    '4': '等级4',
    '5': '等级5'
  };
  return map[String(label)] || String(label);
}

export default {
  PercentileBar,
  IndicatorCard,
  RadarChartComponent,
  DistributionChart,
  RiskPieChart,
  CompositeScoreDisplay
};
