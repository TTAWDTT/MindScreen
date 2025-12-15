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

// 状态图标组件（使用 SVG 代替 emoji）
function StatusIcon({ status, color }) {
  if (status === 'high') {
    return (
      <svg className="status-icon" viewBox="0 0 24 24" fill="none" stroke={color || '#ef4444'} strokeWidth="2">
        <circle cx="12" cy="12" r="10"/>
        <line x1="12" y1="8" x2="12" y2="12"/>
        <circle cx="12" cy="16" r="1" fill={color || '#ef4444'}/>
      </svg>
    );
  }
  if (status === 'low') {
    return (
      <svg className="status-icon" viewBox="0 0 24 24" fill="none" stroke={color || '#10b981'} strokeWidth="2">
        <circle cx="12" cy="12" r="10"/>
        <path d="M8 12l3 3 5-6"/>
      </svg>
    );
  }
  return (
    <svg className="status-icon" viewBox="0 0 24 24" fill="none" stroke={color || '#6366f1'} strokeWidth="2">
      <circle cx="12" cy="12" r="10"/>
      <line x1="8" y1="12" x2="16" y2="12"/>
    </svg>
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
        <div className="indicator-icon-wrapper" style={{ background: `${indicator.color}15` }}>
          <StatusIcon status={indicator.status} color={indicator.color} />
        </div>
        <h4 className="indicator-title">{indicator.label}</h4>
      </div>
      <div className="indicator-content">
        <div className="indicator-score">
          <span className="score-value">{indicator.score?.toFixed(1)}</span>
          <span className="score-label">/5</span>
        </div>
        <div className="indicator-percentile">
          <div className="percentile-track">
            <div 
              className="percentile-fill"
              style={{ 
                width: `${indicator.percentile}%`,
                background: `linear-gradient(90deg, ${statusColors[indicator.status]}80, ${statusColors[indicator.status]})`
              }}
            />
          </div>
          <span className="percentile-text">
            {indicator.statusLabel} · 超过{indicator.percentile?.toFixed(0)}%人群
          </span>
        </div>
      </div>
      
      {/* 悬浮详情 */}
      {isHovered && (
        <div className="indicator-tooltip glass-tooltip">
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
    <div className="chart-container radar-chart">
      {title && <h3 className="chart-title">{title}</h3>}
      <ResponsiveContainer width="100%" height={300}>
        <RadarChart data={chartData}>
          <PolarGrid stroke="rgba(99,102,241,0.1)" />
          <PolarAngleAxis 
            dataKey="subject" 
            tick={{ fill: 'var(--text-secondary)', fontSize: 12 }}
          />
          <PolarRadiusAxis 
            angle={30} 
            domain={[0, 100]} 
            tick={{ fill: 'var(--text-muted)', fontSize: 10 }}
          />
          <Radar
            name="您的得分"
            dataKey="value"
            stroke={COLORS.primary}
            fill="url(#radarGradient)"
            fillOpacity={0.4}
            strokeWidth={2}
          />
          <defs>
            <linearGradient id="radarGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={COLORS.primary} stopOpacity={0.6} />
              <stop offset="100%" stopColor={COLORS.secondary} stopOpacity={0.2} />
            </linearGradient>
          </defs>
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}

// 指标标签映射表
const INDICATOR_LABELS = {
  q9: '无目的浏览',
  q10: '工作学习分心',
  q11: '离线焦躁感',
  q12: '注意力分散',
  q13: '担忧困扰',
  q14: '注意力集中',
  q15: '社交比较',
  q16: '比较后感受',
  q17: '寻求认可',
  q18: '情绪低落',
  q19: '兴趣波动',
  q20: '睡眠问题'
};

// 分布对比图
export function DistributionChart({ percentiles, title }) {
  const chartData = percentiles?.map(p => {
    // 优先使用映射表，然后是label（去掉Q前缀），最后是id
    let displayName = INDICATOR_LABELS[p.id];
    if (!displayName && p.label) {
      displayName = p.label.replace(/^Q\d+\s*/, '').trim();
    }
    if (!displayName) {
      displayName = p.id;
    }
    return {
      name: displayName,
      您的位置: p.percentile || 0,
      人群中位: 50
    };
  }) || [];
  
  return (
    <div className="chart-container distribution-chart">
      {title && <h3 className="chart-title">{title}</h3>}
      <ResponsiveContainer width="100%" height={Math.max(350, chartData.length * 45)}>
        <BarChart 
          data={chartData} 
          layout="vertical"
          margin={{ left: 20, right: 30, top: 10, bottom: 10 }}
        >
          <XAxis 
            type="number" 
            domain={[0, 100]} 
            tick={{ fill: 'var(--text-secondary)', fontSize: 12 }}
            tickFormatter={(v) => `${v}%`}
            axisLine={{ stroke: 'rgba(0,0,0,0.1)' }}
          />
          <YAxis 
            type="category" 
            dataKey="name" 
            width={100}
            tick={{ fill: 'var(--text-primary)', fontSize: 13, fontWeight: 500 }}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip 
            contentStyle={{ 
              background: 'rgba(255,255,255,0.95)', 
              backdropFilter: 'blur(20px)',
              border: '1px solid rgba(255,255,255,0.3)',
              borderRadius: 12,
              boxShadow: '0 8px 32px rgba(0,0,0,0.1)'
            }}
            formatter={(value) => [`${value.toFixed(1)}%`, '']}
            labelStyle={{ color: 'var(--text-primary)', fontWeight: 600 }}
          />
          <Legend 
            wrapperStyle={{ paddingTop: 16 }}
            formatter={(value) => <span style={{ color: 'var(--text-secondary)', fontSize: 13 }}>{value}</span>}
          />
          <Bar 
            dataKey="人群中位" 
            fill="rgba(99,102,241,0.15)" 
            radius={[0, 6, 6, 0]}
            barSize={20}
          />
          <Bar 
            dataKey="您的位置" 
            fill="url(#barGradient)" 
            radius={[0, 6, 6, 0]}
            barSize={20}
          />
          <defs>
            <linearGradient id="barGradient" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor={COLORS.primary} stopOpacity={0.8} />
              <stop offset="100%" stopColor={COLORS.secondary} stopOpacity={1} />
            </linearGradient>
          </defs>
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
  
  // 更新配色以匹配 iOS 风格
  const pieColors = ['#6366f1', '#a5b4fc', '#c4b5fd', '#f0abfc', '#86efac'];
  
  return (
    <div className="chart-container pie-chart">
      {title && <h3 className="chart-title">{title}</h3>}
      <ResponsiveContainer width="100%" height={220}>
        <PieChart>
          <Pie
            data={data}
            dataKey="value"
            nameKey="name"
            cx="50%"
            cy="50%"
            innerRadius={45}
            outerRadius={75}
            paddingAngle={2}
            label={({ name, value }) => `${name} ${value}%`}
            labelLine={{ stroke: 'var(--text-muted)', strokeWidth: 1 }}
          >
            {data.map((entry, idx) => (
              <Cell key={idx} fill={pieColors[idx % pieColors.length]} />
            ))}
          </Pie>
          <Tooltip 
            contentStyle={{ 
              background: 'rgba(255,255,255,0.95)', 
              backdropFilter: 'blur(20px)',
              border: '1px solid rgba(255,255,255,0.3)',
              borderRadius: 10,
              boxShadow: '0 4px 16px rgba(0,0,0,0.08)'
            }}
            formatter={(v) => [`${v}%`, '']}
          />
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
