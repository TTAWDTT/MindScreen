import React, { useState } from 'react';
import axios from 'axios';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar,
  PieChart, Pie, Cell
} from 'recharts';

const API_BASE = '/api';

// 特征名称映射
const FEATURE_NAMES = {
  'daily_screen_time_hours': '每日屏幕时间',
  'work_related_hours': '工作相关时间',
  'entertainment_hours': '娱乐时间',
  'social_media_hours': '社交媒体时间',
  'sleep_duration_hours': '睡眠时长',
  'sleep_quality': '睡眠质量'
};

const CHART_COLORS = ['#1db954', '#ff6b6b', '#845ef7', '#4ecdc4', '#ffd93d', '#6bcb77'];

function App() {
  const [formData, setFormData] = useState({
    age: 25,
    gender: 'Male',
    daily_screen_time_hours: 6,
    work_related_hours: 3,
    entertainment_hours: 2,
    social_media_hours: 2,
    sleep_duration_hours: 7,
    sleep_quality: 5
  });

  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleInputChange = (e) => {
    const { name, value, type } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'number' || type === 'range' ? parseFloat(value) : value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const response = await axios.post(`${API_BASE}/predict`, formData);
      setResults(response.data);
    } catch (err) {
      setError(err.response?.data?.error || '预测失败，请稍后重试');
    } finally {
      setLoading(false);
    }
  };

  // 准备百分位图表数据
  const preparePercentileData = () => {
    if (!results?.percentile_analysis) return [];
    return Object.entries(results.percentile_analysis).map(([key, data]) => ({
      name: FEATURE_NAMES[key] || key,
      value: data.value,
      percentile: data.percentile,
      fullMark: 100
    }));
  };

  // 准备雷达图数据
  const prepareRadarData = () => {
    if (!results?.percentile_analysis) return [];
    return Object.entries(results.percentile_analysis).map(([key, data]) => ({
      subject: FEATURE_NAMES[key] || key,
      A: data.percentile,
      fullMark: 100
    }));
  };

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-content">
          <div className="logo">
            <div className="logo-icon">🧠</div>
            <div>
              <h1>MindScreen</h1>
              <span>心理健康智能评估系统</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="main-content">
        {/* Welcome Section */}
        {!results && (
          <section className="welcome-section">
            <h2>了解您的数字健康</h2>
            <p>
              通过分析您的屏幕使用习惯和睡眠模式，我们可以帮助您了解这些因素如何影响您的心理健康，
              并提供个性化的改善建议。
            </p>
          </section>
        )}

        {/* Input Form */}
        <section className="form-section">
          <h2 className="section-title">输入您的信息</h2>
          <form onSubmit={handleSubmit}>
            <div className="form-grid">
              {/* 年龄 */}
              <div className="form-group">
                <label>年龄</label>
                <input
                  type="number"
                  name="age"
                  value={formData.age}
                  onChange={handleInputChange}
                  min="10"
                  max="100"
                  required
                />
              </div>

              {/* 性别 */}
              <div className="form-group">
                <label>性别</label>
                <select
                  name="gender"
                  value={formData.gender}
                  onChange={handleInputChange}
                >
                  <option value="Male">男性</option>
                  <option value="Female">女性</option>
                  <option value="Other">其他</option>
                </select>
              </div>

              {/* 每日总屏幕时间 */}
              <div className="form-group">
                <label>每日总屏幕时间 (小时)</label>
                <div className="slider-container">
                  <input
                    type="range"
                    name="daily_screen_time_hours"
                    value={formData.daily_screen_time_hours}
                    onChange={handleInputChange}
                    min="0"
                    max="16"
                    step="0.5"
                  />
                  <span className="slider-value">{formData.daily_screen_time_hours}h</span>
                </div>
              </div>

              {/* 工作相关时间 */}
              <div className="form-group">
                <label>工作相关时间 (小时)</label>
                <div className="slider-container">
                  <input
                    type="range"
                    name="work_related_hours"
                    value={formData.work_related_hours}
                    onChange={handleInputChange}
                    min="0"
                    max="12"
                    step="0.5"
                  />
                  <span className="slider-value">{formData.work_related_hours}h</span>
                </div>
              </div>

              {/* 娱乐时间 */}
              <div className="form-group">
                <label>娱乐时间 (小时)</label>
                <div className="slider-container">
                  <input
                    type="range"
                    name="entertainment_hours"
                    value={formData.entertainment_hours}
                    onChange={handleInputChange}
                    min="0"
                    max="10"
                    step="0.5"
                  />
                  <span className="slider-value">{formData.entertainment_hours}h</span>
                </div>
              </div>

              {/* 社交媒体时间 */}
              <div className="form-group">
                <label>社交媒体时间 (小时)</label>
                <div className="slider-container">
                  <input
                    type="range"
                    name="social_media_hours"
                    value={formData.social_media_hours}
                    onChange={handleInputChange}
                    min="0"
                    max="10"
                    step="0.5"
                  />
                  <span className="slider-value">{formData.social_media_hours}h</span>
                </div>
              </div>

              {/* 睡眠时长 */}
              <div className="form-group">
                <label>睡眠时长 (小时)</label>
                <div className="slider-container">
                  <input
                    type="range"
                    name="sleep_duration_hours"
                    value={formData.sleep_duration_hours}
                    onChange={handleInputChange}
                    min="3"
                    max="12"
                    step="0.5"
                  />
                  <span className="slider-value">{formData.sleep_duration_hours}h</span>
                </div>
              </div>

              {/* 睡眠质量 */}
              <div className="form-group">
                <label>睡眠质量评分 (1-10)</label>
                <div className="slider-container">
                  <input
                    type="range"
                    name="sleep_quality"
                    value={formData.sleep_quality}
                    onChange={handleInputChange}
                    min="1"
                    max="10"
                    step="1"
                  />
                  <span className="slider-value">{formData.sleep_quality}</span>
                </div>
              </div>
            </div>

            <button type="submit" className="submit-btn" disabled={loading}>
              {loading ? '分析中...' : '开始分析'}
            </button>
          </form>
        </section>

        {/* Loading */}
        {loading && (
          <div className="loading">
            <div className="loading-spinner"></div>
            <span>正在分析您的数据...</span>
          </div>
        )}

        {/* Error */}
        {error && (
          <div style={{ color: '#ff6b6b', padding: '16px', textAlign: 'center' }}>
            {error}
          </div>
        )}

        {/* Results */}
        {results && !loading && (
          <div className="results-section">
            {/* Score Cards */}
            <div className="score-cards">
              <div className="score-card anxiety">
                <div className="score-card-header">
                  <div className="score-icon">😰</div>
                  <span className="score-card-title">焦虑评分</span>
                </div>
                <div className="score-value">{results.predictions.anxiety_score.toFixed(1)}</div>
                <div className="score-bar">
                  <div 
                    className="score-bar-fill" 
                    style={{ width: `${(results.predictions.anxiety_score / 20) * 100}%` }}
                  />
                </div>
              </div>

              <div className="score-card depression">
                <div className="score-card-header">
                  <div className="score-icon">😔</div>
                  <span className="score-card-title">抑郁评分</span>
                </div>
                <div className="score-value">{results.predictions.depression_score.toFixed(1)}</div>
                <div className="score-bar">
                  <div 
                    className="score-bar-fill" 
                    style={{ width: `${(results.predictions.depression_score / 20) * 100}%` }}
                  />
                </div>
              </div>

              <div className="score-card sleep">
                <div className="score-card-header">
                  <div className="score-icon">😴</div>
                  <span className="score-card-title">预测睡眠质量</span>
                </div>
                <div className="score-value">{results.predictions.predicted_sleep_quality.toFixed(1)}</div>
                <div className="score-bar">
                  <div 
                    className="score-bar-fill" 
                    style={{ width: `${(results.predictions.predicted_sleep_quality / 10) * 100}%` }}
                  />
                </div>
              </div>
            </div>

            {/* Percentile Analysis */}
            <section className="percentile-section">
              <h2 className="section-title">您在人群中的位置</h2>
              <div className="percentile-grid">
                {results.percentile_analysis && Object.entries(results.percentile_analysis).map(([key, data]) => (
                  <div key={key} className="percentile-item">
                    <div className="percentile-header">
                      <span className="percentile-label">{FEATURE_NAMES[key] || key}</span>
                      <span className="percentile-value">{data.value}</span>
                    </div>
                    <div className="percentile-bar">
                      <div 
                        className="percentile-bar-fill" 
                        style={{ width: `${data.percentile}%` }}
                      />
                    </div>
                    <div className="percentile-desc">{data.description}</div>
                  </div>
                ))}
              </div>
            </section>

            {/* Charts */}
            <section className="charts-section">
              <h2 className="section-title">数据可视化</h2>
              
              <div className="chart-container">
                <h3 style={{ marginBottom: '16px', color: '#b3b3b3' }}>各指标百分位分布</h3>
                <ResponsiveContainer width="100%" height={400}>
                  <BarChart data={preparePercentileData()}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                    <XAxis dataKey="name" stroke="#b3b3b3" tick={{ fontSize: 12 }} />
                    <YAxis stroke="#b3b3b3" domain={[0, 100]} />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: '#282828', 
                        border: 'none', 
                        borderRadius: '8px',
                        color: '#fff'
                      }}
                      formatter={(value, name) => {
                        if (name === 'percentile') return [`${value}%`, '百分位'];
                        return [value, '数值'];
                      }}
                    />
                    <Legend />
                    <Bar dataKey="percentile" name="百分位排名" fill="#1db954" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div className="chart-container">
                <h3 style={{ marginBottom: '16px', color: '#b3b3b3' }}>综合指标雷达图</h3>
                <ResponsiveContainer width="100%" height={400}>
                  <RadarChart data={prepareRadarData()}>
                    <PolarGrid stroke="#333" />
                    <PolarAngleAxis dataKey="subject" stroke="#b3b3b3" tick={{ fontSize: 11 }} />
                    <PolarRadiusAxis angle={30} domain={[0, 100]} stroke="#b3b3b3" />
                    <Radar
                      name="您的数据"
                      dataKey="A"
                      stroke="#1db954"
                      fill="#1db954"
                      fillOpacity={0.3}
                    />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: '#282828', 
                        border: 'none', 
                        borderRadius: '8px',
                        color: '#fff'
                      }}
                    />
                    <Legend />
                  </RadarChart>
                </ResponsiveContainer>
              </div>

              <div className="chart-container">
                <h3 style={{ marginBottom: '16px', color: '#b3b3b3' }}>时间分配饼图</h3>
                <ResponsiveContainer width="100%" height={400}>
                  <PieChart>
                    <Pie
                      data={[
                        { name: '工作相关', value: formData.work_related_hours },
                        { name: '娱乐', value: formData.entertainment_hours },
                        { name: '社交媒体', value: formData.social_media_hours },
                        { name: '其他屏幕时间', value: Math.max(0, formData.daily_screen_time_hours - formData.work_related_hours - formData.entertainment_hours - formData.social_media_hours) }
                      ]}
                      cx="50%"
                      cy="50%"
                      outerRadius={150}
                      label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                      labelLine={{ stroke: '#b3b3b3' }}
                    >
                      {[0, 1, 2, 3].map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={CHART_COLORS[index % CHART_COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: '#282828', 
                        border: 'none', 
                        borderRadius: '8px',
                        color: '#fff'
                      }}
                      formatter={(value) => [`${value}小时`, '时长']}
                    />
                    <Legend />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </section>

            {/* Cause Analysis */}
            <section className="analysis-section">
              <h2 className="section-title">原因分析与建议</h2>
              <div className="analysis-grid">
                {/* 焦虑分析 */}
                <div className="analysis-card">
                  <div className="analysis-card-header">
                    <span className="analysis-card-icon">😰</span>
                    <span className="analysis-card-title">焦虑评分分析</span>
                  </div>
                  {results.cause_analysis?.anxiety?.has_issue ? (
                    <>
                      <p className="analysis-message">{results.cause_analysis.anxiety.message}</p>
                      {results.cause_analysis.anxiety.causes?.length > 0 && (
                        <div className="analysis-causes">
                          <h4>可能原因</h4>
                          <ul>
                            {results.cause_analysis.anxiety.causes.map((cause, idx) => (
                              <li key={idx}>{cause}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                      {results.cause_analysis.anxiety.suggestions?.length > 0 && (
                        <div className="analysis-suggestions">
                          <h4>改善建议</h4>
                          <ul>
                            {results.cause_analysis.anxiety.suggestions.map((suggestion, idx) => (
                              <li key={idx}>{suggestion}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </>
                  ) : (
                    <div className="no-issue">
                      <span>✓</span>
                      <span>{results.cause_analysis?.anxiety?.message || '您的焦虑评分处于正常范围'}</span>
                    </div>
                  )}
                </div>

                {/* 抑郁分析 */}
                <div className="analysis-card">
                  <div className="analysis-card-header">
                    <span className="analysis-card-icon">😔</span>
                    <span className="analysis-card-title">抑郁评分分析</span>
                  </div>
                  {results.cause_analysis?.depression?.has_issue ? (
                    <>
                      <p className="analysis-message">{results.cause_analysis.depression.message}</p>
                      {results.cause_analysis.depression.causes?.length > 0 && (
                        <div className="analysis-causes">
                          <h4>可能原因</h4>
                          <ul>
                            {results.cause_analysis.depression.causes.map((cause, idx) => (
                              <li key={idx}>{cause}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                      {results.cause_analysis.depression.suggestions?.length > 0 && (
                        <div className="analysis-suggestions">
                          <h4>改善建议</h4>
                          <ul>
                            {results.cause_analysis.depression.suggestions.map((suggestion, idx) => (
                              <li key={idx}>{suggestion}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </>
                  ) : (
                    <div className="no-issue">
                      <span>✓</span>
                      <span>{results.cause_analysis?.depression?.message || '您的抑郁评分处于正常范围'}</span>
                    </div>
                  )}
                </div>

                {/* 睡眠分析 */}
                <div className="analysis-card">
                  <div className="analysis-card-header">
                    <span className="analysis-card-icon">😴</span>
                    <span className="analysis-card-title">睡眠质量分析</span>
                  </div>
                  {results.cause_analysis?.sleep?.has_issue ? (
                    <>
                      <p className="analysis-message">{results.cause_analysis.sleep.message}</p>
                      {results.cause_analysis.sleep.causes?.length > 0 && (
                        <div className="analysis-causes">
                          <h4>可能原因</h4>
                          <ul>
                            {results.cause_analysis.sleep.causes.map((cause, idx) => (
                              <li key={idx}>{cause}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                      {results.cause_analysis.sleep.suggestions?.length > 0 && (
                        <div className="analysis-suggestions">
                          <h4>改善建议</h4>
                          <ul>
                            {results.cause_analysis.sleep.suggestions.map((suggestion, idx) => (
                              <li key={idx}>{suggestion}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </>
                  ) : (
                    <div className="no-issue">
                      <span>✓</span>
                      <span>{results.cause_analysis?.sleep?.message || '您的睡眠质量处于正常范围'}</span>
                    </div>
                  )}
                </div>
              </div>
            </section>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="footer">
        <p>MindScreen © 2024 - 基于机器学习的心理健康评估系统</p>
        <p style={{ marginTop: '8px', fontSize: '12px' }}>
          声明：本系统仅供参考，不能替代专业医疗诊断。如有需要，请咨询专业心理健康服务机构。
        </p>
      </footer>
    </div>
  );
}

export default App;
