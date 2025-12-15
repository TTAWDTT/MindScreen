import React, { useState } from 'react';
import axios from 'axios';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell
} from 'recharts';

const API_BASE = '/api';
const PLATFORM_OPTIONS = [
  { value: 'TikTok', label: '抖音/TikTok' },
  { value: 'YouTube', label: '油管/YouTube（含 B站 类似用途）' },
  { value: 'Instagram', label: '照片社交/Instagram' },
  { value: 'Facebook', label: '熟人社交/Facebook（类朋友圈）' },
  { value: 'Twitter', label: '微博/X（Twitter）' },
  { value: 'Discord', label: '社区语音/Discord' },
  { value: 'Reddit', label: '论坛/Reddit' },
  { value: 'Pinterest', label: '图片灵感/Pinterest' },
  { value: 'Snapchat', label: '阅后即焚/Snapchat' }
];

const TIME_OPTIONS = [
  { value: 'Less than an Hour', label: '少于 1 小时/天' },
  { value: 'Between 1 and 2 hours', label: '1-2 小时/天' },
  { value: 'Between 2 and 3 hours', label: '2-3 小时/天' },
  { value: 'Between 3 and 4 hours', label: '3-4 小时/天' },
  { value: 'Between 4 and 5 hours', label: '4-5 小时/天' },
  { value: 'More than 5 hours', label: '5 小时以上/天' }
];

const RELATIONSHIP = [
  { value: 'Single', label: '单身' },
  { value: 'In a relationship', label: '恋爱中' },
  { value: 'Married', label: '已婚' },
  { value: 'Divorced', label: '离异' }
];

const STATUS_COLORS = {
  '较低风险': '#2ecc71',
  '较高风险': '#e67e22',
  higher: '#e67e22',
  lower: '#2ecc71'
};

const CHART_COLORS = ['#1db954', '#ff9f43', '#845ef7', '#4ecdc4', '#f5576c'];

const SIMPLE_TITLE = '简版问卷（约 30 秒）';
const DETAILED_TITLE = '完整版问卷（约 1 分钟）';
const LIKERT_OPTIONS = [1, 2, 3, 4, 5];

const DETAILED_EXTRA_QUESTIONS = [
  { id: 'occupation', label: 'Q4. 您目前的职业/身份？', type: 'select', options: ['学生', '上班族', '自由职业', '其他'] },
  { id: 'affiliate', label: 'Q5. 您所在的机构类型？', type: 'select', options: ['学校/大学', '公司/机构', '其他'] },
  { id: 'use_social', label: 'Q6. 您是否使用社交媒体？', type: 'select', options: ['是', '否'] },
  { id: 'q9', label: 'Q9. 您会无目的地刷社交媒体的频率？', type: 'likert' },
  { id: 'q10', label: 'Q10. 当忙碌时被社交媒体分心的频率？', type: 'likert' },
  { id: 'q11', label: 'Q11. 如果一段时间不用社交媒体会感到不安吗？', type: 'likert' },
  { id: 'q12', label: 'Q12. 您容易分心的程度（1-5）', type: 'likert' },
  { id: 'q13', label: 'Q13. 您被担忧困扰的程度（1-5）', type: 'likert' },
  { id: 'q14', label: 'Q14. 您是否难以集中注意力？', type: 'likert' },
  { id: 'q15', label: 'Q15. 您因社交媒体与他人比较的频率（1-5）', type: 'likert' },
  { id: 'q16', label: 'Q16. 对上述比较的感受（1-5）', type: 'likert' },
  { id: 'q17', label: 'Q17. 您寻求社交媒体点赞/评论认可的频率？', type: 'likert' },
  { id: 'q18', label: 'Q18. 您感到沮丧或情绪低落的频率？', type: 'likert' },
  { id: 'q19', label: 'Q19. 您对日常活动兴趣波动的频率？', type: 'likert' },
  { id: 'q20', label: 'Q20. 您遇到睡眠问题的频率？', type: 'likert' }
];

function App() {
  const [formData, setFormData] = useState({
    age: 25,
    gender: 'Male',
    relationship: 'Single',
    avg_time_per_day: 'Between 2 and 3 hours',
    platforms: ['YouTube', 'Instagram']
  });

  const [mode, setMode] = useState('simple'); // simple | detailed

  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const [extraAnswers, setExtraAnswers] = useState({
    occupation: '学生',
    affiliate: '学校',
    use_social: '是',
    q9: 3,
    q10: 3,
    q11: 3,
    q12: 3,
    q13: 3,
    q14: 3,
    q15: 3,
    q16: 3,
    q17: 3,
    q18: 3,
    q19: 3,
    q20: 3
  });

  const handleInputChange = (e) => {
    const { name, value, type } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'number' || type === 'range' ? parseFloat(value) : value
    }));
  };

  const handlePlatformToggle = (platform) => {
    setFormData(prev => {
      const exists = prev.platforms.includes(platform);
      const nextPlatforms = exists
        ? prev.platforms.filter(p => p !== platform)
        : [...prev.platforms, platform];
      return { ...prev, platforms: nextPlatforms };
    });
  };

  const handleExtraChange = (id, value) => {
    setExtraAnswers(prev => ({ ...prev, [id]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const payload = { ...formData };
      if (!payload.avg_time_per_day) {
        payload.avg_time_per_day = 'Between 2 and 3 hours';
      }
      payload.survey = extraAnswers;

      const response = await axios.post(`${API_BASE}/predict`, payload);
      console.log('[DEBUG] Response data:', response.data);
      console.log('[DEBUG] Composite score:', response.data.composite_score);
      setResults(response.data);
    } catch (err) {
      setError(err.response?.data?.error || '预测失败，请稍后重试');
    } finally {
      setLoading(false);
    }
  };

  const renderProbBars = (probs = []) => (
    <div className="prob-list">
      {probs.map((item, idx) => (
        <div key={idx} className="prob-item">
          <div className="prob-label">{translateLabel(item.label)}</div>
          <div className="prob-bar">
            <div
              className="prob-bar-fill"
              style={{ width: `${(item.probability * 100).toFixed(1)}%` }}
            >
              {(item.probability * 100).toFixed(1)}%
            </div>
          </div>
        </div>
      ))}
    </div>
  );

  function translateLabel(lbl) {
    if (lbl == null) return '';
    const map = {
      higher: '较高风险',
      lower: '较低风险'
    };
    return map[String(lbl)] || String(lbl);
  }

  const percentileData = (results?.percentiles || []).map(p => ({
    name: p.label || p.id,
    value: p.percentile ?? 0,
    raw: p.value ?? null,
    baseline: p.baseline_percentile ?? 50,
    baselineRaw: p.baseline_value ?? null,
  }));

  const depChartData = (results?.predictions?.depressed_probs || []).map(p => ({
    name: translateLabel(p.label),
    value: Math.round(p.probability * 1000) / 10
  }));

  const buildAdvice = () => {
    if (!results?.predictions) return [];
    const adv = [];
    if (results.predictions.risk === 'higher') {
      adv.push('心理风险偏高：减少高刺激社交媒体使用时长，设置每日上限，增加线下社交与运动。');
    } else {
      adv.push('心理风险较低：保持良好习惯，定期自我觉察，避免长时间刷短视频。');
    }
    const dep = results.predictions.depressed;
    if (dep >= 4) {
      adv.push('抑郁等级偏高：建议及时与专业人士沟通，确保规律作息与社交支持。');
    } else if (dep === '3' || dep === 3) {
      adv.push('情绪波动中等：保持运动与睡眠，安排可控的小目标，减少信息过载。');
    } else {
      adv.push('情绪状态较稳：继续维持稳定作息与正向社交，关注情绪微调。');
    }
    return adv;
  };

  return (
    <div className="app">
      <header className="header">
        <div className="header-content">
          <div className="logo">
            <div className="logo-icon">🧠</div>
            <div>
              <h1>MindScreen</h1>
              <span>数字使用与心理健康评估</span>
            </div>
          </div>
        </div>
      </header>

      <main className="main-content">
        {!results && (
          <section className="welcome-section">
            <h2>用 smmh 新模型评估心理健康</h2>
            <p>选择「简单测试」或「详细测试」。简单测试仅需基础信息，详细测试可补充更多社交使用情况，输出心理风险（高/低）与抑郁等级（1-5）。</p>
          </section>
        )}

        <section className="form-section">
          <h2 className="section-title">填写您的数据（问卷形式）</h2>
          <div className="mode-switch">
            <button
              type="button"
              className={mode === 'simple' ? 'mode-btn active' : 'mode-btn'}
              onClick={() => setMode('simple')}
            >
              {SIMPLE_TITLE}
            </button>
            <button
              type="button"
              className={mode === 'detailed' ? 'mode-btn active' : 'mode-btn'}
              onClick={() => setMode('detailed')}
            >
              {DETAILED_TITLE}
            </button>
          </div>
          <p className="mode-hint">
            {mode === 'simple'
              ? '简版问卷：将 20 题浓缩为 3 题（年龄、性别、常用平台），其余采用默认值。'
              : '完整版问卷：展示与数据集一致的 20 题，可逐题作答，输入最贴近原始问卷。'}
          </p>
          <form onSubmit={handleSubmit}>
            <div className="form-grid">
              <div className="form-group">
                <label>Q1. 您的年龄？</label>
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

              <div className="form-group">
                <label>Q2. 您的性别？</label>
                <select name="gender" value={formData.gender} onChange={handleInputChange}>
                  <option value="Male">男性</option>
                  <option value="Female">女性</option>
                  <option value="Other">其他</option>
                </select>
              </div>

              <div className="form-group">
                <label>Q7. 您常用的社交/内容平台有哪些？（可多选）</label>
                <div className="platform-grid">
                  {PLATFORM_OPTIONS.map(p => (
                    <label key={p.value} className="platform-item">
                      <input
                        type="checkbox"
                        checked={formData.platforms.includes(p.value)}
                        onChange={() => handlePlatformToggle(p.value)}
                      />
                      <span>{p.label}</span>
                    </label>
                  ))}
                </div>
                <div className="platform-tip">已选 {formData.platforms.length} 个平台（越符合日常越好，至少选择 1 项）</div>
              </div>

              {mode === 'detailed' && (
                <>
                  <div className="form-group">
                    <label>Q3. 您当前的感情状态？</label>
                    <select name="relationship" value={formData.relationship} onChange={handleInputChange}>
                      {RELATIONSHIP.map(r => (
                        <option key={r.value} value={r.value}>{r.label}</option>
                      ))}
                    </select>
                  </div>

                  <div className="form-group">
                    <label>Q8. 您平均每天使用社交/内容平台的时长？</label>
                    <select name="avg_time_per_day" value={formData.avg_time_per_day} onChange={handleInputChange}>
                      {TIME_OPTIONS.map(t => (
                        <option key={t.value} value={t.value}>{t.label}</option>
                      ))}
                    </select>
                    <div className="platform-tip">此项用于更贴近数据集中“每日时长”问卷题</div>
                  </div>

                  {DETAILED_EXTRA_QUESTIONS.map(q => (
                    <div className="form-group" key={q.id}>
                      <label>{q.label}</label>
                      {q.type === 'select' && (
                        <select value={extraAnswers[q.id]} onChange={(e) => handleExtraChange(q.id, e.target.value)}>
                          {q.options.map(opt => (
                            <option key={opt} value={opt}>{opt}</option>
                          ))}
                        </select>
                      )}
                      {q.type === 'likert' && (
                        <div className="likert-row">
                          {LIKERT_OPTIONS.map(val => (
                            <label key={val} className="likert-option">
                              <input
                                type="radio"
                                name={q.id}
                                value={val}
                                checked={Number(extraAnswers[q.id]) === val}
                                onChange={() => handleExtraChange(q.id, val)}
                              />
                              <span>{val}</span>
                            </label>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </>
              )}
            </div>

            <button type="submit" className="submit-btn" disabled={loading}>
              {loading ? '分析中...' : '开始分析'}
            </button>
          </form>
        </section>

        {loading && (
          <div className="loading">
            <div className="loading-spinner"></div>
            <span>正在分析您的数据...</span>
          </div>
        )}

        {error && (
          <div style={{ color: '#ff6b6b', padding: '16px', textAlign: 'center' }}>
            {error}
          </div>
        )}

        {results && !loading && (
          <div className="results-section">
            {/* 综合评分卡片 - 如果有数据则显示 */}
            {results.composite_score && (
              <div className="composite-score-banner">
                <div className="composite-score-content">
                  <div className="composite-score-icon">🎯</div>
                  <div className="composite-score-info">
                    <h3>综合心理健康评分</h3>
                    <div className="composite-score-value">
                      {(results.composite_score.score * 100).toFixed(1)}
                      <span className="composite-score-max">/100</span>
                    </div>
                    <div className="composite-percentile">
                      {results.composite_score.percentile >= 50 
                        ? `心理风险超过 ${results.composite_score.percentile.toFixed(1)}% 用户`
                        : `心理状态优于 ${(100 - results.composite_score.percentile).toFixed(1)}% 用户`
                      }
                    </div>
                    <div className="composite-rank-text">{results.composite_score.rank_description}</div>
                  </div>
                </div>
              </div>
            )}

            <div className="score-cards">
              <div className="score-card">
                <div className="score-card-header">
                  <div className="score-icon" style={{ color: STATUS_COLORS[translateLabel(results.predictions.risk)] || '#1db954' }}>🧩</div>
                  <span className="score-card-title">心理风险 (高/低)</span>
                </div>
                <div className="score-value">{translateLabel(results.predictions.risk)}</div>
                {renderProbBars(results.predictions.risk_probs)}
              </div>

              <div className="score-card">
                <div className="score-card-header">
                  <div className="score-icon" style={{ color: '#ff9f43' }}>⚡</div>
                  <span className="score-card-title">抑郁等级 (1-5)</span>
                </div>
                <div className="score-value">{translateLabel(results.predictions.depressed)}</div>
                {renderProbBars(results.predictions.depressed_probs)}
              </div>
            </div>

            {/* 仅在完整版问卷或有百分位数据时才显示概率分布可视化 */}
            {percentileData.length > 0 && (
              <section className="charts-section">
                <h2 className="section-title">概率分布可视化</h2>
              <div className="chart-grid">
                <div className="chart-card">
                  <h3>各题得分分布（百分位）</h3>
                  <ResponsiveContainer width="100%" height={320}>
                    <BarChart data={percentileData} layout="vertical" margin={{ left: 20 }}>
                      <XAxis type="number" domain={[0, 100]} tick={{ fill: '#b3b3b3' }} />
                      <YAxis type="category" dataKey="name" width={220} tick={{ fill: '#b3b3b3', fontSize: 12 }} />
                      <Tooltip
                        formatter={(v, _, item) => {
                          const raw = item?.payload?.raw;
                          const baseline = item?.payload?.baseline;
                          const pct = Number(v).toFixed(1) + '%';
                          if (item?.dataKey === 'baseline') {
                            const bRaw = item?.payload?.baselineRaw;
                            return [pct, bRaw != null ? `训练集均值 ${bRaw.toFixed(1)}` : '训练集均值'];
                          }
                          return raw != null ? [pct, `得分 ${raw}`] : [pct, '百分位'];
                        }}
                        cursor={{ fill: 'rgba(255,255,255,0.04)' }}
                        contentStyle={{ background: '#181818', border: '1px solid #2a2a2a', borderRadius: 8, color: '#fff' }}
                      />
                      {/* 背景虚化“正常分数”对比：用训练集均值的百分位作为淡色底条 */}
                      <Bar dataKey="baseline" radius={[6, 6, 6, 6]} fill="rgba(255,255,255,0.06)" />
                      <Bar dataKey="value" radius={[6, 6, 6, 6]} fill={CHART_COLORS[0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                <div className="chart-card">
                  <h3>抑郁等级概率分布</h3>
                  <ResponsiveContainer width="100%" height={320}>
                    <PieChart>
                      <Pie
                        data={depChartData}
                        dataKey="value"
                        nameKey="name"
                        cx="50%"
                        cy="50%"
                        innerRadius={50}
                        outerRadius={90}
                        paddingAngle={2}
                        label={({ name, value }) => `${name} ${value}%`}
                      >
                        {depChartData.map((entry, idx) => (
                          <Cell key={`cell-${idx}`} fill={CHART_COLORS[idx % CHART_COLORS.length]} />
                        ))}
                      </Pie>
                      <Tooltip formatter={(v, n) => [`${v}%`, n]} />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              </div>
              </section>
            )}

            <section className="analysis-section">
              <h2 className="section-title">建议</h2>
              <div className="analysis-card">
                {buildAdvice().map((t, i) => (
                  <p key={i} className="analysis-message">• {t}</p>
                ))}
              </div>
            </section>
          </div>
        )}
      </main>

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
