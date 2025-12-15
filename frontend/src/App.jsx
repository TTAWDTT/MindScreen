import React, { useState } from 'react';
import axios from 'axios';
import {
  QUESTIONNAIRE_MODES,
  getDefaultFormValues,
  getAllQuestions,
  getQuestionById
} from './config/questions';
import { QuestionCard, ModeSelector } from './components/QuestionnaireComponents';
import { ResultsPage } from './pages/ResultsPage';
import { ModelInfoModal } from './components/ModelInfoModal';

const API_BASE = '/api';

function App() {
  // 表单状态
  const [formData, setFormData] = useState(getDefaultFormValues());
  const [mode, setMode] = useState('simple');
  
  // 结果状态
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // 弹窗状态
  const [showModelInfo, setShowModelInfo] = useState(false);
  
  // 获取当前模式的问题列表
  const currentQuestions = QUESTIONNAIRE_MODES[mode].questions;
  
  // 处理表单值变化
  const handleFieldChange = (field, value) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };
  
  // 提交表单
  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    
    try {
      // 构建API请求数据
      const payload = {
        age: formData.age,
        gender: formData.gender,
        relationship: formData.relationship || 'Single',
        occupation: formData.occupation || 'University Student',
        avg_time_per_day: formData.avg_time_per_day || 'Between 2 and 3 hours',
        platforms: formData.platforms || [],
        survey: {
          q9: formData.q9 || 3,
          q10: formData.q10 || 3,
          q11: formData.q11 || 3,
          q12: formData.q12 || 3,
          q13: formData.q13 || 3,
          q14: formData.q14 || 3,
          q15: formData.q15 || 3,
          q16: formData.q16 || 3,
          q17: formData.q17 || 3,
          q18: formData.q18 || 3,
          q19: formData.q19 || 3,
          q20: formData.q20 || 3
        }
      };
      
      const response = await axios.post(`${API_BASE}/predict`, payload);
      setResults(response.data);
    } catch (err) {
      setError(err.response?.data?.error || '预测失败，请稍后重试');
    } finally {
      setLoading(false);
    }
  };
  
  // 重置测试
  const handleRetry = () => {
    setResults(null);
    setFormData(getDefaultFormValues());
  };
  
  // 渲染问题
  const renderQuestions = () => {
    let questionIndex = 0;
    
    return currentQuestions.map(qId => {
      const question = getQuestionById(qId);
      if (!question) return null;
      questionIndex++;
      
      return (
        <QuestionCard
          key={question.id}
          question={question}
          value={formData[question.field]}
          onChange={handleFieldChange}
          index={questionIndex - 1}
        />
      );
    });
  };

  return (
    <div className="app">
      {/* 头部 */}
      <header className="header">
        <div className="header-content">
          <div className="logo">
            <div className="logo-icon">🧠</div>
            <div className="logo-text">
              <h1>MindScreen</h1>
              <span>社交媒体与心理健康评估</span>
            </div>
          </div>
          <button 
            className="info-btn"
            onClick={() => setShowModelInfo(true)}
          >
            <span>🔬</span>
            了解预测原理
          </button>
        </div>
      </header>

      <main className="main-content">
        {/* 结果页面 */}
        {results ? (
          <ResultsPage
            results={results}
            surveyAnswers={formData}
            onRetry={handleRetry}
          />
        ) : (
          <>
            {/* 欢迎区域 */}
            <section className="welcome-section">
              <h2>探索社交媒体对心理健康的影响</h2>
              <p>
                基于机器学习的智能评估系统，帮助您了解数字使用习惯与心理健康的关联，
                获取个性化的改善建议。
              </p>
              <div className="welcome-features">
                <div className="feature-item">
                  <span className="feature-icon">⚡</span>
                  <span>快速评估</span>
                </div>
                <div className="feature-item">
                  <span className="feature-icon">📊</span>
                  <span>数据可视化</span>
                </div>
                <div className="feature-item">
                  <span className="feature-icon">💡</span>
                  <span>个性化建议</span>
                </div>
              </div>
            </section>

            {/* 问卷表单 */}
            <section className="form-section">
              <h2 className="section-title">选择评估模式</h2>
              
              <ModeSelector
                modes={QUESTIONNAIRE_MODES}
                currentMode={mode}
                onModeChange={setMode}
              />
              
              <form onSubmit={handleSubmit}>
                <div className="questions-container">
                  {renderQuestions()}
                </div>
                
                {error && (
                  <div className="error-message">
                    <span>⚠️</span> {error}
                  </div>
                )}
                
                <button 
                  type="submit" 
                  className="submit-btn"
                  disabled={loading}
                >
                  {loading ? (
                    <>
                      <span className="loading-spinner"></span>
                      分析中...
                    </>
                  ) : (
                    <>开始评估</>
                  )}
                </button>
              </form>
            </section>
          </>
        )}
      </main>

      {/* 页脚 */}
      <footer className="footer">
        <p>MindScreen © 2024 - 基于机器学习的心理健康评估系统</p>
        <p className="disclaimer">
          声明：本系统仅供参考，不能替代专业医疗诊断。如有需要，请咨询专业心理健康服务机构。
        </p>
      </footer>

      {/* 模型说明弹窗 */}
      <ModelInfoModal
        isOpen={showModelInfo}
        onClose={() => setShowModelInfo(false)}
      />
    </div>
  );
}

export default App;
