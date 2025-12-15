/**
 * 问卷组件 - 可复用的问卷UI组件
 */
import React from 'react';

// Likert量表组件
export function LikertScale({ question, value, onChange }) {
  const options = [1, 2, 3, 4, 5];
  
  return (
    <div className="likert-container">
      <div className="likert-labels">
        <span className="likert-label-low">{question.lowLabel || '从不'}</span>
        <span className="likert-label-high">{question.highLabel || '总是'}</span>
      </div>
      <div className="likert-scale">
        {options.map(opt => (
          <label 
            key={opt} 
            className={`likert-option ${Number(value) === opt ? 'selected' : ''}`}
          >
            <input
              type="radio"
              name={question.id}
              value={opt}
              checked={Number(value) === opt}
              onChange={() => onChange(question.field, opt)}
            />
            <span className="likert-circle">{opt}</span>
          </label>
        ))}
      </div>
    </div>
  );
}

// 多选平台组件
export function PlatformSelector({ question, value = [], onChange }) {
  const handleToggle = (platform) => {
    const newValue = value.includes(platform)
      ? value.filter(p => p !== platform)
      : [...value, platform];
    onChange(question.field, newValue);
  };
  
  return (
    <div className="platform-selector">
      <div className="platform-grid">
        {question.options.map(opt => (
          <label 
            key={opt.value}
            className={`platform-item ${value.includes(opt.value) ? 'selected' : ''}`}
          >
            <input
              type="checkbox"
              checked={value.includes(opt.value)}
              onChange={() => handleToggle(opt.value)}
            />
            <span className="platform-check">
              {value.includes(opt.value) && (
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                  <polyline points="20 6 9 17 4 12"/>
                </svg>
              )}
            </span>
            <span className="platform-label">{opt.label}</span>
          </label>
        ))}
      </div>
      <div className="platform-count">
        已选择 {value.length} 个平台
      </div>
    </div>
  );
}

// 下拉选择组件
export function SelectInput({ question, value, onChange }) {
  return (
    <select 
      value={value || ''} 
      onChange={(e) => onChange(question.field, e.target.value)}
      className="select-input"
    >
      {question.options.map(opt => (
        <option key={opt.value} value={opt.value}>
          {opt.label}
        </option>
      ))}
    </select>
  );
}

// 数字输入组件
export function NumberInput({ question, value, onChange }) {
  return (
    <input
      type="number"
      value={value || ''}
      onChange={(e) => onChange(question.field, Number(e.target.value))}
      min={question.min}
      max={question.max}
      className="number-input"
      placeholder={question.description}
    />
  );
}

// 问题卡片组件
export function QuestionCard({ question, value, onChange, index }) {
  const renderInput = () => {
    switch (question.type) {
      case 'likert':
        return <LikertScale question={question} value={value} onChange={onChange} />;
      case 'multiselect':
        return <PlatformSelector question={question} value={value} onChange={onChange} />;
      case 'select':
        return <SelectInput question={question} value={value} onChange={onChange} />;
      case 'number':
        return <NumberInput question={question} value={value} onChange={onChange} />;
      default:
        return null;
    }
  };
  
  return (
    <div className={`question-card ${question.isKeyIndicator ? 'key-indicator' : ''}`}>
      <div className="question-header">
        <span className="question-number">Q{index + 1}</span>
        <h3 className="question-label">
          {question.fullLabel || question.label}
        </h3>
        {question.required && <span className="required-badge">必填</span>}
      </div>
      {question.description && (
        <p className="question-description">{question.description}</p>
      )}
      <div className="question-input">
        {renderInput()}
      </div>
    </div>
  );
}

// 问卷进度条
export function QuestionnaireProgress({ current, total }) {
  const percentage = (current / total) * 100;
  
  return (
    <div className="progress-container">
      <div className="progress-bar">
        <div 
          className="progress-fill" 
          style={{ width: `${percentage}%` }}
        />
      </div>
      <span className="progress-text">{current} / {total}</span>
    </div>
  );
}

// 模式选择器
export function ModeSelector({ modes, currentMode, onModeChange }) {
  // SVG图标映射
  const modeIcons = {
    simple: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
        <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
      </svg>
    ),
    detailed: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
        <polyline points="14 2 14 8 20 8"/>
        <line x1="16" y1="13" x2="8" y2="13"/>
        <line x1="16" y1="17" x2="8" y2="17"/>
        <polyline points="10 9 9 9 8 9"/>
      </svg>
    )
  };

  return (
    <div className="mode-selector">
      {Object.entries(modes).map(([key, mode]) => (
        <button
          key={key}
          type="button"
          className={`mode-card glass-inner ${currentMode === key ? 'active' : ''}`}
          onClick={() => onModeChange(key)}
        >
          <div className="mode-icon-wrapper">
            {modeIcons[key]}
          </div>
          <div className="mode-info">
            <h4 className="mode-title">{mode.title}</h4>
            <span className="mode-subtitle">{mode.subtitle}</span>
          </div>
          <p className="mode-description">{mode.description}</p>
        </button>
      ))}
    </div>
  );
}

export default {
  LikertScale,
  PlatformSelector,
  SelectInput,
  NumberInput,
  QuestionCard,
  QuestionnaireProgress,
  ModeSelector
};
