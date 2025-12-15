/**
 * 问卷配置文件 - 集中管理所有问卷题目
 * 修改此文件即可调整问卷内容，无需修改组件代码
 */

// 基础信息题目
export const BASIC_QUESTIONS = [
  {
    id: 'age',
    field: 'age',
    label: '您的年龄',
    type: 'number',
    required: true,
    min: 10,
    max: 100,
    defaultValue: 25,
    category: 'basic',
    description: '请输入您的真实年龄'
  },
  {
    id: 'gender',
    field: 'gender',
    label: '您的性别',
    type: 'select',
    required: true,
    options: [
      { value: 'Male', label: '男性' },
      { value: 'Female', label: '女性' },
      { value: 'Other', label: '其他' }
    ],
    defaultValue: 'Male',
    category: 'basic'
  },
  {
    id: 'relationship',
    field: 'relationship',
    label: '您当前的感情状态',
    type: 'select',
    required: false,
    options: [
      { value: 'Single', label: '单身' },
      { value: 'In a relationship', label: '恋爱中' },
      { value: 'Married', label: '已婚' },
      { value: 'Divorced', label: '离异' }
    ],
    defaultValue: 'Single',
    category: 'basic'
  },
  {
    id: 'occupation',
    field: 'occupation',
    label: '您目前的职业/身份',
    type: 'select',
    required: false,
    options: [
      { value: 'University Student', label: '大学生' },
      { value: 'School Student', label: '中学生' },
      { value: 'Salaried Worker', label: '上班族' },
      { value: 'Retired', label: '退休/其他' }
    ],
    defaultValue: 'University Student',
    category: 'basic'
  }
];

// 社交媒体使用题目
export const SOCIAL_MEDIA_QUESTIONS = [
  {
    id: 'platforms',
    field: 'platforms',
    label: '您常用的社交/内容平台',
    type: 'multiselect',
    required: true,
    options: [
      { value: 'TikTok', label: '抖音/TikTok' },
      { value: 'YouTube', label: 'YouTube/B站' },
      { value: 'Instagram', label: 'Instagram/小红书' },
      { value: 'Facebook', label: '微信朋友圈' },
      { value: 'Twitter', label: '微博/X' },
      { value: 'Discord', label: 'Discord/QQ频道' },
      { value: 'Reddit', label: '论坛/贴吧' },
      { value: 'Pinterest', label: 'Pinterest/花瓣' },
      { value: 'Snapchat', label: 'Snapchat' }
    ],
    defaultValue: ['YouTube', 'Instagram'],
    category: 'social',
    description: '选择您日常使用的平台（可多选）'
  },
  {
    id: 'avg_time_per_day',
    field: 'avg_time_per_day',
    label: '您平均每天使用社交媒体的时长',
    type: 'select',
    required: true,
    options: [
      { value: 'Less than an Hour', label: '少于1小时', hours: 0.5 },
      { value: 'Between 1 and 2 hours', label: '1-2小时', hours: 1.5 },
      { value: 'Between 2 and 3 hours', label: '2-3小时', hours: 2.5 },
      { value: 'Between 3 and 4 hours', label: '3-4小时', hours: 3.5 },
      { value: 'Between 4 and 5 hours', label: '4-5小时', hours: 4.5 },
      { value: 'More than 5 hours', label: '5小时以上', hours: 6 }
    ],
    defaultValue: 'Between 2 and 3 hours',
    category: 'social'
  }
];

// Likert量表题目 (1-5分)
export const LIKERT_QUESTIONS = [
  // 数字成瘾相关 (Q9-Q12) - 用于模型特征
  {
    id: 'q9',
    field: 'q9',
    label: '无目的浏览社交媒体',
    fullLabel: '您多久会无目的地刷社交媒体？',
    type: 'likert',
    required: true,
    category: 'addiction',
    weight: 1,
    description: '1=从不，5=非常频繁',
    lowLabel: '从不',
    highLabel: '非常频繁'
  },
  {
    id: 'q10',
    field: 'q10',
    label: '工作/学习时被分心',
    fullLabel: '忙碌时被社交媒体分心的频率？',
    type: 'likert',
    required: true,
    category: 'addiction',
    weight: 1,
    lowLabel: '从不',
    highLabel: '非常频繁'
  },
  {
    id: 'q11',
    field: 'q11',
    label: '不用社媒时的焦躁感',
    fullLabel: '一段时间不用社交媒体会感到焦躁不安吗？',
    type: 'likert',
    required: true,
    category: 'addiction',
    weight: 1,
    lowLabel: '完全不会',
    highLabel: '非常焦躁'
  },
  {
    id: 'q12',
    field: 'q12',
    label: '容易分心程度',
    fullLabel: '您容易分心的程度？',
    type: 'likert',
    required: true,
    category: 'addiction',
    weight: 1,
    lowLabel: '很难分心',
    highLabel: '极易分心'
  },
  // 心理状态相关 (Q13-Q20) - 用于百分位分析
  {
    id: 'q13',
    field: 'q13',
    label: '被担忧困扰程度',
    fullLabel: '您被担忧困扰的程度？',
    type: 'likert',
    required: false,
    category: 'mental',
    weight: 1,
    lowLabel: '完全不会',
    highLabel: '严重困扰'
  },
  {
    id: 'q14',
    field: 'q14',
    label: '难以集中注意力',
    fullLabel: '您是否难以集中注意力做事？',
    type: 'likert',
    required: false,
    category: 'mental',
    weight: 1,
    lowLabel: '完全不会',
    highLabel: '非常困难'
  },
  {
    id: 'q15',
    field: 'q15',
    label: '与他人比较频率',
    fullLabel: '您通过社交媒体与他人比较的频率？',
    type: 'likert',
    required: false,
    category: 'comparison',
    weight: 1,
    lowLabel: '从不比较',
    highLabel: '经常比较'
  },
  {
    id: 'q16',
    field: 'q16',
    label: '比较后的感受',
    fullLabel: '这些比较给您带来的感受？',
    type: 'likert',
    required: false,
    category: 'comparison',
    weight: 1,
    lowLabel: '积极正面',
    highLabel: '消极负面'
  },
  {
    id: 'q17',
    field: 'q17',
    label: '寻求认可频率',
    fullLabel: '您寻求社交媒体点赞/评论认可的频率？',
    type: 'likert',
    required: false,
    category: 'validation',
    weight: 1,
    lowLabel: '从不在意',
    highLabel: '非常在意'
  },
  {
    id: 'q18',
    field: 'q18',
    label: '情绪低落频率',
    fullLabel: '您感到沮丧或情绪低落的频率？',
    type: 'likert',
    required: false,
    category: 'depression',
    weight: 1.5,
    lowLabel: '几乎没有',
    highLabel: '非常频繁',
    isKeyIndicator: true
  },
  {
    id: 'q19',
    field: 'q19',
    label: '日常兴趣波动',
    fullLabel: '您对日常活动兴趣波动的频率？',
    type: 'likert',
    required: false,
    category: 'depression',
    weight: 1,
    lowLabel: '很稳定',
    highLabel: '波动很大'
  },
  {
    id: 'q20',
    field: 'q20',
    label: '睡眠问题频率',
    fullLabel: '您遇到睡眠问题的频率？',
    type: 'likert',
    required: false,
    category: 'sleep',
    weight: 1.5,
    lowLabel: '几乎没有',
    highLabel: '非常频繁',
    isKeyIndicator: true
  }
];

// 题目分类配置
export const QUESTION_CATEGORIES = {
  basic: { name: '基本信息', color: '#6366f1' },
  social: { name: '社交媒体使用', color: '#8b5cf6' },
  addiction: { name: '数字使用习惯', color: '#ec4899' },
  mental: { name: '心理状态', color: '#14b8a6' },
  comparison: { name: '社交比较', color: '#f59e0b' },
  validation: { name: '社交认可', color: '#ef4444' },
  depression: { name: '情绪状态', color: '#8b5cf6' },
  sleep: { name: '睡眠质量', color: '#3b82f6' }
};

// 简版问卷配置 - 只包含核心题目
export const SIMPLE_MODE_QUESTIONS = [
  'age',
  'gender', 
  'platforms',
  'avg_time_per_day',
  'q9',
  'q10',
  'q11',
  'q12'
];

// 完整版问卷配置 - 包含所有题目
export const DETAILED_MODE_QUESTIONS = [
  ...BASIC_QUESTIONS.map(q => q.id),
  ...SOCIAL_MEDIA_QUESTIONS.map(q => q.id),
  ...LIKERT_QUESTIONS.map(q => q.id)
];

// 获取所有问题的映射
export const getAllQuestions = () => {
  return [...BASIC_QUESTIONS, ...SOCIAL_MEDIA_QUESTIONS, ...LIKERT_QUESTIONS];
};

// 根据ID获取问题配置
export const getQuestionById = (id) => {
  return getAllQuestions().find(q => q.id === id);
};

// 获取默认表单值
export const getDefaultFormValues = () => {
  const values = {};
  getAllQuestions().forEach(q => {
    values[q.field] = q.defaultValue ?? (q.type === 'likert' ? 3 : null);
  });
  return values;
};

// 问卷模式配置
export const QUESTIONNAIRE_MODES = {
  simple: {
    title: '快速评估',
    subtitle: '约30秒',
    description: '核心问题快速筛查，适合初步了解心理健康状态',
    questions: SIMPLE_MODE_QUESTIONS
  },
  detailed: {
    title: '完整评估', 
    subtitle: '约2分钟',
    description: '全面评估社交媒体使用与心理健康的关联',
    questions: DETAILED_MODE_QUESTIONS
  }
};

// 关键指标配置（用于结果展示）
export const KEY_INDICATORS = [
  { id: 'digital_addiction', label: '数字成瘾风险', questions: ['q9', 'q10', 'q11', 'q12'], color: '#ec4899' },
  { id: 'depression_risk', label: '情绪低落风险', questions: ['q18', 'q19'], color: '#8b5cf6' },
  { id: 'sleep_quality', label: '睡眠质量', questions: ['q20'], color: '#3b82f6' },
  { id: 'social_comparison', label: '社交比较压力', questions: ['q15', 'q16'], color: '#f59e0b' }
];

export default {
  BASIC_QUESTIONS,
  SOCIAL_MEDIA_QUESTIONS,
  LIKERT_QUESTIONS,
  QUESTION_CATEGORIES,
  QUESTIONNAIRE_MODES,
  KEY_INDICATORS,
  getAllQuestions,
  getQuestionById,
  getDefaultFormValues
};
