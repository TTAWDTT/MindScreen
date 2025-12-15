/**
 * 建议生成工具
 * 根据评估结果生成个性化建议
 */

import { KEY_INDICATORS, LIKERT_QUESTIONS } from '../config/questions';

// 建议模板库
const ADVICE_TEMPLATES = {
  // 风险等级建议
  risk: {
    higher: [
      {
        title: '降低社交媒体使用时间',
        content: '建议将每日社交媒体使用控制在2小时以内，设置使用提醒和时间限制。',
        priority: 'high'
      },
      {
        title: '增加线下社交活动',
        content: '每周安排2-3次面对面社交，如与朋友聚餐、参加兴趣社团等。',
        priority: 'high'
      },
      {
        title: '建立健康的使用边界',
        content: '睡前1小时和起床后30分钟避免使用社交媒体，减少被动浏览。',
        priority: 'medium'
      }
    ],
    lower: [
      {
        title: '保持良好习惯',
        content: '您的社交媒体使用习惯较为健康，继续保持有意识的使用方式。',
        priority: 'low'
      },
      {
        title: '定期自我检视',
        content: '每月回顾一次自己的数字使用习惯，及时调整不良倾向。',
        priority: 'low'
      }
    ]
  },
  
  // 抑郁等级建议
  depression: {
    high: [ // 4-5级
      {
        title: '寻求专业帮助',
        content: '建议咨询心理健康专业人士，获取专业的评估和支持。',
        priority: 'critical'
      },
      {
        title: '建立支持网络',
        content: '向信任的家人或朋友倾诉，不要独自承受压力。',
        priority: 'high'
      },
      {
        title: '保持规律作息',
        content: '固定作息时间，保证7-8小时睡眠，有助于情绪稳定。',
        priority: 'high'
      }
    ],
    medium: [ // 3级
      {
        title: '关注情绪变化',
        content: '记录每日情绪状态，识别影响情绪的因素和触发点。',
        priority: 'medium'
      },
      {
        title: '增加身体活动',
        content: '每天30分钟中等强度运动，如散步、慢跑，可有效改善情绪。',
        priority: 'medium'
      }
    ],
    low: [ // 1-2级
      {
        title: '维持积极状态',
        content: '您的情绪状态较为稳定，继续保持健康的生活方式。',
        priority: 'low'
      }
    ]
  },
  
  // 具体指标建议
  indicators: {
    digital_addiction: {
      high: [
        {
          title: '数字排毒计划',
          content: '尝试每周一天"无社交媒体日"，体验脱离数字设备的生活。'
        },
        {
          title: '使用管理工具',
          content: '安装屏幕时间管理应用，设置每日使用上限和定时提醒。'
        }
      ],
      medium: [
        {
          title: '有意识使用',
          content: '每次打开社交媒体前问自己：我想要什么？避免无目的浏览。'
        }
      ]
    },
    sleep_quality: {
      high: [
        {
          title: '改善睡眠环境',
          content: '保持卧室安静、黑暗、凉爽，睡前避免使用电子设备。'
        },
        {
          title: '建立睡眠仪式',
          content: '睡前进行放松活动如阅读、冥想，帮助身体进入睡眠状态。'
        }
      ]
    },
    social_comparison: {
      high: [
        {
          title: '调整关注内容',
          content: '取消关注让您产生负面情绪的账号，关注更多正向、真实的内容。'
        },
        {
          title: '培养自我认同',
          content: '记录自己的成就和进步，专注于个人成长而非与他人比较。'
        }
      ]
    }
  }
};

// 热线资源
export const HELP_RESOURCES = [
  {
    name: '全国心理援助热线',
    number: '400-161-9995',
    available: '24小时'
  },
  {
    name: '北京心理危机研究与干预中心',
    number: '010-82951332',
    available: '24小时'
  },
  {
    name: '生命热线',
    number: '400-821-1215',
    available: '24小时'
  }
];

/**
 * 根据评估结果生成建议
 */
export function generateAdvice(results, surveyAnswers) {
  const advice = [];
  const predictions = results?.predictions || {};
  
  // 1. 根据风险等级添加建议
  const riskLevel = predictions.risk;
  if (riskLevel === 'higher' || riskLevel === 1 || riskLevel === '1') {
    advice.push(...ADVICE_TEMPLATES.risk.higher);
  } else {
    advice.push(...ADVICE_TEMPLATES.risk.lower);
  }
  
  // 2. 根据抑郁等级添加建议
  const depLevel = Number(predictions.depressed) || 3;
  if (depLevel >= 4) {
    advice.push(...ADVICE_TEMPLATES.depression.high);
  } else if (depLevel === 3) {
    advice.push(...ADVICE_TEMPLATES.depression.medium);
  } else {
    advice.push(...ADVICE_TEMPLATES.depression.low);
  }
  
  // 3. 根据具体指标添加建议
  if (surveyAnswers) {
    // 检查数字成瘾分数
    const addictionScore = ['q9', 'q10', 'q11', 'q12']
      .reduce((sum, q) => sum + (Number(surveyAnswers[q]) || 3), 0);
    if (addictionScore >= 16) {
      advice.push(...(ADVICE_TEMPLATES.indicators.digital_addiction.high || []));
    } else if (addictionScore >= 12) {
      advice.push(...(ADVICE_TEMPLATES.indicators.digital_addiction.medium || []));
    }
    
    // 检查睡眠问题
    const sleepScore = Number(surveyAnswers.q20) || 3;
    if (sleepScore >= 4) {
      advice.push(...(ADVICE_TEMPLATES.indicators.sleep_quality.high || []));
    }
    
    // 检查社交比较
    const comparisonScore = (Number(surveyAnswers.q15) || 3) + (Number(surveyAnswers.q16) || 3);
    if (comparisonScore >= 8) {
      advice.push(...(ADVICE_TEMPLATES.indicators.social_comparison.high || []));
    }
  }
  
  // 去重并按优先级排序
  const priorityOrder = { critical: 0, high: 1, medium: 2, low: 3 };
  const uniqueAdvice = advice.reduce((acc, item) => {
    if (!acc.find(a => a.title === item.title)) {
      acc.push(item);
    }
    return acc;
  }, []);
  
  return uniqueAdvice.sort((a, b) => 
    (priorityOrder[a.priority] || 3) - (priorityOrder[b.priority] || 3)
  );
}

/**
 * 计算关键指标的状态
 */
export function calculateKeyIndicators(surveyAnswers, percentiles) {
  return KEY_INDICATORS.map(indicator => {
    const scores = indicator.questions.map(q => Number(surveyAnswers?.[q]) || 3);
    const avgScore = scores.reduce((a, b) => a + b, 0) / scores.length;
    
    // 从percentiles中获取对应的百分位数据
    const percentileData = percentiles?.filter(p => 
      indicator.questions.includes(p.id)
    ) || [];
    const avgPercentile = percentileData.length > 0
      ? percentileData.reduce((sum, p) => sum + (p.percentile || 50), 0) / percentileData.length
      : 50;
    
    let status = 'normal';
    let statusLabel = '正常';
    if (avgPercentile >= 75) {
      status = 'high';
      statusLabel = '偏高';
    } else if (avgPercentile >= 50) {
      status = 'medium';
      statusLabel = '中等';
    } else {
      status = 'low';
      statusLabel = '良好';
    }
    
    return {
      ...indicator,
      score: avgScore,
      percentile: avgPercentile,
      status,
      statusLabel
    };
  });
}

/**
 * 生成综合评价文案
 */
export function generateSummary(results) {
  const composite = results?.composite_score;
  if (!composite) return null;
  
  const percentile = composite.percentile || 50;
  
  if (percentile >= 80) {
    return {
      level: 'critical',
      title: '需要重点关注',
      message: '您的评估结果显示心理健康风险较高，建议尽快寻求专业帮助。',
      color: '#ef4444'
    };
  } else if (percentile >= 60) {
    return {
      level: 'warning',
      title: '存在一定风险',
      message: '您的部分指标高于平均水平，建议关注并适当调整生活方式。',
      color: '#f59e0b'
    };
  } else if (percentile >= 40) {
    return {
      level: 'normal',
      title: '状态中等',
      message: '您的整体状态处于人群中等水平，保持健康习惯即可。',
      color: '#6366f1'
    };
  } else {
    return {
      level: 'good',
      title: '状态良好',
      message: '您的心理健康状态优于大多数人，继续保持！',
      color: '#10b981'
    };
  }
}

export default {
  generateAdvice,
  calculateKeyIndicators,
  generateSummary,
  HELP_RESOURCES
};
