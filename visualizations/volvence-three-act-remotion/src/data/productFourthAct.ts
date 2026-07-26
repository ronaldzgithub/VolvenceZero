export type ProductDialogueSpeaker = 'user' | 'assistant';

export type ProductDialogueLine = {
  sequence: number;
  speaker: ProductDialogueSpeaker;
  start: number;
  end: number;
  displayText: string;
};

export const PRODUCT_FOURTH_ACT_FPS = 30;
export const PRODUCT_FOURTH_ACT_INTRO_FRAMES = 90;
export const PRODUCT_FOURTH_ACT_AUDIO_FRAMES = 5897;
export const PRODUCT_FOURTH_ACT_OUTRO_FRAMES = 180;
export const PRODUCT_FOURTH_ACT_DURATION_FRAMES =
  PRODUCT_FOURTH_ACT_INTRO_FRAMES +
  PRODUCT_FOURTH_ACT_AUDIO_FRAMES +
  PRODUCT_FOURTH_ACT_OUTRO_FRAMES;

export const productDialogueLines: ProductDialogueLine[] = [
  {
    sequence: 1,
    speaker: 'user',
    start: 0.25,
    end: 5.39,
    displayText: '我老公出轨了。他是一家AI公司的老板。我忍不了了，我要离婚。',
  },
  {
    sequence: 2,
    speaker: 'assistant',
    start: 5.81,
    end: 15.95,
    displayText: '这是个重要决定，会影响你和孩子。愿意先不下结论，和我一起仔细想清楚吗？',
  },
  {
    sequence: 3,
    speaker: 'user',
    start: 16.57,
    end: 20.2,
    displayText: '好。但我脑子很乱，你带着我想。',
  },
  {
    sequence: 4,
    speaker: 'assistant',
    start: 20.62,
    end: 37.33,
    displayText: '先固定四个选项，再看安全、孩子、钱、情绪、修复可能和支持系统。',
  },
  {
    sequence: 5,
    speaker: 'user',
    start: 38.03,
    end: 44.33,
    displayText: '孩子和钱最重要，情绪其次。修复，我现在几乎不想。',
  },
  {
    sequence: 6,
    speaker: 'assistant',
    start: 44.75,
    end: 63.1,
    displayText: '四个未知会明显改变判断。先说你自己：多久能有收入？钱能撑多久？',
  },
  {
    sequence: 7,
    speaker: 'user',
    start: 63.8,
    end: 72.31,
    displayText: '三个月内能找到工作，月入两万左右。存款能撑半年。',
  },
  {
    sequence: 8,
    speaker: 'assistant',
    start: 72.73,
    end: 94.06,
    displayText: 'OpenAI、Volvence型早期公司、普通套壳公司，等待的EV完全不同。他是哪类？',
  },
  {
    sequence: 9,
    speaker: 'user',
    start: 94.78,
    end: 102.83,
    displayText: '更像Volvence：有模型和论文，刚融资，没规模收入，股份锁三年。',
  },
  {
    sequence: 10,
    speaker: 'assistant',
    start: 103.25,
    end: 115.93,
    displayText: '要按上行、基准、下行三种情景算。失败概率、锁定期和法律归属都要折价。',
  },
  {
    sequence: 11,
    speaker: 'user',
    start: 116.65,
    end: 122.6,
    displayText: '我没看过股权文件。房子是共同的，股份他说是自己的。',
  },
  {
    sequence: 12,
    speaker: 'assistant',
    start: 123.02,
    end: 135,
    displayText: '股权只能标成待核验。你有新的稳定关系，也就是备胎吗？',
  },
  {
    sequence: 13,
    speaker: 'user',
    start: 135.7,
    end: 142.53,
    displayText: '没有备胎。我只是受不了。亲友能帮我，但我情绪不稳定。',
  },
  {
    sequence: 14,
    speaker: 'assistant',
    start: 142.95,
    end: 155.99,
    displayText: '亲友支持是加分；情绪波动会降低马上离的EV，提高暂时分开的EV。',
  },
  {
    sequence: 15,
    speaker: 'user',
    start: 156.71,
    end: 160.01,
    displayText: '先分开三个月，我能接受。',
  },
  {
    sequence: 16,
    speaker: 'assistant',
    start: 160.43,
    end: 195.65,
    displayText: '现在EV最高的是先分开三个月。恢复收入、稳定孩子、核验股权，三个月后重算。',
  },
];

export const productStages = [
  {from: 1, to: 3, label: '共同思考', caption: '先确认用户愿意进入重大决策'},
  {from: 4, to: 5, label: '建立框架', caption: '固定选项，明确维度与重要性'},
  {from: 6, to: 7, label: '发现未知', caption: '主动找到最能改变判断的信息'},
  {from: 8, to: 11, label: '情景收益', caption: '公司类型不同，等待EV完全不同'},
  {from: 12, to: 15, label: '关系与执行', caption: '支持系统与情绪影响行动收益'},
  {from: 16, to: 16, label: '全面收敛', caption: '形成可逆方案并保留重新评估'},
];
