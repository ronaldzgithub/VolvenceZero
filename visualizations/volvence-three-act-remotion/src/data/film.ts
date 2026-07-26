export const FILM_FPS = 30;

export type ActKey = 'problem' | 'loop' | 'deployment' | 'product';

export type Act = {
  key: ActKey;
  start: number;
  end: number;
  index: string;
  eyebrow: string;
  title: string;
  subtitle: string;
};

export const acts: Act[] = [
  {
    key: 'problem',
    start: 0,
    end: 1080,
    index: '第一幕',
    eyebrow: 'THE MISSING LOOP',
    title: '会回答，不等于会从结果中学习',
    subtitle: '聊天与 Agent 的问题，来自同一个断点',
  },
  {
    key: 'loop',
    start: 1080,
    end: 3960,
    index: '第二幕',
    eyebrow: 'THE MODEL LEARNS TO DECIDE',
    title: '把状态、决策、结果和学习放回模型内部',
    subtitle: '自然交互进入，同一条闭环持续形成认知',
  },
  {
    key: 'deployment',
    start: 3960,
    end: 5850,
    index: '第三幕',
    eyebrow: 'ONE MODEL, MANY INDIVIDUALS',
    title: '共享能力，动态加载每个人',
    subtitle: '一个 AI 销售 Agent 的推理与规模化部署',
  },
  {
    key: 'product',
    start: 5850,
    end: 12836,
    index: '第四幕',
    eyebrow: '产品能力实演',
    title: '不替用户下结论，和用户一起把问题解决',
    subtitle: '同一个婚姻危机，两种模型的真实对话',
  },
];

export const FILM_DURATION_FRAMES = acts[acts.length - 1].end;
