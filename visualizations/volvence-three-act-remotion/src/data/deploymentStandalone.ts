export type DeploymentNarrationLine = {
  sequence: number;
  start: number;
  end: number;
  caption: string;
};

export const DEPLOYMENT_FPS = 30;
export const DEPLOYMENT_INTRO_FRAMES = 90;
export const DEPLOYMENT_AUDIO_FRAMES = 3691;
export const DEPLOYMENT_OUTRO_FRAMES = 180;
export const DEPLOYMENT_DURATION_FRAMES =
  DEPLOYMENT_INTRO_FRAMES +
  DEPLOYMENT_AUDIO_FRAMES +
  DEPLOYMENT_OUTRO_FRAMES;

export const deploymentNarration: DeploymentNarrationLine[] = [
  {
    sequence: 1,
    start: 0.25,
    end: 13.28,
    caption: 'Base + Adapter统一部署；每个用户只拥有独立档案',
  },
  {
    sequence: 2,
    start: 13.8,
    end: 25.17,
    caption: 'Base负责通用语言、知识、推理与工具能力',
  },
  {
    sequence: 3,
    start: 25.69,
    end: 48.88,
    caption: 'Adapter保存所有用户共享的Volvence决策先验',
  },
  {
    sequence: 4,
    start: 49.44,
    end: 76.37,
    caption: '用户档案不是模型权重，而是每个人的真实状态',
  },
  {
    sequence: 5,
    start: 76.97,
    end: 92.76,
    caption: '按身份读取获授权状态，再动态加载到共享服务',
  },
  {
    sequence: 6,
    start: 93.32,
    end: 110.32,
    caption: '共享模型无状态扩容，用户档案独立存储',
  },
  {
    sequence: 7,
    start: 110.88,
    end: 122.11,
    caption: '共享能力，隔离状态',
  },
];
