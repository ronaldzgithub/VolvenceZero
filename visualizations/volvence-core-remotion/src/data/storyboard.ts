export type SceneKey =
  | 'thesis'
  | 'turn'
  | 'predictionError'
  | 'timescales'
  | 'temporal'
  | 'memory'
  | 'gate'
  | 'loop';

export type StoryScene = {
  key: SceneKey;
  start: number;
  end: number;
  eyebrow: string;
  title: string;
  subtitle: string;
};

export const scenes: StoryScene[] = [
  {
    key: 'thesis',
    start: 0,
    end: 330,
    eyebrow: 'Core thesis',
    title: 'Volvence 是持续适应的关系智能体',
    subtitle: '不是静态模型加 prompt，而是冻结基底、控制器、记忆、信用与门控组成的有界学习系统。',
  },
  {
    key: 'turn',
    start: 330,
    end: 690,
    eyebrow: 'Snapshot-first runtime',
    title: '每一轮交互先变成可检查的快照',
    subtitle: 'owner 发布不可变 snapshot；consumer 只读公共契约，不重建上游内部状态。',
  },
  {
    key: 'predictionError',
    start: 690,
    end: 1050,
    eyebrow: 'Primitive learning signal',
    title: 'Prediction Error 是学习原始信号',
    subtitle: '系统先预测结果，再比较真实 outcome；credit、memory、evaluation 都只是 PE 的下游读数。',
  },
  {
    key: 'timescales',
    start: 1050,
    end: 1410,
    eyebrow: 'Nested learning',
    title: '学习被拆成四个时间尺度',
    subtitle: 'online-fast 适应当前轮次，session-medium 稳定场景，background-slow 异步沉淀，rare-heavy 离线刷新 artifact。',
  },
  {
    key: 'temporal',
    start: 1410,
    end: 1770,
    eyebrow: 'ETA control',
    title: '内部控制发生在 z_t / beta_t 空间',
    subtitle: 'metacontroller 选择低维 controller code，switch gate 决定延续或切换抽象动作，而不是做 token 空间 RL。',
  },
  {
    key: 'memory',
    start: 1770,
    end: 2130,
    eyebrow: 'Continuum memory',
    title: '记忆是连续谱，不是短期和长期二分',
    subtitle: 'CMS tower 把瞬态、会话、持久与派生索引连接起来；慢反思产出记忆沉淀和策略沉淀。',
  },
  {
    key: 'gate',
    start: 2130,
    end: 2490,
    eyebrow: 'Bounded self-modification',
    title: '高风险自修改必须经过 ModificationGate',
    subtitle: 'live runtime 默认冻结基底；adapter delta、LoRA、policy artifact 走 review、evidence、rollback 路径。',
  },
  {
    key: 'loop',
    start: 2490,
    end: 2880,
    eyebrow: 'Continuous adaptation',
    title: '每一轮都留下可审计的学习证据',
    subtitle: '关系连续性不是任务成功的副作用，而是 world / self 双轨预测、记忆和控制共同维护的结果。',
  },
];

export const modules = [
  {id: 'environment', label: 'Environment / User', owner: 'lifeform-* adapter', x: 130, y: 440},
  {id: 'substrate', label: 'Frozen substrate', owner: 'vz-substrate', x: 430, y: 250},
  {id: 'contracts', label: 'Snapshot bus', owner: 'vz-contracts', x: 760, y: 440},
  {id: 'temporal', label: 'ETA controller', owner: 'vz-temporal', x: 1070, y: 250},
  {id: 'cognition', label: 'PE / credit / regime', owner: 'vz-cognition', x: 1070, y: 635},
  {id: 'memory', label: 'CMS memory', owner: 'vz-memory', x: 1390, y: 440},
  {id: 'expression', label: 'Expression', owner: 'lifeform facade', x: 1660, y: 440},
];

export const edges = [
  ['environment', 'substrate'],
  ['substrate', 'contracts'],
  ['contracts', 'temporal'],
  ['contracts', 'cognition'],
  ['cognition', 'memory'],
  ['memory', 'temporal'],
  ['temporal', 'expression'],
  ['expression', 'environment'],
] as const;

export const peAxes = [
  {label: 'task', value: 0.44},
  {label: 'relationship', value: 0.68},
  {label: 'regime', value: 0.35},
  {label: 'action', value: 0.53},
];

export const timescaleLanes = [
  {label: 'online-fast', detail: 'per turn controller / memory update', cadence: 30},
  {label: 'session-medium', detail: 'scene-level pattern consolidation', cadence: 90},
  {label: 'background-slow', detail: 'post-session reflection queue', cadence: 150},
  {label: 'rare-heavy', detail: 'offline artifact refresh behind gates', cadence: 270},
];

export const gateStages = [
  'candidate artifact',
  'shadow evidence',
  'ModificationGate',
  'rollback point',
  'active import',
];
