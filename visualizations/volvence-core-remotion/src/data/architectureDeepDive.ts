export const ARCHITECTURE_DEEP_DIVE_FPS = 30;

export type ArchitectureSceneKey =
  | 'opening'
  | 'gap'
  | 'stack'
  | 'hydration'
  | 'state-kv'
  | 'eta'
  | 'nl'
  | 'active-learning'
  | 'prediction-error'
  | 'rare-heavy'
  | 'chat'
  | 'agent'
  | 'evidence'
  | 'outro';

export type ArchitectureScene = {
  key: ArchitectureSceneKey;
  start: number;
  end: number;
  index: string;
  chapter: string;
  title: string;
  subtitle: string;
  status?: 'CURRENT' | 'TARGET' | 'HYBRID';
};

export const architectureScenes: ArchitectureScene[] = [
  {
    key: 'opening',
    start: 0,
    end: 360,
    index: '00',
    chapter: 'Volvence model system',
    title: '语言模型，开始拥有“持续形成判断”的能力',
    subtitle:
      '不是更长的提示词，也不是每个用户一套模型，而是一套能感知状态、形成决策、观察结果并持续修正的模型系统。',
    status: 'HYBRID',
  },
  {
    key: 'gap',
    start: 360,
    end: 900,
    index: '01',
    chapter: 'Why current models break',
    title: '今天的大模型，每一轮都在重新认识世界',
    subtitle:
      '聊天里容易随用户最后一句话摇摆；任务型 Agent 能调用工具，却很难知道何时继续、何时求证、何时切换策略。',
    status: 'CURRENT',
  },
  {
    key: 'stack',
    start: 900,
    end: 1500,
    index: '02',
    chapter: 'Three different kinds of state',
    title: '先把“模型能力、用户状态、当轮决策”彻底分开',
    subtitle:
      '基底和共享适配器面向所有用户；个人状态按请求动态加载；ETA 控制代码在每个决策步快速变化。',
    status: 'TARGET',
  },
  {
    key: 'hydration',
    start: 1500,
    end: 2130,
    index: '03',
    chapter: 'Pre-inference hydration',
    title: '用户不做 context engineering，模型先自行装载状态',
    subtitle:
      '每个 owner 发布不可变快照；运行时只传播版本、权限和血缘，不重新解释用户，也不把画像拼成隐藏 prompt。',
    status: 'TARGET',
  },
  {
    key: 'state-kv',
    start: 2130,
    end: 2820,
    index: '04',
    chapter: 'Volvence State KV',
    title: '状态在第一个 token 之前进入注意力空间',
    subtitle:
      'Key 决定何时应该注意这类状态，Value 决定注意后把什么带入计算；精确事实仍留在可引用、可审计的事实通道。',
    status: 'TARGET',
  },
  {
    key: 'eta',
    start: 2820,
    end: 3600,
    index: '05',
    chapter: 'ETA · temporal abstraction',
    title: '不在 token 海洋里做强化学习，而在低维决策代码中行动',
    subtitle:
      '编码器从残差序列形成候选控制代码，βₜ 决定延续还是切换抽象动作，解码器把 zₜ 变成有界残差控制。',
    status: 'HYBRID',
  },
  {
    key: 'nl',
    start: 3600,
    end: 4380,
    index: '06',
    chapter: 'NL · nested learning',
    title: '不同知识，以不同频率学习',
    subtitle:
      '即时状态、会话经验、跨会话认知和共享模型能力不写进同一参数块，也不以同一个节奏更新。',
    status: 'HYBRID',
  },
  {
    key: 'active-learning',
    start: 4380,
    end: 4920,
    index: '07',
    chapter: 'Active learning',
    title: '不知道时不猜：只问最值得问的问题',
    subtitle:
      '不确定性、信息增益、决策影响和不可逆风险共同决定是否求证；主动学习把稀疏人工反馈集中在关键样本上。',
    status: 'CURRENT',
  },
  {
    key: 'prediction-error',
    start: 4920,
    end: 5580,
    index: '08',
    chapter: 'Prediction-error closure',
    title: '结果不是评分表，而是下一轮学习的起点',
    subtitle:
      '模型先留下预测，外部结果回来后形成 Prediction Error；信用分配判断该修正状态、记忆、控制策略还是提交慢速训练。',
    status: 'HYBRID',
  },
  {
    key: 'rare-heavy',
    start: 5580,
    end: 6240,
    index: '09',
    chapter: 'Rare-heavy adaptation',
    title: '稳定技能进入共享 LoRA，不能在用户对话中偷偷改基底',
    subtitle:
      '高价值经历经过匿名化、反事实训练、消融、安全评估和 ModificationGate，才成为新的共享适配器版本。',
    status: 'CURRENT',
  },
  {
    key: 'chat',
    start: 6240,
    end: 6900,
    index: '10',
    chapter: 'Chat intelligence',
    title: '聊天模型从“顺着说”，变成共同建模、共同收敛',
    subtitle:
      '它保持关系连续性，主动补齐关键变量，区分事实、信念和预测，并根据用户真实结果修正后续判断。',
    status: 'TARGET',
  },
  {
    key: 'agent',
    start: 6900,
    end: 7560,
    index: '11',
    chapter: 'Agent intelligence',
    title: '任务型 Agent 从“会调工具”，变成“知道为什么做、何时换”',
    subtitle:
      'ETA 管子目标边界和策略切换，NL 保存跨任务经验，主动学习在高风险步骤请求确认，PE 负责长程信用。',
    status: 'TARGET',
  },
  {
    key: 'evidence',
    start: 7560,
    end: 8160,
    index: '12',
    chapter: 'Diligence-grade evidence',
    title: '架构差异必须被逐个消融，而不是靠完整系统赢一次',
    subtitle:
      '同一基底下分别关闭 State KV、ETA、主动学习、PE 和 LoRA；再用错用户、过期状态、撤销状态做负对照。',
    status: 'HYBRID',
  },
  {
    key: 'outro',
    start: 8160,
    end: 8520,
    index: '13',
    chapter: 'Volvence',
    title: '从一次回答，到一个会持续形成认知的语言模型',
    subtitle:
      '共享能力稳定进化，个体状态即时加载，决策在低维空间持续调整，真实结果进入下一轮学习。',
    status: 'TARGET',
  },
];

export const ARCHITECTURE_DEEP_DIVE_DURATION_FRAMES =
  architectureScenes[architectureScenes.length - 1].end;

