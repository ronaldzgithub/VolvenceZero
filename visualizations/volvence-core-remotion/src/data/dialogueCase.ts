export type Speaker = 'user' | 'assistant';

export type DialogueUtterance = {
  id: string;
  speaker: Speaker;
  start: number;
  end: number;
  text: string;
};

export type PanoramaItem = {
  label: string;
  state: string;
  tone: 'task' | 'relation' | 'risk' | 'evidence';
};

export type MechanismPhase = {
  id: string;
  start: number;
  end: number;
  step: string;
  title: string;
  action: string;
  activeQuestion: string;
  taskState: string[];
  relationState: string[];
  panorama: PanoramaItem[];
  memory: string[];
  learning: string;
  activeModules: string[];
};

export const AUDIO_START_SECONDS = 4;
export const AUDIO_DURATION_SECONDS = 138.4;
export const OUTRO_DURATION_SECONDS = 6;

export const dialogueUtterances: DialogueUtterance[] = [
  {
    id: 'u1',
    speaker: 'user',
    start: 0,
    end: 5.3,
    text: '我老公出轨了！他是一家AI公司的老板。我真的忍不了了。我要离婚！',
  },
  {
    id: 'a1',
    speaker: 'assistant',
    start: 5.75,
    end: 17.25,
    text: '这一下，真的很难扛。先别逼自己今天就决定离不离。我先确认最要紧的：你现在安全吗？今晚需要马上离开吗？',
  },
  {
    id: 'u2',
    speaker: 'user',
    start: 18.03,
    end: 25.44,
    text: '我们现在是安全的，孩子也不在旁边。我是在他手机里发现的。他承认，已经半年了。',
  },
  {
    id: 'a2',
    speaker: 'assistant',
    start: 25.89,
    end: 42.29,
    text: '好。至少今晚不用先处理安全问题。半年，而且他已经承认，这不是一次误会。你现在最怕哪件事？孩子受影响、以后没钱，还是你已经没法继续跟他住在一起？',
  },
  {
    id: 'u3',
    speaker: 'user',
    start: 43.07,
    end: 52.53,
    text: '我最怕孩子失去稳定，也怕自己以后没钱。可是我现在每天看到他都觉得恶心，根本没法正常说话。',
  },
  {
    id: 'a3',
    speaker: 'assistant',
    start: 52.98,
    end: 73.49,
    text: '那先分开住，可能比现在逼自己决定离婚更合适。分居不是原谅，也不是最终决定，只是先让你能喘口气。接下来我想弄清两件事：你短期能不能养活自己？家里的钱、房子和公司股权，你知道多少？',
  },
  {
    id: 'u4',
    speaker: 'user',
    start: 74.27,
    end: 85.88,
    text: '我可以重新工作，只是短期挣得不多。房子是共同的，家里现金不多。他做持续学习模型，公司刚融资，但股份要锁三年。',
  },
  {
    id: 'a4',
    speaker: 'assistant',
    start: 86.33,
    end: 107.5,
    text: '先别把刚融资，和以后有钱画等号。股份锁三年，公司也可能涨，也可能跌。我可以帮你查公开的融资和公司情况，但股权到底跟你有没有关系，要让律师看文件。别拿自己的婚姻，去赌一家公司的未来。',
  },
  {
    id: 'u5',
    speaker: 'user',
    start: 108.28,
    end: 116.96,
    text: '所以，我不用今天决定离不离。可以先分开住，把孩子和财产的事弄清楚，再决定。对吗？',
  },
  {
    id: 'a5',
    speaker: 'assistant',
    start: 117.41,
    end: 137.62,
    text: '对。今天先做三件事。告诉一个你信任的人。把能合法拿到的房产、账户和股权材料备份好。再单独约一位家事律师，只问孩子、财产和分居怎么安排。你不是在拖延。你是在把选择权，拿回来。',
  },
];

export const mechanismPhases: MechanismPhase[] = [
  {
    id: 'stabilize',
    start: 0,
    end: 18.03,
    step: '01 先稳住',
    title: '先确认安全，不替用户仓促决定',
    action: '稳定情绪 · 确认安全',
    activeQuestion: '此刻是否存在人身风险？',
    taskState: ['婚姻危机', '离婚冲动', '配偶经营AI公司'],
    relationState: ['强烈背叛感', '情绪过载', '需要恢复选择权'],
    panorama: [
      {label: '安全', state: '优先确认', tone: 'risk'},
      {label: '决定', state: '暂缓', tone: 'task'},
      {label: '关系', state: '先承接', tone: 'relation'},
      {label: '信息', state: '严重不足', tone: 'evidence'},
    ],
    memory: ['出轨事件', '用户明确提出离婚', '配偶为AI公司老板'],
    learning: '建立下一步预测：先降低冲击，用户才可能提供真实情况',
    activeModules: ['input', 'dual', 'decision', 'residual', 'output'],
  },
  {
    id: 'clarify',
    start: 18.03,
    end: 43.07,
    step: '02 找出问题',
    title: '从一句“我要离婚”中找出真正的冲突',
    action: '厘清目标 · 主动追问',
    activeQuestion: '孩子、经济与共同生活，哪一项最难承受？',
    taskState: ['确认安全', '出轨持续半年', '对方已经承认'],
    relationState: ['愤怒仍在', '开始提供事实', '可进入共同建模'],
    panorama: [
      {label: '孩子', state: '待了解', tone: 'relation'},
      {label: '经济', state: '待了解', tone: 'task'},
      {label: '共同生活', state: '待了解', tone: 'risk'},
      {label: '事实基础', state: '逐步清晰', tone: 'evidence'},
    ],
    memory: ['当前安全', '孩子不在现场', '出轨持续半年且已承认'],
    learning: '主动选择最有信息量的问题，而不是继续堆砌离婚建议',
    activeModules: ['input', 'dual', 'panorama', 'question', 'decision', 'residual', 'output'],
  },
  {
    id: 'model',
    start: 43.07,
    end: 74.27,
    step: '03 共同推演',
    title: '把相互冲突的目标变成可行动的临时方案',
    action: '构造可逆选项 · 保留选择权',
    activeQuestion: '什么方案能先止损，又不替用户作最终决定？',
    taskState: ['孩子需要稳定', '短期经济不确定', '无法正常共同生活'],
    relationState: ['厌恶与痛苦', '害怕失去稳定', '需要喘息空间'],
    panorama: [
      {label: '分开住', state: '可逆', tone: 'task'},
      {label: '马上离婚', state: '信息不足', tone: 'risk'},
      {label: '继续同住', state: '当前不可承受', tone: 'relation'},
      {label: '后续决定', state: '保留', tone: 'evidence'},
    ],
    memory: ['孩子稳定是核心目标', '用户担心经济', '共同生活已难以维持'],
    learning: '形成阶段性策略：先恢复行动能力，再补齐财务事实',
    activeModules: ['dual', 'panorama', 'decision', 'residual', 'output', 'memory'],
  },
  {
    id: 'research',
    start: 74.27,
    end: 108.28,
    step: '04 补齐事实',
    title: '自动研究能查什么，专业边界必须交给谁',
    action: '事实求证 · 边界分流',
    activeQuestion: '公司价值、股权归属与短期生活能力分别如何确认？',
    taskState: ['可重新工作', '共同房产', '现金有限', '股权锁定三年'],
    relationState: ['情绪开始稳定', '愿意面对现实', '需要避免财富幻觉'],
    panorama: [
      {label: '公开融资', state: '可以研究', tone: 'evidence'},
      {label: '公司未来', state: '不可承诺', tone: 'risk'},
      {label: '股权归属', state: '律师核验', tone: 'task'},
      {label: '婚姻选择', state: '不押注估值', tone: 'relation'},
    ],
    memory: ['房子为共同财产', '家庭现金有限', '公司刚融资', '股份锁定三年'],
    learning: '区分公开证据、概率判断与法律结论，避免模型越权',
    activeModules: ['input', 'dual', 'research', 'panorama', 'decision', 'residual', 'output', 'memory'],
  },
  {
    id: 'converge',
    start: 108.28,
    end: AUDIO_DURATION_SECONDS,
    step: '05 共同收敛',
    title: '用户形成自己的判断，系统帮助它落到行动',
    action: '确认判断 · 形成下一步',
    activeQuestion: '怎样把重新获得的选择权变成今天能完成的行动？',
    taskState: ['先分开住', '补齐财产信息', '咨询家事律师'],
    relationState: ['从失控到稳定', '判断由用户形成', '选择权回到用户手中'],
    panorama: [
      {label: '支持网络', state: '联系可信任的人', tone: 'relation'},
      {label: '证据材料', state: '合法备份', tone: 'evidence'},
      {label: '专业咨询', state: '聚焦孩子与财产', tone: 'task'},
      {label: '最终决定', state: '不在今天强迫完成', tone: 'risk'},
    ],
    memory: ['用户偏好可逆决策', '孩子稳定与经济安全并重', '需要专业法律支持'],
    learning: '真实结果进入归因：个人记忆形成，策略候选进入验证',
    activeModules: ['dual', 'panorama', 'decision', 'residual', 'output', 'learning', 'memory', 'gate'],
  },
];

export const progressSteps = ['先稳住', '找出问题', '共同推演', '补齐事实', '共同收敛'];
