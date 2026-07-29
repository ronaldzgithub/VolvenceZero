export const FPS = 30;
export const DURATION = 110 * FPS;

export type CaptionCue = {
  start: number;
  end: number;
  text: string;
  accent?: string;
};

export const captions: CaptionCue[] = [
  {start: 15, end: 180, text: 'Volvence，深耕持续学习大语言模型', accent: '走向通用人工智能'},
  {start: 180, end: 360, text: '20人核心团队，来自全球顶尖院校', accent: '科研 · 工程 · 商业'},
  {start: 360, end: 540, text: '40余篇机器学习论文，18篇国际顶会顶刊'},
  {start: 540, end: 650, text: '主动学习全球前列，原创主动学徒学习'},

  {start: 675, end: 825, text: '一个API，创造能够持续成长的数字生命'},
  {start: 825, end: 990, text: '独立人格、长期记忆、情感感知', accent: '无需复杂上下文工程'},
  {start: 990, end: 1155, text: '传统大模型：闭环由人完成'},
  {start: 1155, end: 1365, text: 'Volvence：从交互与结果中理解人、学会相处'},

  {start: 1395, end: 1545, text: '历史人物、艺术形象、真实个体', accent: '都可以成为持续成长的数字分身'},
  {start: 1545, end: 1710, text: '母婴专家 · 小鹿听声 · 特级教师分身'},
  {start: 1710, end: 1875, text: 'Mira数字闺蜜 · AI红娘 · 数字员工'},
  {start: 1875, end: 2040, text: '数字医生 · 跨境数字专员 · 数字永续'},
  {start: 2070, end: 2220, text: '让每个家庭，拥有可靠的智能顾问'},
  {start: 2220, end: 2370, text: '让每个孩子被理解，让每个青年被陪伴'},
  {start: 2370, end: 2510, text: '让创业者拥有能创造收益的数字同事'},
  {start: 2510, end: 2625, text: '每一个来过世间的人，都值得被记住'},

  {start: 2660, end: 2810, text: '我们正与各领域头部伙伴深度合作'},
  {start: 2810, end: 3020, text: '母婴 · 内容 · 出海 · 医疗 · 教育'},
  {start: 3040, end: 3180, text: 'Volvence', accent: '模型进化，生命涌现'},
  {start: 3180, end: 3285, text: '每个人，都值得被记住'},
];

export const applications = [
  {mark: '育', name: '母婴专家', note: '孕育 · 0—6岁'},
  {mark: '听', name: '小鹿听声', note: '儿童情绪 · 亲子'},
  {mark: '师', name: '特级教师分身', note: '学情 · 家校共育'},
  {mark: '伴', name: 'Mira数字闺蜜', note: '倾听 · 陪伴'},
  {mark: '缘', name: 'AI红娘', note: '婚恋 · 沟通'},
  {mark: '工', name: '数字员工', note: '客服 · 销售 · 运营'},
  {mark: '医', name: '数字医生', note: '随访 · 健康宣教'},
  {mark: '海', name: '跨境数字专员', note: '内容 · 出海'},
  {mark: '续', name: '数字永续', note: '人格 · 记忆'},
] as const;

export const partners = [
  '国内头部MCN',
  '宝宝树',
  '千万级内容创作者',
  '头部出海服务平台',
  '卫健委直属医疗单位',
  'K12课改专家',
] as const;
