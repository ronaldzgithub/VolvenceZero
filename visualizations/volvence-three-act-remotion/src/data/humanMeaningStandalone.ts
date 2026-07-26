export type HumanMeaningStage =
  | 'parents'
  | 'child-growth'
  | 'child-understood'
  | 'youth'
  | 'startup'
  | 'moonlight'
  | 'past-self'
  | 'ancestor'
  | 'legacy';

export type HumanMeaningNarrationLine = {
  sequence: number;
  start: number;
  end: number;
  stage: HumanMeaningStage;
  image: string;
  chapter: string;
  headline: string;
  supporting: string;
  align: 'left' | 'right' | 'center';
};

export const HUMAN_MEANING_FPS = 30;
export const HUMAN_MEANING_INTRO_FRAMES = 150;
export const HUMAN_MEANING_AUDIO_FRAMES = 6015;
export const HUMAN_MEANING_OUTRO_FRAMES = 240;
export const HUMAN_MEANING_DURATION_FRAMES =
  HUMAN_MEANING_INTRO_FRAMES +
  HUMAN_MEANING_AUDIO_FRAMES +
  HUMAN_MEANING_OUTRO_FRAMES;

export const humanMeaningNarration: HumanMeaningNarrationLine[] = [
  {
    sequence: 1,
    start: 0,
    end: 18.314,
    stage: 'parents',
    image: '01-parents-night.png',
    chapter: '来到世界之前',
    headline: '父母，真的准备好了吗？',
    supporting: '成为父母，也守住一个家',
    align: 'left',
  },
  {
    sequence: 2,
    start: 18.314,
    end: 25.683,
    stage: 'parents',
    image: '01-parents-night.png',
    chapter: '给每一位父母',
    headline: '一个高认知的家庭顾问',
    supporting: '育儿 · 家庭 · 婚姻',
    align: 'left',
  },
  {
    sequence: 3,
    start: 25.683,
    end: 47.281,
    stage: 'child-growth',
    image: '02-child-growth.png',
    chapter: '孩子出生以后',
    headline: '成长，不该只剩模糊的回忆',
    supporting: '记录每一个瞬间，理解每一次表达',
    align: 'right',
  },
  {
    sequence: 4,
    start: 47.281,
    end: 69.168,
    stage: 'child-understood',
    image: '03-child-understood.png',
    chapter: '当他不敢说、说不清、没人懂',
    headline: '谁能站在他的世界里，看见他？',
    supporting: '真正懂他的老师、朋友和家长',
    align: 'left',
  },
  {
    sequence: 5,
    start: 69.168,
    end: 90.654,
    stage: 'youth',
    image: '04-youth-city.png',
    chapter: '再后来，他走进人海',
    headline: '谁是命运里那个对的人？',
    supporting: '懂他的红娘，也是一位知心的朋友',
    align: 'right',
  },
  {
    sequence: 6,
    start: 90.654,
    end: 114.708,
    stage: 'startup',
    image: '05-startup-team.png',
    chapter: '当他创业、承担、不肯认输',
    headline: '把梦想变成收入，把想法变成事业',
    supporting: '销售员 · 内容专员 · 程序员',
    align: 'left',
  },
  {
    sequence: 7,
    start: 114.708,
    end: 134.529,
    stage: 'moonlight',
    image: '06-moonlight-box.png',
    chapter: '到了老去的那一天',
    headline: '如果能打开一只月光宝盒',
    supporting: '你想回到哪一刻？',
    align: 'right',
  },
  {
    sequence: 8,
    start: 134.529,
    end: 144.095,
    stage: 'past-self',
    image: '07-past-self.png',
    chapter: '与过去的自己，再说一次话',
    headline: '那时的我，为何勇敢？为何怯懦？',
    supporting: '终于理解，今天的自己从何而来',
    align: 'center',
  },
  {
    sequence: 9,
    start: 144.095,
    end: 160.658,
    stage: 'ancestor',
    image: '08-ancestor-guidance.png',
    chapter: '甚至在一个人离开之后',
    headline: '他的判断，仍能照亮后来的人',
    supporting: '认知 · 记忆 · 科学观 · 经验',
    align: 'left',
  },
  {
    sequence: 10,
    start: 160.658,
    end: 200.489,
    stage: 'legacy',
    image: '09-ordinary-lives.png',
    chapter: '每一个普通人',
    headline: '都值得，也应该被记住',
    supporting: '我们迁移的，是一个人面对世界的反馈模式',
    align: 'center',
  },
];
