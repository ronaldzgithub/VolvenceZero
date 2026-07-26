export type AppEcosystemNarrationLine = {
  sequence: number;
  start: number;
  end: number;
  caption: string;
  product:
    | 'opening'
    | 'past-self'
    | 'relationship'
    | 'character'
    | 'employee'
    | 'economy';
};

export const APP_ECOSYSTEM_FPS = 30;
export const APP_ECOSYSTEM_INTRO_FRAMES = 90;
export const APP_ECOSYSTEM_AUDIO_FRAMES = 6272;
export const APP_ECOSYSTEM_OUTRO_FRAMES = 180;
export const APP_ECOSYSTEM_DURATION_FRAMES =
  APP_ECOSYSTEM_INTRO_FRAMES +
  APP_ECOSYSTEM_AUDIO_FRAMES +
  APP_ECOSYSTEM_OUTRO_FRAMES;

export const appEcosystemNarration: AppEcosystemNarrationLine[] = [
  {
    sequence: 1,
    start: 0.25,
    end: 19.15,
    caption: '当 AI 不再是一次回答，会出现什么新产品？',
    product: 'opening',
  },
  {
    sequence: 2,
    start: 19.75,
    end: 52.73,
    caption: '产品形态 01：与过去某一时刻的自己继续对话',
    product: 'past-self',
  },
  {
    sequence: 3,
    start: 53.35,
    end: 86.9,
    caption: '产品形态 02：对一段关系持续负责，而不是陪聊一次',
    product: 'relationship',
  },
  {
    sequence: 4,
    start: 87.54,
    end: 123.49,
    caption: '产品形态 03：人物与角色拥有生命周期，而不是扮演',
    product: 'character',
  },
  {
    sequence: 5,
    start: 124.11,
    end: 163.23,
    caption: '产品形态 04：会积累经验、主动调用真人的数字员工',
    product: 'employee',
  },
  {
    sequence: 6,
    start: 163.87,
    end: 208.05,
    caption: '产品形态 05：数字生命本身成为可训练、授权的资产',
    product: 'economy',
  },
];
