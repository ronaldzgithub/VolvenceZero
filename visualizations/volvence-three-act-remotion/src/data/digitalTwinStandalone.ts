export type DigitalTwinNarrationLine = {
  sequence: number;
  start: number;
  end: number;
  caption: string;
  stage: 'question' | 'sources' | 'align' | 'core' | 'proof' | 'verify' | 'deploy' | 'learn';
};

export const DIGITAL_TWIN_FPS = 30;
export const DIGITAL_TWIN_INTRO_FRAMES = 90;
export const DIGITAL_TWIN_AUDIO_FRAMES = 6865;
export const DIGITAL_TWIN_OUTRO_FRAMES = 180;
export const DIGITAL_TWIN_DURATION_FRAMES =
  DIGITAL_TWIN_INTRO_FRAMES +
  DIGITAL_TWIN_AUDIO_FRAMES +
  DIGITAL_TWIN_OUTRO_FRAMES;

export const digitalTwinNarration: DigitalTwinNarrationLine[] = [
  {
    sequence: 1,
    start: 0.25,
    end: 17.76,
    caption: '数字分身，不是换一张脸、复制一种声音',
    stage: 'question',
  },
  {
    sequence: 2,
    start: 18.36,
    end: 50.27,
    caption: '六类人生证据，共同还原同一个人',
    stage: 'sources',
  },
  {
    sequence: 3,
    start: 50.83,
    end: 73.1,
    caption: '不是把资料塞进检索库，而是先对齐证据',
    stage: 'align',
  },
  {
    sequence: 4,
    start: 73.7,
    end: 114.25,
    caption: '把资料编译成四个可运行、可更新的个人内核',
    stage: 'core',
  },
  {
    sequence: 5,
    start: 114.89,
    end: 141.82,
    caption: '为什么它是这个人，而不只是像这个人',
    stage: 'proof',
  },
  {
    sequence: 6,
    start: 142.46,
    end: 171.91,
    caption: '用未见问题和历史选择，验证认知连续性',
    stage: 'verify',
  },
  {
    sequence: 7,
    start: 172.53,
    end: 197.95,
    caption: '共享模型承载能力，个人状态承载这个人',
    stage: 'deploy',
  },
  {
    sequence: 8,
    start: 198.55,
    end: 227.82,
    caption: '数字分身上线以后，仍会沿着这个人的轨迹成长',
    stage: 'learn',
  },
];
