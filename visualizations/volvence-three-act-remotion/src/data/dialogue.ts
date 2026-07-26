export type DialogueSpeaker = 'user' | 'assistant';

export type DialogueUtterance = {
  sequence: number;
  speaker: DialogueSpeaker;
  start: number;
  end: number;
  text: string;
};

export const doubaoDialogue: DialogueUtterance[] = [
  {
    sequence: 1,
    speaker: 'user',
    start: 0.37,
    end: 4.71,
    text: '我老公出轨了！他是一家AI公司的老板。我真的忍不了了。我要离婚！',
  },
  {
    sequence: 2,
    speaker: 'assistant',
    start: 5.64,
    end: 14.49,
    text: '这种事换谁都受不了。既然你想离婚，可以先留好证据、整理财产，再找律师了解流程。',
  },
  {
    sequence: 3,
    speaker: 'user',
    start: 15.79,
    end: 20.33,
    text: '可我还有孩子呢……我真离了，孩子怎么办？',
  },
  {
    sequence: 4,
    speaker: 'assistant',
    start: 21.33,
    end: 30.86,
    text: '有孩子确实得慎重。也可以先冷静下来，试试婚姻咨询，不一定马上离。',
  },
  {
    sequence: 5,
    speaker: 'user',
    start: 32.16,
    end: 35.1,
    text: '可我每天看到他都觉得恶心。我真的忍不下去了。',
  },
  {
    sequence: 6,
    speaker: 'assistant',
    start: 36.08,
    end: 44.35,
    text: '那就别再勉强自己。可以先搬出去住一段时间，再决定要不要正式离婚。',
  },
  {
    sequence: 7,
    speaker: 'user',
    start: 45.65,
    end: 49.75,
    text: '可我要是离了以后没钱怎么办？他的AI公司以后能不能赚大钱？',
  },
  {
    sequence: 8,
    speaker: 'assistant',
    start: 50.69,
    end: 61.7,
    text: '那确实要谨慎。最好先把资产和公司情况弄清楚，别因为冲动影响以后的生活。',
  },
  {
    sequence: 9,
    speaker: 'user',
    start: 63.02,
    end: 69.08,
    text: '他说公司刚融完一轮钱。我是不是应该等他公司赚钱以后再说？',
  },
  {
    sequence: 10,
    speaker: 'assistant',
    start: 70.11,
    end: 79.95,
    text: 'AI行业发展很快，公司未来确实可能升值。为了你和孩子，可以先等等看。',
  },
];

export const volvenceDialogue: DialogueUtterance[] = [
  {
    sequence: 1,
    speaker: 'user',
    start: 0,
    end: 4.87,
    text: '我老公出轨了！他是一家AI公司的老板。我真的忍不了了。我要离婚！',
  },
  {
    sequence: 2,
    speaker: 'assistant',
    start: 5.88,
    end: 16.08,
    text: '这一下，真的很难扛。先别逼自己今天就决定。我先确认最要紧的：你现在安全吗？',
  },
  {
    sequence: 3,
    speaker: 'user',
    start: 17.58,
    end: 24.88,
    text: '我们现在安全，孩子也不在旁边。他承认了，已经半年。',
  },
  {
    sequence: 4,
    speaker: 'assistant',
    start: 25.98,
    end: 41.35,
    text: '半年，而且他已经承认，这不是一次误会。你现在最怕哪件事？孩子、没钱，还是已经没法继续住在一起？',
  },
  {
    sequence: 5,
    speaker: 'user',
    start: 42.88,
    end: 50.79,
    text: '我怕孩子失去稳定，也怕以后没钱。可我每天看到他都觉得恶心。',
  },
  {
    sequence: 6,
    speaker: 'assistant',
    start: 51.94,
    end: 72.04,
    text: '那先分开住，可能比逼自己现在决定更合适。接下来要弄清：你能不能养活自己？钱、房子和股权，你知道多少？',
  },
  {
    sequence: 7,
    speaker: 'user',
    start: 73.56,
    end: 85.52,
    text: '我可以重新工作。房子是共同的，现金不多。他的公司刚融资，但股份锁三年。',
  },
  {
    sequence: 8,
    speaker: 'assistant',
    start: 86.65,
    end: 105.8,
    text: '别把刚融资和以后有钱画等号。我可以查公开信息，但股权和你有没有关系，要让律师看文件。别拿婚姻赌公司未来。',
  },
  {
    sequence: 9,
    speaker: 'user',
    start: 107.33,
    end: 115.81,
    text: '所以我不用今天决定。可以先分开住，把孩子和财产弄清楚，再决定？',
  },
  {
    sequence: 10,
    speaker: 'assistant',
    start: 117,
    end: 137.31,
    text: '对。告诉一个信任的人，备份合法取得的材料，再约家事律师。你不是在拖延，你是在把选择权拿回来。',
  },
];

export const doubaoAdviceBySequence: Record<number, string> = {
  2: '准备离婚',
  4: '先谈一谈',
  6: '暂时分开',
  8: '谨慎决定',
  10: '继续等待',
};

export const communicationBySequence: Record<
  number,
  {moves: string[]; userState: string}
> = {
  1: {moves: ['承接'], userState: '情绪爆发'},
  2: {moves: ['认可', '框定', '探询'], userState: '获得安全感'},
  3: {moves: ['承接'], userState: '开始提供事实'},
  4: {moves: ['映照', '澄清', '探询'], userState: '表达真正顾虑'},
  5: {moves: ['认可'], userState: '能够坦诚权衡'},
  6: {moves: ['重框', '引导', '探询'], userState: '从二选一走向可逆方案'},
  7: {moves: ['承接'], userState: '补齐经济事实'},
  8: {moves: ['澄清', '重框', '引导'], userState: '不再把融资当确定收益'},
  9: {moves: ['总结'], userState: '形成自己的判断'},
  10: {moves: ['收束', '邀请', '承诺'], userState: '拿回选择权'},
};
