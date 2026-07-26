import React from 'react';
import {
  AbsoluteFill,
  Audio,
  interpolate,
  Sequence,
  staticFile,
  useCurrentFrame,
} from 'remotion';
import {
  PRODUCT_FOURTH_ACT_AUDIO_FRAMES,
  PRODUCT_FOURTH_ACT_DURATION_FRAMES,
  PRODUCT_FOURTH_ACT_FPS,
  PRODUCT_FOURTH_ACT_INTRO_FRAMES,
  productDialogueLines,
  productStages,
  type ProductDialogueLine,
} from './data/productFourthAct';
import './product-fourth-act.css';

export {
  PRODUCT_FOURTH_ACT_DURATION_FRAMES,
  PRODUCT_FOURTH_ACT_FPS,
};

const clamp = {
  extrapolateLeft: 'clamp' as const,
  extrapolateRight: 'clamp' as const,
};

const currentLineAt = (time: number): ProductDialogueLine => {
  let current = productDialogueLines[0];
  for (const line of productDialogueLines) {
    if (line.start <= time) current = line;
  }
  return current;
};

const stageIndexAt = (sequence: number) =>
  productStages.findIndex(
    (stage) => sequence >= stage.from && sequence <= stage.to,
  );

const communicationMoves: Record<number, string[]> = {
  1: ['承接'],
  2: ['认可', '框定', '邀请'],
  3: ['承接'],
  4: ['框定', '澄清'],
  5: ['确认权重'],
  6: ['主动探询'],
  7: ['归纳'],
  8: ['识别关键变量'],
  9: ['补齐事实'],
  10: ['情景推演'],
  11: ['标记未知'],
  12: ['主动探询'],
  13: ['承接'],
  14: ['重框', '引导'],
  15: ['形成承诺'],
  16: ['总结', '收束'],
};

const BrandChrome: React.FC<{
  sequence: number;
  frame: number;
}> = ({sequence, frame}) => {
  const stageIndex = stageIndexAt(sequence);
  return (
    <>
      <div className="p4-brand">
        <span>V</span>
        <b>VOLVENCE</b>
        <i>独立产品演示</i>
      </div>
      <div className="p4-stage-progress">
        {productStages.map((stage, index) => (
          <div
            key={stage.label}
            className={`${index === stageIndex ? 'active' : ''} ${
              index < stageIndex ? 'visited' : ''
            }`}
          >
            <span>{String(index + 1).padStart(2, '0')}</span>
            <b>{stage.label}</b>
          </div>
        ))}
      </div>
      <div className="p4-time">
        {String(Math.floor(frame / PRODUCT_FOURTH_ACT_FPS / 60)).padStart(2, '0')}:
        {String(Math.floor(frame / PRODUCT_FOURTH_ACT_FPS) % 60).padStart(2, '0')}
      </div>
    </>
  );
};

const ConversationColumn: React.FC<{
  time: number;
  line: ProductDialogueLine;
}> = ({time, line}) => {
  const visible = productDialogueLines
    .filter((item) => item.start <= time + 0.1)
    .slice(-2);
  const moves = communicationMoves[line.sequence] ?? [];
  return (
    <div className="p4-conversation">
      <div className="p4-conversation-head">
        <span>V</span>
        <div>
          <b>Volvence</b>
          <small>和用户一起把问题想清楚</small>
        </div>
        <i>自然对话</i>
      </div>
      <div className="p4-message-list">
        {visible.map((item) => {
          const active =
            item.sequence === line.sequence && time <= item.end + 0.35;
          return (
            <div
              key={item.sequence}
              className={`p4-message ${item.speaker} ${
                active ? 'active' : ''
              }`}
            >
              <span>{item.speaker === 'user' ? '用户' : 'Volvence'}</span>
              <p>{item.displayText}</p>
              {active ? (
                <i className="p4-speaking"><b /><b /><b /><b /></i>
              ) : null}
            </div>
          );
        })}
      </div>
      <div className="p4-communication">
        <span>当前沟通</span>
        <div>
          {moves.map((move) => <b key={move}>{move}</b>)}
        </div>
      </div>
      <div className="p4-user-journey">
        <span>用户状态</span>
        <div>
          {['情绪爆发', '开始思考', '提供事实', '形成判断'].map(
            (state, index) => {
              const thresholds = [1, 3, 7, 15];
              return (
                <React.Fragment key={state}>
                  <b className={line.sequence >= thresholds[index] ? 'active' : ''}>
                    {state}
                  </b>
                  {index < 3 ? <i /> : null}
                </React.Fragment>
              );
            },
          )}
        </div>
      </div>
    </div>
  );
};

const unknowns = [
  {
    label: '自身赚钱能力',
    resolvedAt: 7,
    activeFrom: 6,
    activeTo: 7,
    value: '三个月恢复 · 月入约2万',
  },
  {
    label: '公司真实价值',
    resolvedAt: 9,
    activeFrom: 8,
    activeTo: 11,
    value: 'Volvence型 · 高潜力高波动',
  },
  {
    label: '新关系与支持',
    resolvedAt: 13,
    activeFrom: 12,
    activeTo: 13,
    value: '无备胎 · 有亲友支持',
  },
  {
    label: '情绪执行能力',
    resolvedAt: 14,
    activeFrom: 14,
    activeTo: 15,
    value: '有波动 · 可接受分开三月',
  },
];

const CriticalUnknowns: React.FC<{sequence: number}> = ({sequence}) => (
  <div className={`p4-unknowns ${sequence >= 6 ? 'visible' : ''}`}>
    <div className="p4-section-kicker">
      <span>AI 主动发现</span>
      <b>最能改变判断的四个未知</b>
    </div>
    <div className="p4-unknown-grid">
      {unknowns.map((item) => {
        const resolved = sequence >= item.resolvedAt;
        const active =
          sequence >= item.activeFrom && sequence <= item.activeTo;
        return (
          <div
            key={item.label}
            className={`${resolved ? 'resolved' : ''} ${
              active ? 'active' : ''
            }`}
          >
            <span>{item.label}</span>
            <b>{resolved ? item.value : active ? '正在探索' : '待探索'}</b>
            <i>{resolved ? '✓' : '?'}</i>
          </div>
        );
      })}
    </div>
  </div>
);

const companyScenarios = [
  {
    name: 'OpenAI级别',
    ev: '+8.4',
    range: '价值已验证 · 兑现确定性高',
    tone: 'high',
  },
  {
    name: 'Volvence型',
    ev: '+1.6',
    range: '区间 -3.1 ～ +7.2',
    tone: 'variable',
  },
  {
    name: '普通套壳公司',
    ev: '-2.7',
    range: '资产弱 · 时间成本高',
    tone: 'low',
  },
];

const CompanyScenarioEV: React.FC<{sequence: number}> = ({sequence}) => {
  const visible = sequence >= 8;
  const selected = sequence >= 9;
  const equityUnknown = sequence >= 11;
  return (
    <div className={`p4-company-ev ${visible ? 'visible' : ''}`}>
      <div className="p4-ev-formula">
        <span>等待的期望收益（EV）</span>
        <b>可归属权益 × 兑现概率 × 折现 − 时间成本 − 情绪成本</b>
        <i>示意值，随证据更新</i>
      </div>
      <div className="p4-scenario-cards">
        {companyScenarios.map((scenario) => (
          <div
            key={scenario.name}
            className={`${scenario.tone} ${
              selected && scenario.name === 'Volvence型' ? 'selected' : ''
            }`}
          >
            <span>{scenario.name}</span>
            <strong>{scenario.ev}</strong>
            <small>{scenario.range}</small>
            {scenario.name === 'Volvence型' && equityUnknown ? (
              <i>股权归属待核验，继续折价</i>
            ) : null}
          </div>
        ))}
      </div>
    </div>
  );
};

const dimensions = [
  {name: '安全底线', weight: '10%', revealAt: 4, scores: ['6', '5', '9', '4']},
  {name: '孩子稳定', weight: '28%', revealAt: 5, scores: ['4', '7', '8', '8']},
  {name: '经济权益', weight: '28%', revealAt: 11, scores: ['5', '6', '8', '7']},
  {name: '情绪恢复', weight: '22%', revealAt: 14, scores: ['5', '6', '9', '3']},
  {name: '修复可能', weight: '7%', revealAt: 5, scores: ['3', '8', '5', '7']},
  {name: '支持系统', weight: '5%', revealAt: 13, scores: ['7', '6', '8', '6']},
];

const DecisionMatrix: React.FC<{sequence: number}> = ({sequence}) => {
  const options = ['马上离', '谈好再离', '暂时分开', '继续修复'];
  const totals = ['4.8', '6.3', '8.1', '6.1'];
  const weightsReady = sequence >= 5;
  const converged = sequence >= 16;
  return (
    <div className={`p4-matrix ${sequence >= 4 ? 'visible' : ''}`}>
      <div className="p4-matrix-title">
        <div>
          <span>全景决策表</span>
          <b>选项不变，事实、权重和分数持续更新</b>
        </div>
        <i>{converged ? '本轮已收敛' : '动态评估中'}</i>
      </div>
      <div className="p4-matrix-grid">
        <div className="p4-matrix-corner">维度 <span>重要性</span></div>
        {options.map((option, index) => (
          <div
            key={option}
            className={`p4-option ${
              index === 2 && sequence >= 14 ? 'leading' : ''
            }`}
          >
            {option}
          </div>
        ))}
        {dimensions.map((dimension) => {
          const revealed = sequence >= dimension.revealAt;
          return (
            <React.Fragment key={dimension.name}>
              <div className={`p4-dimension ${revealed ? 'revealed' : ''}`}>
                <b>{dimension.name}</b>
                <span>{weightsReady ? dimension.weight : '待确认'}</span>
              </div>
              {dimension.scores.map((score, index) => (
                <div
                  key={`${dimension.name}-${options[index]}`}
                  className={`p4-score ${revealed ? 'revealed' : ''} ${
                    index === 2 && revealed && sequence >= 14 ? 'leading' : ''
                  }`}
                >
                  {revealed ? score : '·'}
                </div>
              ))}
            </React.Fragment>
          );
        })}
        <div className="p4-total-label">
          <b>综合EV</b>
          <span>{converged ? '已收敛' : '持续更新'}</span>
        </div>
        {totals.map((total, index) => (
          <div
            key={`${options[index]}-total`}
            className={`p4-score total ${converged ? 'revealed' : ''} ${
              index === 2 && converged ? 'winner' : ''
            }`}
          >
            {converged ? total : '—'}
          </div>
        ))}
      </div>
    </div>
  );
};

const ExplorationFocus: React.FC<{sequence: number}> = ({sequence}) => {
  if (sequence >= 16) {
    return (
      <div className="p4-final-plan">
        <div>
          <span>当前最优可逆方案</span>
          <strong>先分开三个月</strong>
        </div>
        <ul>
          <li>三个月恢复收入</li>
          <li>稳定孩子安排</li>
          <li>合法备份资产材料</li>
          <li>律师核验股权归属</li>
          <li>观察是否承担责任</li>
          <li>三个月后重新计算</li>
        </ul>
      </div>
    );
  }
  if (sequence >= 12) {
    return (
      <div className="p4-relation-eval">
        <div className={sequence >= 13 ? 'resolved' : 'active'}>
          <span>有没有备胎？</span>
          <b>{sequence >= 13 ? '没有新关系' : '正在询问'}</b>
          <small>不把不存在的新关系计入收益</small>
        </div>
        <div className={sequence >= 14 ? 'resolved' : ''}>
          <span>能否稳定执行？</span>
          <b>{sequence >= 14 ? '情绪波动' : '待确认'}</b>
          <small>提高可逆方案价值，降低冲动决策价值</small>
        </div>
      </div>
    );
  }
  if (sequence >= 8) {
    return <CompanyScenarioEV sequence={sequence} />;
  }
  return (
    <div className={`p4-framework-note ${sequence >= 4 ? 'visible' : ''}`}>
      <span>{sequence < 4 ? '第一步' : sequence < 6 ? '共同建模' : '开始探索'}</span>
      <b>
        {sequence < 4
          ? '先确认用户愿意共同思考'
          : sequence < 6
            ? '先固定选项与维度，再确认重要性'
            : '不平均提问，只找最能改变判断的未知'}
      </b>
    </div>
  );
};

const DecisionWorkspace: React.FC<{sequence: number}> = ({sequence}) => {
  const stage = productStages[stageIndexAt(sequence)] ?? productStages[0];
  return (
    <div className="p4-workspace">
      <div className="p4-workspace-head">
        <div>
          <span>{stage.label}</span>
          <b>{stage.caption}</b>
        </div>
        <i>全景决策支持</i>
      </div>
      <CriticalUnknowns sequence={sequence} />
      <ExplorationFocus sequence={sequence} />
      <DecisionMatrix sequence={sequence} />
    </div>
  );
};

export const VolvenceProductFourthAct: React.FC = () => {
  const frame = useCurrentFrame();
  const audioFrame = frame - PRODUCT_FOURTH_ACT_INTRO_FRAMES;
  const audioTime = Math.max(0, audioFrame / PRODUCT_FOURTH_ACT_FPS);
  const line = currentLineAt(audioTime);
  const introOpacity = interpolate(frame, [0, 22, 68, 90], [0, 1, 1, 0], clamp);
  const contentOpacity = interpolate(frame, [70, 105], [0, 1], clamp);
  const outroStart =
    PRODUCT_FOURTH_ACT_INTRO_FRAMES + PRODUCT_FOURTH_ACT_AUDIO_FRAMES;
  const outroOpacity = interpolate(
    frame,
    [outroStart, outroStart + 26],
    [0, 1],
    clamp,
  );

  return (
    <AbsoluteFill className="p4-film">
      <div className="p4-ambient p4-ambient-a" />
      <div className="p4-ambient p4-ambient-b" />
      <div className="p4-grid-bg" />
      <BrandChrome sequence={line.sequence} frame={frame} />

      <div className="p4-intro" style={{opacity: introOpacity}}>
        <span>VOLVENCE · 产品能力实演</span>
        <strong>重大决定，不是给答案</strong>
        <b>而是共同建模 · 共同探索 · 共同收敛</b>
      </div>

      <div className="p4-content" style={{opacity: contentOpacity}}>
        <ConversationColumn time={audioTime} line={line} />
        <DecisionWorkspace sequence={line.sequence} />
      </div>

      <div className="p4-outro" style={{opacity: outroOpacity}}>
        <span>VOLVENCE</span>
        <strong>不是替用户决定</strong>
        <b>是帮用户把选择权拿回来</b>
        <div>
          {['全景决策支持', '高情商教练', '主动发现关键未知', '情景EV推演', '持续重算'].map(
            (item) => <i key={item}>{item}</i>,
          )}
        </div>
      </div>

      <Sequence
        from={PRODUCT_FOURTH_ACT_INTRO_FRAMES}
        durationInFrames={PRODUCT_FOURTH_ACT_AUDIO_FRAMES}
      >
        <Audio
          src={staticFile('product-fourth-act/volvence-product-fourth-act.wav')}
          volume={1}
        />
      </Sequence>
    </AbsoluteFill>
  );
};
