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
  APP_ECOSYSTEM_AUDIO_FRAMES,
  APP_ECOSYSTEM_DURATION_FRAMES,
  APP_ECOSYSTEM_FPS,
  APP_ECOSYSTEM_INTRO_FRAMES,
  appEcosystemNarration,
  type AppEcosystemNarrationLine,
} from './data/appEcosystemStandalone';
import './app-ecosystem-standalone.css';

export {
  APP_ECOSYSTEM_DURATION_FRAMES,
  APP_ECOSYSTEM_FPS,
};

const clamp = {
  extrapolateLeft: 'clamp' as const,
  extrapolateRight: 'clamp' as const,
};

const currentNarrationAt = (time: number): AppEcosystemNarrationLine => {
  let current = appEcosystemNarration[0];
  for (const line of appEcosystemNarration) {
    if (line.start <= time) current = line;
  }
  return current;
};

const ProductOpening: React.FC<{active: boolean}> = ({active}) => (
  <section className={`a6-product-opening ${active ? 'active' : ''}`}>
    <div className="a6-opening-question">
      <span>如果 AI 能在时间中保持连续</span>
      <strong>什么产品，过去根本做不出来？</strong>
    </div>
    <div className="a6-opening-cards">
      {[
        ['01', '与过去的自己对话', 'Moonlight Box · Past Self'],
        ['02', '对一段关系负责', 'Elder · Commitment · Repair30'],
        ['03', '人物拥有生命周期', 'Historical · Character · Novel'],
        ['04', '会成长的数字员工', 'Digital Employee · AutoCompany'],
        ['05', '可授权的生命资产', 'Lifeform Store · Exchange · Accounts'],
      ].map(([id, title, apps]) => (
        <div key={id}>
          <span>{id}</span>
          <b>{title}</b>
          <i>{apps}</i>
        </div>
      ))}
    </div>
  </section>
);

const PastSelfProduct: React.FC<{active: boolean}> = ({active}) => (
  <section className={`a6-past-self ${active ? 'active' : ''}`}>
    <div className="a6-product-label">
      <span>MOONLIGHT BOX + PAST SELF</span>
      <b>过去的你，不再只是一堆资料</b>
    </div>
    <div className="a6-time-rail">
      {[
        ['2018', '刚进入机器学习', '论文笔记 · 录音 · 选择'],
        ['2021', '决定放弃读博', '日记 43 篇 · 对话 218 条'],
        ['2026', '今天的你', '重新理解当时的自己'],
      ].map(([year, title, evidence], index) => (
        <div key={year} className={index === 1 ? 'selected' : ''}>
          <span>{year}</span>
          <b>{title}</b>
          <i>{evidence}</i>
        </div>
      ))}
    </div>
    <div className="a6-past-chat">
      <header>
        <span>2021 年的我</span>
        <b>时间点生命体 · 已恢复</b>
      </header>
      <div className="user">我一直想知道，你当时为什么真的放弃了？</div>
      <div className="assistant">
        不是因为我不喜欢研究。那时我最怕的，是把人生交给一条看不到尽头的路。
        <small>依据：2021-03 日记 · 与导师对话 · 离校前录音</small>
      </div>
    </div>
    <div className="a6-product-difference">
      <s>搜索旧日记</s><i>→</i><b>与当时那套认知继续对话</b>
    </div>
  </section>
);

const RelationshipProduct: React.FC<{active: boolean; frame: number}> = ({active, frame}) => {
  const activeStep = Math.floor((frame / 46) % 4);
  return (
    <section className={`a6-relationship ${active ? 'active' : ''}`}>
      <div className="a6-product-label">
        <span>ELDER COMPANION · COMMITMENT KEEPER · REPAIR30</span>
        <b>产品不是一次聊天，而是一段需要被照料的关系</b>
      </div>
      <div className="a6-relation-person">
        <span>家庭关系</span>
        <strong>母亲 ↔ 女儿</strong>
        <b>同一个陪伴生命体持续在场</b>
      </div>
      <div className="a6-relation-timeline">
        {[
          ['承诺', '周日和女儿视频'],
          ['落空', '连续两次没有接通'],
          ['理解', '母亲不喜欢被催促'],
          ['修复', '先询问意愿，再邀请女儿'],
        ].map(([title, detail], index) => (
          <div key={title} className={index === activeStep ? 'active' : ''}>
            <span>{index + 1}</span>
            <b>{title}</b>
            <i>{detail}</i>
          </div>
        ))}
      </div>
      <div className="a6-relation-message">
        <span>Volvence</span>
        <b>“我记得您不喜欢被催。要不要我先问问女儿今天是否方便？”</b>
        <i>不是提醒任务完成，而是在保护关系中的尊重感</i>
      </div>
      <div className="a6-relation-handoff">
        <span>出现持续低落或关系破裂风险</span>
        <b>请求获授权家人介入</b>
        <i>重要行动可追溯 · 不越过同意边界</i>
      </div>
    </section>
  );
};

const CharacterProduct: React.FC<{active: boolean}> = ({active}) => (
  <section className={`a6-character ${active ? 'active' : ''}`}>
    <div className="a6-product-label">
      <span>HISTORICAL LIFEFORMS · CHARACTER LAB · NOVEL WORLDS</span>
      <b>人物不是一段角色提示，而是一个可以长期存在的“谁”</b>
    </div>
    <div className="a6-character-source">
      <span>证据与创作设定</span>
      <div>论文与书信</div>
      <div>作者访谈</div>
      <div>人物关系</div>
      <div>价值与边界</div>
    </div>
    <div className="a6-character-life">
      <header><span>AI_ID</span><b>einstein:historical:v3</b></header>
      <strong>持久人物生命体</strong>
      <div>
        <span>稳定身份</span>
        <span>人物记忆</span>
        <span>独立关系</span>
        <span>生命周期审核</span>
      </div>
    </div>
    <div className="a6-character-world">
      <header><span>你与这个人物</span><b>第 184 天</b></header>
      <div className="assistant">
        你上次说，想用“思想实验”帮助女儿理解相对性。她后来怎么回应？
      </div>
      <i>人物保持自身认知，同时记得与你共同发生的经历</i>
    </div>
    <div className="a6-product-difference">
      <s>模仿口吻的角色聊天</s><i>→</i><b>可以建立关系、经历时间的人物生命体</b>
    </div>
  </section>
);

const EmployeeProduct: React.FC<{active: boolean; frame: number}> = ({active, frame}) => {
  const highlighted = Math.floor((frame / 55) % 3);
  return (
    <section className={`a6-employee ${active ? 'active' : ''}`}>
      <div className="a6-product-label">
        <span>DIGITAL EMPLOYEE + AUTOCOMPANY</span>
        <b>数字员工不是执行一次任务，而是把工作经历变成能力</b>
      </div>
      <div className="a6-employee-inbox">
        <header><span>本周任务</span><b>内容运营员工 · 江河</b></header>
        {[
          ['已完成', '整理新品发布素材', '结果：互动率 +18%'],
          ['执行中', '生成渠道差异化版本', '沿用上周有效策略'],
          ['待批准', '发布并回复高风险评论', '需要真人复核'],
        ].map(([status, task, result], index) => (
          <div key={task} className={index === highlighted ? 'active' : ''}>
            <span>{status}</span><b>{task}</b><i>{result}</i>
          </div>
        ))}
      </div>
      <div className="a6-employee-memory">
        <span>这名员工已经学会</span>
        <b>什么内容有效</b>
        <b>什么边界不能越过</b>
        <b>哪类结果需要继续观察</b>
        <i>经验属于这名员工，不只属于一次会话</i>
      </div>
      <div className="a6-rent-human">
        <span>申请真人协作</span>
        <strong>“这条公开回复可能引发法律风险，请法务在 20 分钟内复核。”</strong>
        <div><b>预算 ¥120</b><b>不可逆动作冻结</b><b>等待批准</b></div>
      </div>
      <div className="a6-product-difference">
        <s>人寻找一个 AI 工具</s><i>→</i><b>数字员工在边界处主动寻找合适的人</b>
      </div>
    </section>
  );
};

const EconomyProduct: React.FC<{active: boolean}> = ({active}) => (
  <section className={`a6-lifeform-economy ${active ? 'active' : ''}`}>
    <div className="a6-product-label">
      <span>LIFEFORM STORE · CAPABILITY EXCHANGE · ACCOUNTS</span>
      <b>交易的不再只是模型调用，而是一个经过经历训练的数字主体</b>
    </div>
    <div className="a6-store-card featured">
      <span>可领养生命体</span>
      <strong>跨境电商增长顾问</strong>
      <i>工作经历 2,184 次 · 服务关系 46 个 · 通过边界审核</i>
      <div><b>市场研究</b><b>决策推演</b><b>渠道复盘</b></div>
      <small>可调用：研究能力 · 数据工具 · 真人法务复核</small>
    </div>
    <div className="a6-store-card secondary">
      <span>人物生命体</span>
      <strong>作者共读伙伴</strong>
      <i>共同阅读 870 小时 · 关系记忆独立</i>
      <small>仅授权共读与讨论 · 禁止冒充作者发表</small>
    </div>
    <div className="a6-economy-flow">
      {['创建 / 领养', '训练', '审核', '授权', '出租', '结算'].map((item, index) => (
        <React.Fragment key={item}>
          <span>{item}</span>{index < 5 ? <i>→</i> : null}
        </React.Fragment>
      ))}
    </div>
    <div className="a6-economy-value">
      <s>按 Token 购买通用智能</s>
      <b>为身份、经历、能力边界与责任付费</b>
    </div>
    <div className="a6-economy-close">
      <span>这些不是 28 个界面</span>
      <strong>它们是新产品物种出现的第一批样本</strong>
      <i>成熟度因 App 而异 · 关键商业闭环仍以真实证据为准</i>
    </div>
  </section>
);

export const VolvenceAppEcosystemStandalone: React.FC = () => {
  const frame = useCurrentFrame();
  const audioFrame = frame - APP_ECOSYSTEM_INTRO_FRAMES;
  const audioTime = Math.max(0, audioFrame / APP_ECOSYSTEM_FPS);
  const line = currentNarrationAt(audioTime);
  const introOpacity = interpolate(frame, [0, 22, 63, 90], [0, 1, 1, 0], clamp);
  const contentOpacity = interpolate(frame, [70, 106], [0, 1], clamp);
  const outroStart = APP_ECOSYSTEM_INTRO_FRAMES + APP_ECOSYSTEM_AUDIO_FRAMES;
  const outroOpacity = interpolate(frame, [outroStart, outroStart + 28], [0, 1], clamp);

  return (
    <AbsoluteFill className="a6-film">
      <div className="a6-grid" />
      <div className="a6-brand">
        <span>V</span><b>VOLVENCE</b><i>NEW PRODUCT FORMS</i>
      </div>
      <div className="a6-progress">
        {appEcosystemNarration.map((item) => (
          <span key={item.sequence} className={item.sequence === line.sequence ? 'active' : ''} />
        ))}
      </div>

      <div className="a6-intro" style={{opacity: introOpacity}}>
        <span>第六幕 · 产品形态</span>
        <strong>如果 AI 能够持续存在，会诞生什么新产品？</strong>
        <b>从 28 个 App 中，看五种过去不存在的产品</b>
      </div>

      <main className="a6-content" style={{opacity: contentOpacity}}>
        <div className="a6-heading">
          <span>{String(line.sequence).padStart(2, '0')}</span>
          <div>
            <b>{line.caption}</b>
            <i>
              {line.sequence === 1
                ? '不再解释技术，直接看产品'
                : `典型 App · ${line.product === 'past-self'
                    ? 'Moonlight Box / Past Self'
                    : line.product === 'relationship'
                      ? 'Elder Companion / Commitment Keeper / Repair30'
                      : line.product === 'character'
                        ? 'Historical Lifeforms / Character Lab / Novel Worlds'
                        : line.product === 'employee'
                          ? 'Digital Employee / AutoCompany'
                          : 'Lifeform Store / Capability Exchange / Accounts'}`}
            </i>
          </div>
        </div>
        <div className="a6-stage">
          <ProductOpening active={line.product === 'opening'} />
          <PastSelfProduct active={line.product === 'past-self'} />
          <RelationshipProduct active={line.product === 'relationship'} frame={frame} />
          <CharacterProduct active={line.product === 'character'} />
          <EmployeeProduct active={line.product === 'employee'} frame={frame} />
          <EconomyProduct active={line.product === 'economy'} />
        </div>
      </main>

      <div className="a6-outro" style={{opacity: outroOpacity}}>
        <span>VOLVENCE</span>
        <strong>不是把旧产品加上 AI</strong>
        <b>而是创造过去无法存在的新产品物种</b>
        <i>身份持续 · 经历积累 · 关系发展 · 现实行动</i>
      </div>

      <Sequence
        from={APP_ECOSYSTEM_INTRO_FRAMES}
        durationInFrames={APP_ECOSYSTEM_AUDIO_FRAMES}
      >
        <Audio
          src={staticFile('app-ecosystem-standalone/volvence-app-ecosystem-standalone.wav')}
          volume={1}
        />
      </Sequence>
    </AbsoluteFill>
  );
};
