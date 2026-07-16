# Runtime Ingestion + Apprenticeship Trigger Spec

> Status: draft
> Last updated: 2026-07-16
> 对应需求: R6, R8, R10, R15
> 来源: `docs/implementation/13_emogpt_prd_alignment_upgrade.md` Gap 2 + Gap 3

## 要解决的问题

VolvenceZero 当前的"经验入口"只有两条：

1. **离线 scenario**：`lifeform-evolution.scenario_pack` 把 JSON scenario 文件喂给 super-loop / benchmark 训练管线
2. **在线 turn**：`LifeformSession.run_turn(user_input)` 走完整 cognition pipeline

但产品场景需要第三类入口：

- 操作员上传一份 PDF / 一组网页 / 历史聊天记录，让 lifeform"读完然后吸收"
- 工具自动产出的长结构化结果（`task_result` 大于一两个 chunk 的情况）
- DLaaS Teaching Case：操作员示范"这种问题应该这样回答"，让 lifeform 学会

EmoGPT v4 PRD §10 给出 5 种 ContentSource（novel / materials / persona_research / real_person / imagine）+ 学徒模式 forced compliance + ANALYZE_AND_CLASSIFY_MATERIALS atomic 入口。本 spec 把这套需求映射到 VolvenceZero 第一性设计：

- **不**为 5 种源各写一个 ContentSource 类，**不**新增"学徒 pipeline"
- **不**新增门控；forced compliance 复用 `WiringLevel.SHADOW` + vitals override
- 把所有摄入收敛成两条新增能力：
  - **Apprenticeship trigger**（Gap 2，lifeform-core 内部扩展）：给 turn 加 `trigger_kind` 标签
  - **Runtime Ingestion adapter**（Gap 3，新 wheel）：把外部 corpus 切片成多次 `run_turn` 调用，每次带 `trigger_kind="ingestion"`
- durable 化**只能**走 R6 session-post slow loop，不存在"特殊学习路径"
- 在 Environment Interface 口径下，ingestion 是 `Observe` 面的 source adapter：它把外部 corpus 规范化为 canonical turn / event 输入，不成为独立学习管线或 owner。

## 关键不变量

- **不**新增 owner、**不**新增 kernel slot
- 所有 ingestion 内容**只能**通过 `LifeformSession.run_turn(chunk_text, trigger_kind="ingestion")` 进入 kernel
- ingestion 不可直接修改 `memory` / `case_memory` / `domain_knowledge` / `regime` / `temporal` 任一 owner 状态
- ingestion 期间 vitals override 默认 `apprenticeship`：drive 不消耗、PE_weight 减弱（band 内 deviation = 0），但 perception / temporal / regime PE 仍正常计算
- `lifeform-ingestion` wheel 不可被任何 `vz-*` 反向 import；CI 由 `tests/contracts/test_import_boundaries.py` 强制
- chunk 失败必须**显式记录**到 `IngestionEnvelope.partial_failures`，**禁止**静默吞错（红线 C）
- ingestion adapter **禁止**做任何"内容关键词分类"：source_kind 由文件类型 / URL 协议决定，不是 LLM 关键词判定
- web ingestion 第一阶段**禁止** Playwright / 浏览器自动安装，只用 `requests` + `readability-lxml`
- ingestion session 与用户 session 隔离：`session_id` 命名前缀 `ingestion-` 强制，host 不可在 ingestion session 与用户 session 之间共享 transcript
- ingestion 产生的 evidence 必须能回溯到 Environment Event / envelope provenance；durable 化仍只能通过标准 PE + session-post slow loop。
- fictional character 的“主观 live-through”不得以 raw `BOOK` ingestion 作为 ACTIVE 主路径。BOOK ingestion 表示“读到一份材料”；角色亲历必须先产出 reviewed chapter artifact（主观 setting / decision / outcome / semantic events），再通过 character replay driver 进入 canonical turn path。

## 工程挑战

- chunk 切片粒度：太粗（一个 turn 喂 50KB）会让 PE 抖动失真；太细（一个 turn 喂一句话）让 reflection 写回压力暴增
- 异步并发：ingestion 可能跑很久，期间 user turn 不能被阻塞
- 重复 ingestion：同一文档第二次喂应当被 memory consolidation 识别（PE 显著降低）—— 这是验收 gate
- DLaaS TeachingCase 路径：操作员上传 → service 层 → ingestion adapter → 跑完后期望 family report 提升

## 算法候选

来自 `docs/next_gen_emogpt.md`：

| 算法 | 用途 |
|---|---|
| NL CMS 多频率层 | ingestion 内容由 high-freq 层吸收，consolidation 把它压到 low-freq 层 |
| ETA SSL pass | ingestion 是离线 SSL 序列的"在线版本"——把外部文本作为行为压缩输入 |
| R-PE 主链 | ingestion chunk 同样产生 PE，是慢反思的输入 |

## 接口契约

### Gap 2：Apprenticeship trigger（lifeform-core 内部扩展）

`lifeform-core/types.py`：

```python
class TurnTriggerKind(str, Enum):
    USER_INPUT = "user_input"
    INTERNAL_DRIVE = "internal_drive"
    FOLLOWUP_DUE = "followup_due"
    TOOL_RESULT = "tool_result"
    APPRENTICE = "apprentice"           # 操作员示教 turn
    INGESTION = "ingestion"             # 长 corpus 切片注入

@dataclass(frozen=True)
class TurnSummary:                       # 已有 dataclass 扩展
    # ... 现有字段保留 ...
    trigger_kind: TurnTriggerKind = TurnTriggerKind.USER_INPUT
```

`LifeformSession.run_turn`:

```python
async def run_turn(
    self,
    user_input: str,
    *,
    trigger_kind: TurnTriggerKind = TurnTriggerKind.USER_INPUT,
) -> AgentTurnResult:
    """Run one cognition turn.

    Apprenticeship & ingestion semantics:
    - When ``trigger_kind in {APPRENTICE, INGESTION}``, vitals are put into
      apprenticeship override for this turn only. Drive deviation contributes
      0 to slow-scale PE; per-turn recharge is still applied.
    - Cognition pipeline (perception / temporal / memory / regime / reflection)
      runs IDENTICALLY to a normal user turn. There is no "special learning
      pipeline".
    - Scene closure logic is unchanged: scene closes either by idle or
      explicit ``end_scene()``; durable consolidation goes through the same
      session-post slow loop.
    """
    ...
```

`VitalsModule.with_apprentice_override(enabled: bool)`：

```python
class VitalsModule:
    def with_apprentice_override(self, enabled: bool) -> "VitalsModule":
        """Return a new VitalsModule with apprenticeship override toggled.

        Apprenticeship override means:
        - drive deviation outside band still computed but contributes 0 to
          slow-scale PE for the duration of the override
        - per-turn recharge still applied (the lifeform IS engaged in a turn)
        - decay_per_tick still applies if a SYSTEM tick fires during the turn
          (not the override's job to suppress decay)
        """
        ...
```

`LifeformSession` 在每个 turn 结束时**必须** assert override 状态被复原（防泄漏）。

### Gap 3：Runtime Ingestion adapter（新 wheel）

```
packages/lifeform-ingestion/
├── pyproject.toml              # depends on lifeform-core only
└── src/lifeform_ingestion/
    ├── __init__.py
    ├── envelope.py             # IngestionEnvelope / IngestionChunk
    ├── pipeline.py             # IngestionPipeline
    └── sources/
        ├── __init__.py
        ├── book.py             # PDF / DOCX / TXT
        ├── web.py              # plain HTML via requests + readability-lxml
        └── task_result.py      # structured JSON 大结果
```

`envelope.py`：

```python
class IngestionSourceKind(str, Enum):
    BOOK = "book"
    WEB = "web"
    TASK_RESULT = "task_result"
    CORPUS = "corpus"            # 通用文本 corpus

class IngestionComplianceProfile(str, Enum):
    FORCED = "forced"            # 全部 chunk 喂入，无视 vitals
    CONSULTATIVE = "consultative"  # 普通 turn，vitals 正常消耗

@dataclass(frozen=True)
class IngestionProvenance:
    uploader: str                # 操作员 ID / "system"
    upload_ts_ms: int
    source_uri: str              # file path / URL
    integrity_hash: str          # SHA256 of original source

@dataclass(frozen=True)
class IngestionChunk:
    chunk_id: str
    text: str
    locator: str                 # "page=3,offset=1024" / "url+offset=2048" / "row=42"
    confidence: float            # 解析置信度 [0, 1]
    parse_error: str = ""        # 非空表示该 chunk 解析失败但保留位置占位

@dataclass(frozen=True)
class IngestionEnvelope:
    envelope_id: str
    source_kind: IngestionSourceKind
    chunks: tuple[IngestionChunk, ...]
    provenance: IngestionProvenance
    compliance_profile: IngestionComplianceProfile = IngestionComplianceProfile.FORCED
    partial_failures: tuple[str, ...] = ()      # chunk_id list that fully failed
```

`pipeline.py`：

```python
class IngestionPipeline:
    async def process_envelope(
        self,
        env: IngestionEnvelope,
        *,
        session: LifeformSession,
        max_concurrency: int = 1,
        end_scene_after: bool = True,
    ) -> IngestionReport:
        """Feed each chunk through ``session.run_turn(chunk.text, trigger_kind=INGESTION)``.

        Runs sequentially by default (max_concurrency=1) because chunks are
        often co-dependent (later chunks reference earlier facts). Concurrency
        > 1 is allowed for explicitly independent corpora and the caller
        accepts that ordering effects on memory consolidation are not
        guaranteed.

        After all chunks are processed, optionally calls
        ``session.end_scene(reason='ingestion-end')`` so the R6 session-post
        slow loop fires on the ingested material.
        """
```

每个 source adapter 实现 `def to_envelope(input) -> IngestionEnvelope`，**只**做切片 + 解析，**不**调用 kernel。

**Web source 限制**：

- 第一阶段使用 `requests` + `readability-lxml`，**禁止** Playwright / 浏览器自动安装
- 强制 timeout（默认 10s）+ retry（默认 1 次）
- 解析失败的 page → `parse_error` 字段 + chunk 保留 locator，**禁止**静默丢弃
- 不绕过 robots.txt（`requests` 调用前显式 `urllib.robotparser` 检查）

### Service-side TeachingCase 接入

`lifeform-service`（属于 Gap 6 阶段）的 teaching_case endpoint：

```python
async def handle_teaching_case_submit(req: web.Request) -> web.Response:
    payload = await req.json()
    # 1. 写 service 层 DB 行
    case = await db.insert_teaching_case(payload)

    # 2. 翻译成 IngestionEnvelope
    env = IngestionEnvelope(
        envelope_id=f"teaching-{case.case_id}",
        source_kind=IngestionSourceKind.CORPUS,
        chunks=(
            IngestionChunk(
                chunk_id=f"{case.case_id}-user",
                text=case.simulated_user_turns,
                locator=f"teaching_case={case.case_id};field=user_turns",
                confidence=1.0,
            ),
            IngestionChunk(
                chunk_id=f"{case.case_id}-ai",
                text=case.ideal_ai_response,
                locator=f"teaching_case={case.case_id};field=ai_response",
                confidence=1.0,
            ),
        ),
        provenance=IngestionProvenance(...),
        compliance_profile=IngestionComplianceProfile.FORCED,
    )

    # 3. 通过 ingestion pipeline 注入 lifeform session
    pipeline = IngestionPipeline()
    report = await pipeline.process_envelope(env, session=session)
    return web.json_response(report.to_json())
```

**关键不变量**：service 层只调 `IngestionPipeline.process_envelope`，不直接戳 `LifeformSession.run_turn`。

## ETA / NL 集成

- **R6 慢反思**：每个 ingestion scene 闭合后跟 R6 session-post slow loop，consolidate ingested 内容到 background-slow 频带
- **R8 owner 单写者**：ingestion 永远走 `run_turn` → owner snapshot 主链；adapter 不持有 owner store 引用
- **R10 有界自修改**：apprenticeship override 是有界的（仅当前 turn）；ModificationGate 在 substrate 层默认仍 frozen
- **R15 可回滚**：每个 envelope 有 `envelope_id`，reflection writeback 的 evidence 链可追溯到 envelope；如果 ingestion 后行为劣化，operator 可在 service 层把 case retire（已有 lifecycle 机制）

## 当前 proof surface

引入后必须能证明的命题：

1. `ingestion-uses-canonical-turn-only`
   - `lifeform-ingestion` 包内 grep 找不到任何 `MemoryStore` / `RegimeModule` / `CaseMemoryStore` 直接 import
   - acceptance：`tests/contracts/test_ingestion_isolation.py`
2. `repeat-ingestion-shows-consolidation`
   - 同一份 5KB MD 喂两次，第二次 PE magnitude（aggregate）显著低于第一次（≥ 30% 下降）
   - acceptance：`tests/lifeform_e2e/test_book_ingestion_consolidation.py`
3. `apprenticeship-override-leak-free`
   - 100 次随机交错 user / apprenticeship turn 后，vitals override 状态机正确收敛
   - acceptance：`tests/lifeform_e2e/test_apprentice_no_leak.py`
4. `ingestion-does-not-block-user-turn`
   - 异步 ingestion 期间 user turn P95 延迟漂移 ≤ 5%
   - acceptance：`lifeform-bench --concurrent-ingestion` 报告
5. `partial-failure-is-explicit`
   - PDF 含 1 个无法解析的 page，envelope 必须有 `partial_failures != ()` 且对应 chunk 的 `parse_error != ""`
   - acceptance：`tests/ingestion/test_partial_failure_explicit.py`

## 接口契约（公开数据流向）

**消费的输入**：

- 外部文件 / URL / 结构化结果
- `LifeformSession`（进入 kernel 的唯一通道）

**产出的输出**：

- `IngestionReport`（总 turn 数 / 总 chunk 数 / partial_failures / 总 PE 累积 / scene_id）
- 通过 `LifeformSession.run_turn` 间接产生：所有标准 turn snapshot

**ingestion 不发布 kernel slot**。它是 lifeform 层 adapter，副作用通过 turn 路径生效。

## 与其他能力域的关系

| 关系 | 能力域 | 说明 |
|---|---|---|
| 依赖 | Environment Interface | ingestion 是外部材料进入系统的 `Observe` source adapter，必须归并到 canonical event / turn 路径 |
| 依赖 | 契约式运行时 | 所有 chunk 走 `run_turn` 主链 |
| 依赖 | 连续记忆系统 | consolidation 路径由 R6 slow loop 处理 |
| 依赖 | Lifeform Vitals | apprenticeship override 走 vitals 已有机制 |
| 协作 | 信用分配与自修改 | ingestion outcome 可作为 RewardRecord 的 evidence 链来源 |
| 协作 | Domain Experience Layer | TeachingCase 内容可被 promote 进 vertical 的 DomainExperiencePackage（人工 review 后） |
| 协作 | DLaaS 控制面（未来 Gap 6） | service 层 teaching / asset / persona_research endpoint 都通过 ingestion pipeline |

## 回滚

`IngestionPipeline` 没有 wiring level（它是 adapter，无运行时副作用）。

回滚路径：

- 单 source 故障（如 web parsing 库出 bug）：把 `WebContentSource` 标记 `disabled=True`，service 层 endpoint 拒绝 web ingest
- apprenticeship override 出问题：`VitalsModule.allow_apprentice_override = False`（配置项，默认 True），所有 ingestion 退化成普通 turn（vitals 正常消耗）
- ingestion 内容污染了 memory：reflection writeback 的 `evidence_lineage` 字段可定位到 `envelope_id`，运维侧可在 R6 slow loop 调 `retire_case_by_lineage(envelope_id)` 清理

## Character Live-Through 边界（2026-07-13）

`lifeform-domain-character` 的逐章烘焙可以使用小说原文作为离线抽取输入，但 runtime 只消费 reviewed `ReviewedChapterExperience`。这不是 ingestion 的替代实现，而是不同语义：

| 路径 | 语义 | 允许内容 |
|---|---|---|
| `IngestionEnvelope(source_kind=BOOK)` | lifeform 读一份外部材料 | 原文 / 解析文本，按 provenance 可追溯 |
| `ReviewedChapterExperience` → replay | fictional protagonist 亲历或当时得知 | 人工审核转述、主观可知事实、typed outcome / semantic proposal |

因此全书第三人称原文不得直接写入 fictional protagonist 的 canonical first-half-of-life memory checkpoint。若保留 legacy raw ingestion，必须显式标为 rollback / SHADOW，并在模板 provenance 中暴露。

## 变更日志

- 2026-07-16（I1，slice 2b + TeachingCase 闭合）：
  - **web adapter land**：`lifeform_ingestion/sources/web.py` 提供 `envelope_from_url`（requests + readability-lxml，`[web]` extra 懒加载缺依赖 fail loudly）与 `envelope_from_html`（离线解析路径）。遵守本 spec 约束：显式 `urllib.robotparser` robots.txt 门（404/410 = 无策略放行，disallow 或 robots 不可达 → raise，不假定许可）、timeout 默认 10s + 1 次 retry、content-type 白名单（text/html / xhtml / text/plain）、响应字节上限、scheme 仅 http/https 且拒绝 userinfo（SSRF 第一遍纪律）、无 Playwright。`extractor="stdlib"` 是**显式**调用方兜底（整页文本），绝不静默替代 readability。测试 `tests/ingestion/test_sources_web.py`（fake requests，无网络）。
  - **TeachingCase service 闭合**：`lifeform_service/teaching_case.py` 落地 §Service-side TeachingCase 接入——`TeachingCase` typed 契约 + `teaching_case_envelope`（CORPUS / `teaching-{case_id}` / per-field locator，与 spec 样例一致）+ `TeachingCaseService`（in-memory audit store，只调 `IngestionPipeline.process_envelope`，场景结束触发 R6 slow loop）+ 路由 `POST /v1/sessions/{sid}/teaching-cases`（强制 `ingestion-` session 前缀隔离）/ `GET /v1/teaching-cases` / `POST /v1/teaching-cases/{id}/retire`（R15 回滚标记，保留 envelope lineage）。`lifeform-service` 显式声明 `lifeform-ingestion` 依赖。测试 `tests/service/test_teaching_case.py`。
- 2026-07-13：明确 fictional character 主观 live-through 与 raw BOOK ingestion 的边界；BOOK ingestion 不能代表主角亲历，逐章烘焙必须先经过 reviewed chapter artifact。
- 2026-04-29：初始版本，对应 `docs/implementation/13_emogpt_prd_alignment_upgrade.md` Gap 2 + Gap 3 设计冻结。
