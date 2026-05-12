# Figure-Vertical Persona Verification Spec

> Status: draft
> Last updated: 2026-05-12
> 对应需求: R2, R8, R12, R15
> 对应代码: `packages/lifeform-domain-figure/src/lifeform_domain_figure/verification/persona/`

## 要解决的问题

[`figure-vertical.md`](./figure-vertical.md) 把 corpus → bundle → LoRA → synthesizer 全链跑通了，但**怎么证明这条链真让模型既有 Einstein 的口气，又有 Einstein 的认知？** Wave G 的 e2e 只测「LoRA 改了 forward」+「enforcer 接进来了」，并不打分；reviewer 看 transcript 凭感觉拍版，不能进 CI。

本 spec 描述自动化验证管线：从 [Wave K curated bundle](./figure-corpus-cleaning.md) 出发，自动产题、跑 ablation、deterministic 打分、emit 4-gate verdict，让 figure 上线前通过同一条管线被打分；评分逻辑全在仓库内可重现，不依赖外部 LLM judge。

## 不在范围

- 不评 ETA 控制器（regime selection / β_t）—— 它对 voice/cognition 的影响二阶。
- 不重训基底（R2 frozen base），只测 LoRA 适配层。
- 不接外部 LLM judge —— 全程用 bundle 自带的 `retrieval_index` + `style_prior` 做评分。

## 三 condition ablation

| Condition | 配置 | 用途 |
|---|---|---|
| `RAW` | synthesizer + `figure_bundle=None` | 基底 LLM 单独，无任何 figure-vertical 约束 |
| `BUNDLE` | synthesizer + bundle，临时把 LoRA pool 记录摘掉 | L1 + L3 + L4 enforcer 全开，但 LoRA 不激活 |
| `BUNDLE_LORA` | synthesizer + bundle + pool 已注册 → 自动 activate | 生产配置 |

`BUNDLE` 的实现做法：进上下文前 `pool.deregister(figure_id)` + 缓存 record，出上下文 `pool.register(...)` 用同样字段恢复（`record_id` 由 `(figure_id, source_bundle_id)` 决定，所以恢复字节级一致）。同一 `runtime` 在三个 condition 间复用，避免重复加载 Qwen。

## 测试题集生成

### in-corpus 立场题（自动）

`generate_in_corpus_questions(bundle, max_questions=20)`：

1. 按 `chunk_id` 排序遍历 `bundle.retrieval_index.chunk_records`（确定性）。
2. 对每个 chunk，从首段提取 ≥3 个长度 ≥ 4 的 content word 拼成 `topic_phrase`（跳过 stop-words），打不到 3 个就跳过该 chunk。
3. 用固定模板 `"Speaking from your own primary writings, what is your perspective on the relationship between {topic}?"` 包出问题。
4. ground truth = 整个 chunk 的 `locator` + 前 240 字符 excerpt。

**为什么不用 LLM 生成问题**：(a) reproducibility — 模板生成 byte-identical；LLM 生成漂；(b) CI hermetic — 不需要 Qwen 下载就能验证 harness；(c) question 质量不是评分上限 —— 评分用 `retrieval_index.assertion_is_supported(response_text)`，问题只要 on-topic 就够。

### out-of-scope 拒答探针（人工）

5 道 reviewer-curated 显著 off-corpus 问题（tiramisu / sourdough / Python tutorial / car maintenance / pop song）。常量：`OUT_OF_SCOPE_REFUSAL_QUESTIONS`。增删要保持「明显 off-corpus」的语义，否则 refusal 评分失真。

## 自动评分公式

### Voice fidelity

```
top_words_overlap = |R_top80 ∩ S_top80| / 80
sentence_length_match = max(0, 1 - min(1, |median(response_sentence_lens) - p50| / 100))
voice_score = 0.6 * top_words_overlap + 0.4 * sentence_length_match
```

`R_top80` = response 文本里频次最高的 80 个 ≥3 字母词；`S_top80` = `bundle.style_prior.top_words[:80]`；`p50` = `bundle.style_prior.sentence_length_percentiles["p50"]`。

注意：分母固定 80（不是 `min(|R|, 80)`），所以**短回复**自动拿不到高分 —— 8 个 distinct word 上限是 8/80 = 0.10。

### Cognition accuracy

```
supports = bundle.retrieval_index.assertion_is_supported(response_text)
hits_ground_truth = any(s.locator == gt_locator for s in supports)
cognition_score = max(s.cosine_score for s in supports if s.locator == gt_locator) if hits else 0.0
```

只看 ground-truth chunk 的 cosine —— 模型答到了"另一篇 Einstein 文章"也算 0 分，因为问题问的是这一篇。

### Refusal precision

```
refused = "l4_scope_refusal" in rationale_tags or response_text starts with refusal-preamble
refusal_correct = (refused == expected_refusal)
```

`expected_refusal = True` for OUT_OF_SCOPE_REFUSAL questions, `False` otherwise. 同时覆盖「该拒就拒」和「该答就答」两个方向。

## 4-gate verdict

| Gate | 公式 | 默认阈值 | 解读 |
|---|---|---|---|
| `gate_cognition_improves` | `bundle.cognition - raw.cognition ≥ Δ_cognition` | 0.05 | bundle 的 retrieval enforcement 让模型抓 GT chunk 的能力比 raw 强 |
| `gate_voice_improves_with_lora` | `bundle_lora.voice - bundle.voice ≥ Δ_voice` | 0.02 | LoRA **真**改了 forward，voice 评分有量化提升（载荷性 gate） |
| `gate_refusal_works` | `bundle.out_of_scope_refusal_rate ≥ refusal_min` | 0.80 | 5 道 off-corpus 题至少 4 道触发 L4 拒答 |
| `gate_evidence_emerges` | `max(bundle, bundle_lora).l3_evidence_count ≥ evidence_min` | 1 | L3 grounded decoder 至少在某些 in-corpus 题上回了 evidence 指针 |

四个 gate 全过 → CLI 退出码 0；任一不过 → 退出码 2 + `verdict.json.gates[i].passed=False`。

### 阈值的来源 / 局限

阈值是 **Wave P 经验值**，刻意取得保守：

- Wave K curated corpus 当前只有 ~2 篇 substantive paper，cosine score 上限不高 → `cognition_delta=0.05` 而不是 0.20。
- LoRA 在小 corpus 上 forward 改动幅度有限 → `voice_delta=0.02` 而不是 0.10。
- `refusal_min=0.80` 留了 1/5 容错，避免 reviewer 写的拒答模板偶尔不命中 LLM 的拒答检测。

阈值校准是 follow-up debt（见 [`docs/known-debts.md`](../known-debts.md)）：等收集到一批历史 verdict run 后用 ROC 曲线选阈值，而不是当前的 hand-tuned 默认值。

## CLI / 编排

`scripts/figure_verify_einstein_persona.sh`：

```bash
# 0. assert prerequisites (curated bundle id + cleaning store + metadata)
# 1. (optional) bake real PEFT persona LoRA on Wave K curated bundle
#    via figure_bake_einstein_persona_lora.sh (skip with SKIP_BAKE=1)
# 2. drive python -m lifeform_domain_figure.verification.persona.cli
#    -> questions.jsonl / results/<cond>.jsonl / scores.json /
#       verdict.json / transcript.md
# 3. cat verdict.json (exit code 0/2 reflects gate outcome)
```

`python -m lifeform_domain_figure.verification.persona.cli --help` 列全部 flag。关键 flag：

- `--runtime {transformers,synthetic}` — synthetic 用于 harness smoke；transformers 跑真 HF 模型
- `--qwen-model-id` — 默认 `sshleifer/tiny-gpt2`（CPU smoke 用）；推荐 `Qwen/Qwen2.5-1.5B-Instruct` for real run
- `--cognition-delta` / `--voice-delta` / `--refusal-min` / `--evidence-min` — 每个 gate 的阈值覆盖

输出目录结构：

```
artifacts/figure_verify/<run_id>/
    questions.jsonl              # 全部测试题
    results/raw.jsonl            # raw condition 响应
    results/bundle.jsonl
    results/bundle_lora.jsonl
    scores.json                  # per-question score + per-condition aggregate
    verdict.json                 # 4-gate 终评
    transcript.md                # reviewer-friendly 三栏并排
```

## 怎么读 transcript.md

- 头部：bundle id / pool record id / overall pass-fail / 4 道 gate 各自 observed vs threshold
- 中段：per-condition aggregate（voice / cognition / refusal_rate / l3_evidence_count）
- 主体：每道题三栏并排（raw / bundle / bundle_lora），含 wall_ms + rationale tag 列表

reviewer 5 分钟内能看出：(a) bundle 是否真在引证；(b) bundle_lora 文风是否更像；(c) out-of-scope 拒答是否短路。

## 关键不变量守门

1. **R2 frozen base** —— `with_condition` 出 context 后 runtime base `state_dict_hash` 字节稳定（Wave D contract test 复用）。
2. **R8 SSOT** —— `verification.persona` 子模块只走 `lifeform_domain_figure` 公开 surface（bundle / pool / synthesizer / retrieval_index）+ `volvence_zero.substrate` 公开 Protocol（runtime / pool / activate_lora），不直接 import 其他 lifeform 子模块。
3. **R15 重现性** —— `generate_in_corpus_questions` 模板固定 + 排序确定性 + `--questions-cache` 缓存 → 同 cleaning store + 同 LoRA + 同 question cache → 同 `verdict.json` byte-identical。
4. **不滥用 hasattr** —— scoring 直接读 `bundle.style_prior.top_words` / `bundle.retrieval_index.assertion_is_supported`，类型已固定，不用 `getattr` 兜底。
5. **不引关键词匹配驱动决策** —— refusal score 走 `rationale_tags` + reviewer-written refusal preamble；自动产题不是关键词匹配，而是结构化 chunk 遍历。

## 测试覆盖

| 测试 | 内容 | CI |
|---|---|---|
| `test_persona_verification_smoke.py` | 6 case：question 生成确定性 / 三 condition context manager 行为 / ablation grid 完整 / scoring schema / verdict 4-gate / CLI 全输出树 | 默认跑 |
| `test_persona_verification_real_qwen.py` | `@pytest.mark.hf`：用 tiny-gpt2 跑真 Transformers runtime + 完整 ablation + verdict shape 检查 | `pytest -m hf` opt-in |

## Follow-up debts

见 [`docs/known-debts.md`](../known-debts.md)：

- voice score 现在用 hashing-embedding cosine 在 `style_prior.top_words` 上 overlap；未来希望升级到 substrate residual cosine
- 4 个 gate 阈值是经验初值，等积累足够多 verdict run 后用 ROC 校准
