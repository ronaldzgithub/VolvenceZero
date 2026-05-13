# Companion Bench OpenRouter Setup

> Status: smoke v0.1
> Last updated: 2026-05-13
> 适用：第一次接手 companion-bench reference run / smoke 跑通 / 排查 401·rate-limit·timeout

## 1. 为什么用 OpenRouter

OpenRouter 一家 key 代理多家模型（GPT-5 / Claude / Gemini / DeepSeek / Qwen / Llama / Mistral …），全部走 **OpenAI compat** `/v1/chat/completions`，与 [`OpenAIChatClient`](../../packages/companion-bench/src/companion_bench/sut_client.py) / [`OpenAIUtteranceClient`](../../packages/companion-bench/src/companion_bench/user_simulator.py) **零改造适配**——不需要为每家厂商写新 HTTP client。

## 2. Phase 0 — 一次性账号准备

1. **注册 OpenRouter**：<https://openrouter.ai/> → 注册 → 进 [Settings → Keys](https://openrouter.ai/settings/keys)
2. **充值** ~$10（OpenRouter 余额制；smoke 跑只用 ~$1-3）
3. **创建 API key**：copy `sk-or-v1-xxxxxxxx`（只显示一次）

## 3. .local/llm.env 配置

在仓库根的 `.local/llm.env` 加（这个文件 gitignored）：

```bash
# OpenRouter — 给 SUT / user_simulator / judge 三处统一用
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxx

# OpenRouter 推荐 attribution headers (rate limit 友好 + 可在 openrouter.ai/rankings 显示)
OPENROUTER_HTTP_REFERER=https://volvence.zero
OPENROUTER_X_TITLE=VolvenceZero CompanionBench

# VZ 本地 SUT 走 lifeform-openai-compat，server 不验 Bearer，任何非空值即可
LIFEFORM_LOCAL_API_KEY=local-dev-no-auth
```

## 4. 跑 smoke

```bash
bash scripts/companion_bench/run_companion_bench_smoke.sh
```

预期：~10-20 min wallclock，~$1-3 token cost，输出：

```
artifacts/companion_bench_smoke/
├── openrouter-gpt-5-mini-smoke/
│   ├── summary.json
│   └── arcs/*.bundle.json    (4 个：4 scenario × 1 seed)
├── lifeform-companion-smoke/
│   ├── summary.json
│   └── arcs/*.bundle.json
└── aggregate_results.json
```

外加 `site/data/{aggregate_results,scenarios,pairwise}.json` + `site/data/submissions/*.json`。

本地预览：浏览器打开 `site/index.html`（或 `site/leaderboard/index.html`）。

## 5. 选项 / 常用 env

```bash
# 跑别的 family（默认 F1，4 scenarios）
SMOKE_FAMILY=F2 bash scripts/companion_bench/run_companion_bench_smoke.sh

# 不起 VZ SUT（只跑 OpenRouter SUT，比如不想本地装 lifeform-serve）
SKIP_VZ_SUT=1 bash scripts/companion_bench/run_companion_bench_smoke.sh
# 此时 reference_systems.smoke.yaml 里 lifeform-companion 会因 endpoint 不可达失败，
# score_reference_systems.py 会 log 然后跳过 aggregate row，OpenRouter 部分仍出
```

## 6. 故障排查

### 401 Unauthorized

```
urllib.error.HTTPError: HTTP Error 401: Unauthorized
```

- 检查 `.local/llm.env` 的 `OPENROUTER_API_KEY` 没拼错 / 没多余空格
- OpenRouter 余额是否充够（账户首页看 credit balance）
- `Authorization: Bearer {key}` 由 [`OpenAIChatClient`](../../packages/companion-bench/src/companion_bench/sut_client.py) 自动加，不需要你手动

### 429 Rate Limit

```
urllib.error.HTTPError: HTTP Error 429: Too Many Requests
```

- 加 `OPENROUTER_HTTP_REFERER` + `OPENROUTER_X_TITLE`（OpenRouter 对带 attribution 的请求 rate limit 更宽松）
- 减并发 / 加 `--paraphrase-seeds 0`（本 smoke 已是）
- 充更多 credit（OpenRouter rate limit 与 balance 挂钩）

### TimeoutError / `urllib.error.URLError: <urlopen error [Errno 110]>`

- 网络问题；OpenRouter 在国内可能需要代理 / 加速器
- `OpenAIChatClient` 默认 `request_timeout_s=120.0`；可改 [`packages/companion-bench/src/companion_bench/sut_client.py`](../../packages/companion-bench/src/companion_bench/sut_client.py) line 74

### VZ SUT 启动失败

```
ERROR: VZ SUT did not become healthy within 30s
```

- 检查 `lifeform-serve` 是否安装：`pip install -e packages/lifeform-service`
- 检查 8000 端口是否被占：`lsof -i :8000` (Linux/Mac) / `netstat -ano | findstr :8000` (Windows)
- 改端口：`VZ_SUT_PORT=8001 bash scripts/companion_bench/start_vz_sut.sh start`，然后 reference_systems.smoke.yaml 里也改对应 base_url
- 看 log：`tail -f artifacts/companion_bench_smoke/vz_sut.log`

### 某个 SUT 失败但其他 SUT 仍跑

`score_reference_systems.py` 失败是 **per-SUT 容错**：单 SUT crash 不阻塞其他 SUT。最终 `aggregate_results.json` 只含成功的 SUT。

## 7. 扩到更大跑分

smoke 跑通后选项：

| 选项 | 命令 / 改动 | 估算 |
|---|---|---|
| 跑全 24 公开 scenario × 2 SUT | 删 `--family F1` | ~$10-30 = ¥70-200 |
| 加更多 SUT（gpt-5 / claude-opus / gemini / deepseek） | 编辑 [`reference_systems.smoke.yaml`](../../scripts/companion_bench/reference_systems.smoke.yaml) 加 systems | × 系数 |
| 扩 paraphrase seeds 到 3 | `--paraphrase-seeds 0,1,2` | × 3 |
| 跑 96 held-out（release tier） | clone submodule + `--include-heldout --require-heldout` | $5-15k |
| Wire CI workflow nightly | 见 [`.github/workflows/companion-bench-paper-suite-small.yml`](../../.github/workflows/companion-bench-paper-suite-small.yml) | nightly $200-400 |

## 8. SSOT 约束

- `.local/llm.env` 永不进 git（`.gitignore`）
- `artifacts/companion_bench_smoke/` 永不进 git（`.gitignore` 已含 `artifacts/`）
- `site/data/` 是公开的；smoke 跑的数据**不要 commit**（标 demo / 不是 reference run）
- 真正 reference run 走 [`scripts/companion_bench/run_companion_bench_paper_suite_small.sh`](../../scripts/companion_bench/run_companion_bench_paper_suite_small.sh)（`paper_suite` 命名表示 v1.0 公开榜单数据）

## 9. Judge 合格度档级（**重要**）

`docs/specs/companion-bench.md` §5 require：**arc judge 必须来自与 per-turn judge 不同的 model family**（cross-family rotation 缓解 family-bias）。spec 把强制层放在 orchestrator（即本仓库的 `score_reference_systems.py` / `run_real_submission.py`），wheel 只接受任意 client。

按合格度从低到高 3 档：

| 档级 | 配置 | 适用场景 | 不能用于 |
|---|---|---|---|
| **档 A — Weak Proxy** | per-turn = `qwen3-max`，arc = `qwen-plus`（同 family，不同 size） | pipeline 验证 / dev iteration / 看 SUT 大致排名 | 公开 leaderboard / arXiv / 客户引证 |
| **档 B — Family Rotation** | per-turn = `openai/gpt-5-mini`，arc = `anthropic/claude-3.7-sonnet`（OpenRouter） | v0.1 公开 launch judge / 内部 reference run | 需要 ρ ≥ 0.75 inter-rater 证据的场合 |
| **档 C — 严肃合格** | 档 B + 跑 [debt #48](../../docs/known-debts.md) cross-family robustness sweep（5+ family judge × ρ ≥ 0.75 + per-axis σ < 8.0）+ [debt #52](../../docs/known-debts.md) calibration sweep + [debt #53](../../docs/known-debts.md) simulator robustness | v1.0 official launch / arxiv preprint / 客户尽调 | — |

Smoke run 默认走 **档 A**（[`reference_systems.smoke_qwen.yaml`](../../scripts/companion_bench/reference_systems.smoke_qwen.yaml)）。Qwen-内 size 维度的 weak robustness 自验由 [`scripts/companion_bench/qwen_judge_robustness_replay.py`](../../scripts/companion_bench/qwen_judge_robustness_replay.py) 提供——结果只能 catch 显著的 size sensitivity，**不**等同于真 cross-family ρ。

档 A → 档 B 切换：`SMOKE_PROVIDER=openrouter` （需用户 .env 已加 `OPENROUTER_API_KEY`）。
档 B → 档 C 切换：跑 [`scripts/companion_bench/judge_robustness_sweep.py`](../../scripts/companion_bench/judge_robustness_sweep.py) (#48 真 sweep) + 公开报告 [`docs/external/companion-bench-judge-robustness-v0.md`](companion-bench-judge-robustness-v0.md) 回填真数据。

## 10. 与 packet / 文档关联

- packet plan：[`docs/moving forward/companion-bench-public-launch-packet.md`](../moving%20forward/companion-bench-public-launch-packet.md) §3
- rollout：[`docs/moving forward/commercialization-evidence-rollout.md`](../moving%20forward/commercialization-evidence-rollout.md) §6 W2
- spec：[`docs/specs/companion-bench.md`](../specs/companion-bench.md)
- RFC：[`docs/external/companion-bench-rfc-v0.md`](companion-bench-rfc-v0.md)

## 变更日志

- 2026-05-13: v0.1 initial setup guide for smoke run。
