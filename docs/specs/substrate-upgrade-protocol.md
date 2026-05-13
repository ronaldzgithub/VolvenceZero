# Substrate Upgrade Protocol

> Status: scaffold v0.1 (SHADOW)
> Last updated: 2026-05-13
> Owner: cross-cutting-foundation-packet F-C (debt #47)

## 1. 范围

任何 substrate 升级（如 `Qwen2.5-1.5B → Qwen3-1.5B` / `Llama-3 → Llama-4`）触发后，本 spec 决定下游三个商业方向（P5 / P1 / P2）的 artifact 兼容性判定 + 升级 / 降级 / 不兼容三档处置。

参考：[`commercialization-assessment.md`](../business/commercialization-assessment.md) §8.1.1（"substrate 升级对 figure bundle 兼容性破坏"列为高 × 高风险）+ cross-cutting-foundation-packet F-C 子任务 1-3。

## 2. SubstrateFingerprint 定义

每个 substrate 由 [`SubstrateFingerprint`](../../packages/vz-substrate/src/volvence_zero/substrate/substrate_fingerprint.py)（debt #47）唯一标识：

```python
SubstrateFingerprint(
    model_id="Qwen/Qwen2.5-1.5B-Instruct",
    version="v2.5",
    weights_sha256="<sha256(state_dict)>",
)
```

`weights_sha256` 是 substrate 的真主键。`model_id` + `version` 提供人类可读的上下文。

`LEGACY_FINGERPRINT`（`tinygpt2 / legacy / legacy`）是迁移 shim：在 #47 字段加入之前 bake 的 bundle 反序列化时默认填入此值，让旧 bundle 仍可加载并 warn。

## 3. 三方向兼容性消费

| 字段 | 消费者 | 行为 |
|---|---|---|
| `FigureArtifactBundle.compatible_substrates: tuple[SubstrateFingerprint, ...]` | figure runtime | 启动时校验当前 substrate 在 set 中；不匹配 fail-loud（L2 LoRA 绑死 substrate weights） |
| `GrowthAdvisorProfile.validated_substrates: tuple[SubstrateFingerprint, ...]` | growth-advisor lifeform builder | 启动时校验；空 = "untested / generic"，runtime warn 不 fail（profile 是 application 层 typed records） |
| `RunRecord.sut_substrate_fingerprint: str \| None` | companion-bench 公开榜单 | 按 substrate 分组展示；None = 历史/匿名跑分 |

## 4. 升级 → 兼容性判定矩阵

| 当前 substrate fingerprint | 新 substrate fingerprint | figure bundle 处置 | growth-advisor profile 处置 | companion-bench 处置 |
|---|---|---|---|---|
| 同 model_id + 同 version + **同 weights** | 完全相同 | 兼容（无变化） | 兼容 | 同一榜单组 |
| 同 model_id + 同 version + **不同 weights**（如 fine-tune） | 微小变化 | **重 bake**（L2 LoRA 不能复用） | warn but continue | 新榜单组（标 fork） |
| 同 model_id + **不同 version** | 中等变化 | **必须重 bake**；旧 bundle 通过 fingerprint mismatch fail-loud | 推荐 re-validate；若 validated_substrates 不为空 warn | 新榜单组 |
| **不同 model_id**（如 Qwen → Llama） | 大变化 | 完全不兼容；旧 bundle 不可降级运行 | 推荐 re-validate；warn | 新榜单组（独立比较） |

## 5. 降级路径（rollback to N-1）

如果新 substrate ACTIVE 后发现回退需求（commercialisation §6 P1 高客单价客户合同 SLA 写"提供回滚证据"）：

1. **优先**：回退到 N-1 substrate weights（保留旧 weights 至少 N+1 个 release cycle）
2. **次优**：如果旧 weights 已删，从 `figure_audit/<figure_id>/` 取最近一次匹配旧 fingerprint 的 bake audit，重 bake 一份兼容当前 substrate 的 bundle（成本: 见 P1 packet §3 单位经济）
3. **不可回退**：触发 [`commercialization-assessment.md`](../business/commercialization-assessment.md) §8.1.1 "substrate 升级破坏" 风险事件，启动 N-2 substrate 的 emergency hotfix 计划

## 6. 升级前 checklist

substrate 团队主动升级前必须完成：

- [ ] 新 weights SHA-256 已记录 + 对外发布 fingerprint
- [ ] 跑一次 [`scripts/rollback_drill_substrate_upgrade.sh`](../../scripts/rollback_drill_substrate_upgrade.sh)（F-D rollback drill）确认 byte-identical 回滚
- [ ] 跑一次 [`scripts/realistic_load_*.py`](../../scripts/realistic_load_companion.py) 三个 vertical baseline，对比 N vs N+1 性能（确保 P99 latency 不退化 > 20%）
- [ ] 通知 figure / growth-advisor 团队 N+1 fingerprint，准备 re-bake / re-validate
- [ ] companion-bench 团队收到 fingerprint，规划新榜单组

## 7. SSOT 约束

| 不变量 | 守门 |
|---|---|
| 每个 substrate runtime 必须实现 `fingerprint() -> SubstrateFingerprint` | `tests/contracts/test_substrate_fingerprint_propagation.py` |
| `FigureArtifactBundle.compatible_substrates` 默认空 (`()`) 且**不**进 `compute_bundle_integrity_hash` 当空时 | 同上：相同 inputs + 空 compatible_substrates → byte-identical bundle hash |
| 当 `compatible_substrates` 非空时，bundle hash **必须** include 它（R15 byte-level 回滚） | 同上：相同 inputs + 不同 compatible_substrates → 不同 bundle hash |
| LEGACY_FINGERPRINT 仅用于迁移 shim，不允许作为 production fingerprint | 由 bake CLI 拒绝 (`raise if fingerprint == LEGACY_FINGERPRINT`) |

## 8. 退出标准

| 阶段 | 标准 |
|---|---|
| **SHADOW**（W3-W4） | `SubstrateFingerprint` + `LEGACY_FINGERPRINT` + `fingerprint_set_sha256` 落 vz-substrate；`FigureArtifactBundle.compatible_substrates` 字段加（默认 `()` 兼容旧 bundle）；`compute_bundle_integrity_hash` 接受 `compatible_substrates` 参数；`GrowthAdvisorProfile.validated_substrates` 字段加；`RunRecord.sut_substrate_fingerprint` 字段加；契约测试 `test_substrate_fingerprint_propagation.py` 通过 |
| **ACTIVE**（W6） | 三个 backend (synthetic / Transformers / hf) 实现 `fingerprint()`；CLI bake 自动调 `runtime.fingerprint()` 写入 bundle；rollback drill `scripts/rollback_drill_substrate_upgrade.sh` 真跑通 |

## 9. 风险

| 风险 | 应对 |
|---|---|
| 新 substrate weights 算 SHA-256 慢（> 1 GB / 几秒） | runtime fingerprint 缓存到本地 `~/.cache/vz/substrate_fingerprints.json`，weights mtime + size 命中即跳 SHA-256 |
| 旧 bundle pickle 反序列化时 missing `compatible_substrates` 字段 | dataclass field default `()` 自动填空；migration shim 在 [`bundle_io`](../../packages/lifeform-domain-figure/src/lifeform_domain_figure/bundle_io.py) `load_bundle` 后填 `LEGACY_FINGERPRINT` 并 warn |
| 客户上传"我们用了你们的 N-2 substrate"无法识别 | sales 工艺：客户合同必须 record fingerprint；不在合同里的 fingerprint 视为 unsupported |

## 变更日志

- 2026-05-13: v0.1 SHADOW scaffold land。
