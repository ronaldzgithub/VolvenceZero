# Companion Standard 公开发布 Packet

> Status: A2 in-repo 准备物已完成；外部动作待执行
> Owner: OSS 发布线（与 Companion Bench 同一 governance 姿态）
> Spec: `docs/specs/oss-relationship-representation-standard.md` / `docs/specs/oss-relationship-encoder.md`
> 实施记录: 2026-07-18（Phase A1 + A2 in-repo + B-M1）

## 1. 命名终版

| 事项 | 终版 |
|---|---|
| 标准包名（PyPI / import） | `companion-standard` / `companion_standard` |
| 数据管线包名 | `companion-trajgen` / `companion_trajgen` |
| 对外名称 | **Relationship Representation Standard** |
| 家族归属 | Companion Bench 同族（companionbench.com / companionbench GitHub org） |
| mirror repo 建议名 | `companionbench/standard` |
| 版权行 | `Copyright 2026 Companion Standard Contributors` |

命名决策：与 companion-bench 同族命名（用户已确认），复用 bench 已建立的域名 / org / 公信力，announcement 可以合并发布。

## 2. 已完成的 in-repo 准备物

| 准备物 | 位置 | 状态 |
|---|---|---|
| 标准包（9 类 snapshot + ToM + prediction signal + embedding seam + Snapshot + trajectory schema + conformance kit） | `packages/companion-standard/` | 完成，Apache-2.0 + header |
| 数据管线（FSM 标注器 + arc_runner 导出 + 双模式 CLI） | `packages/companion-trajgen/` | 完成，Apache-2.0 + header |
| 公开 RFC | `docs/external/relationship-representation-rfc-v0.md` | 完成 |
| JSON Schema 导出（draft 2020-12） | `docs/external/relationship-representation-trajectory.schema.json` | 完成，drift 守门在 `test_companion_standard_conformance.py` |
| mirror 单向同步脚本 | `scripts/companion_standard/publish_public_standard.sh` | 完成（--dry-run 默认） |
| 守门测试 | `tests/contracts/test_companion_standard_*` / `test_companion_trajgen_boundaries.py` | 完成，CI 强制 |
| license header 守门 | `tests/contracts/test_apache_license_header_present.py` | 已扩展覆盖两个新包 |

## 3. 清洗 checklist 执行记录（2026-07-18）

按 spec §清洗清单逐项执行，全部通过：

- [x] 客户名 grep（`谌` / `客户` / `customer`）：两个包 + RFC + schema + 测试文件 **零命中**
- [x] 无模型权重 / 二进制：包内除 `.py` / `.toml` / `.md` / `py.typed` 外仅 `__pycache__`（publish 脚本 denylist 排除）
- [x] 无 `docs/business` 引用：**零命中**
- [x] 无 `volvence_zero.*` / `lifeform_*` / `dlaas_platform_*` import：AST 守门测试 + publish 脚本双重把关
- [x] held-out 场景结构性排除：trajgen 禁止 import `heldout_loader` + `include_held_out=False` AST 强制
- [x] 纯 stdlib：`test_companion_standard_is_pure_stdlib` 守门

publish 脚本内置两道 staged-tree 检查（internal-reference / business-material），push 前自动复查。

## 4. 渠道与 announcement

与 Companion Bench 合并发布（同族叙事）：

1. **主 announcement**（bench 渠道复用）："Companion Bench 测量关系行为的结果面；Relationship Representation Standard 命名产生这些行为的状态面。bench 场景 → 标准轨迹由 companion-trajgen 打通。"
2. 渠道顺序：companionbench.com 站点新增 standard 页 → GitHub mirror repo → PyPI 双包 → HN / X / LessWrong（复用 bench RFC 的发布模板）。
3. RFC 反馈入口：mirror repo issues。

## 5. FAQ（"核心内核不开源"口径衔接）

**Q: 你们开源的是什么，不开源的是什么？**
开源的是**表示标准**（关系状态长什么样）+ **合成数据管线** + **评测（Companion Bench）**。不开源的是我们自己的运行时内核：owner 实现、metacontroller（z_t / β_t）、学习循环、prompt 资产。标准的设计原则就是"定义 what，不约束 how"——我们的内核只是标准的一个（专有）实现。

**Q: 为什么相信这个 schema？**
它不是从白板上画出来的，是我们生产系统运行时契约的公开子集，每个类型都有真实运行时里的对应 owner。RFC §7 anti-claims 明确了它是提案不是定论。

**Q: 有没有模型权重？**
标准本身没有权重（见 RFC anti-claim #1）。开源关系编码器是独立后续（`docs/specs/oss-relationship-encoder.md`，release gate G1-G4 未过前不发布）。

## 6. 留给手动的外部动作（不在本仓库）

- [ ] PyPI 注册 `companion-standard` / `companion-trajgen`（发布前先占名）
- [ ] GitHub 创建 `companionbench/standard` mirror repo（read-only 姿态，README 注明 SSOT 在私有 monorepo）
- [ ] companionbench.com 增加 standard 页面
- [ ] 合并 announcement 排期（与 bench 下一次公开动作同窗口）
- [ ] `companion-trajgen` 依赖的 `companion-bench` wheel 需先在 PyPI 可安装（bench 发布线的前置）

## 7. 回滚

- mirror push 前一切无外部效力；push 后回滚 = mirror repo force-push 上一版（脚本天然支持，SSOT 在 monorepo）。
- PyPI 撤包走 yank（不删版本号）。
- 两个包在 monorepo 内的 license 翻转可单 commit revert（不影响任何内部 re-export 消费者）。
