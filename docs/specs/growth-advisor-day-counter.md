# Growth-Advisor Day Counter Spec

> Status: scaffold v0.1 (SHADOW)
> Owner: growth-advisor-pilot-packet G-B (debt #65)

## 1. 问题

`GrowthAdvisorStrategyPrior.applicability_scope=("growth_advisor:day1",)` 等 7 day playbook 路由依赖"今天是用户第几天"——但 day_counter 来源 owner、计算公式、tz 处理在 [`fixture_uptake.py`](../../packages/lifeform-domain-growth-advisor/src/lifeform_domain_growth_advisor/fixture_uptake.py) 第 31-35 行注释里明确"字符串透传未实现"（debt #65 描述准确）。

本 spec 锁定 day_counter SSOT。

## 2. day_counter SSOT 决策

### 2.1 候选

| 候选 | 来源 | 优势 | 劣势 |
|---|---|---|---|
| (a) `scoped_memory.onboarding_at` | scoped memory typed field | 与 closed-alpha session lifecycle 自然对齐；持久 | 需要 onboarding hook 落 memory；首次用户 None handling |
| (b) `UserIdentity.created_at` | identity provider | 简单 | 不区分 onboarding vs 重激活 |
| (c) HTTP header `X-Onboarding-Date` | 客户端传 | 灵活 | 客户端需实现；不可信 |

### 2.2 推荐：(a) scoped_memory.onboarding_at

理由：
- 与 closed-alpha 已有 scoped memory 体系自然集成
- 持久（重启服务后 day_counter 仍正确）
- 客户端无新职责

### 2.3 计算公式

```python
def compute_growth_advisor_day(
    *,
    onboarding_at_ms: int,
    now_ms: int,
    tz_offset_minutes: int = 480,  # 中国时区默认 +8h
) -> int:
    """Return 1..7+ day number (1-indexed)."""
    onboarding_dt = ms_to_local_date(onboarding_at_ms, tz_offset_minutes)
    now_dt = ms_to_local_date(now_ms, tz_offset_minutes)
    delta_days = (now_dt - onboarding_dt).days + 1  # day 1 是 onboarding 当天
    return max(1, delta_days)
```

`day8+` 全部映射到 `growth_advisor:day7+`（playbook 第 7 天后进入"持续运营"模式，没有专属 rule）。

## 3. tz 处理

- 默认中国时区 (+8h)；P2 客户多在国内
- 海外客户走 `--tz-offset-minutes` 配置
- DST 不处理（tz_offset_minutes 是固定常量）

## 4. profile schema 增强

`GrowthAdvisorStrategyPrior.applicability_scope` 加 enum 校验：

```python
def _validate_applicability_scope(scope: tuple[str, ...]) -> None:
    valid_day_tags = {f"growth_advisor:day{n}" for n in range(1, 8)} | {"growth_advisor:day7+"}
    valid_funnel_tags = {"funnel:height", "funnel:immunity", "funnel:nutrition", "funnel:vision"}
    valid = valid_day_tags | valid_funnel_tags
    invalid = set(scope) - valid - {""}  # 空字符串退出 (legacy)
    if invalid:
        raise ValueError(
            f"GrowthAdvisorStrategyPrior.applicability_scope unknown tags: {sorted(invalid)}"
        )
```

加在 `GrowthAdvisorStrategyPrior.__post_init__` (debt #65 子任务 b)。

## 5. 客户 ops dashboard 端点

`GET /v1/tenants/{tid}/users/{uid}/current-day` → `{"day": 3, "tag": "growth_advisor:day3"}`

让客户运营总监能验证"我的某用户今天处于第 3 天"，证明 day-counter 真生效。

## 6. 退出标准

| 阶段 | 标准 |
|---|---|
| **SHADOW**（W4） | 本 spec v0.1 + `tests/contracts/test_growth_advisor_day_routing.py` 骨架（仅 enum 校验）+ `applicability_scope` enum 校验加在 `GrowthAdvisorStrategyPrior.__post_init__` |
| **ACTIVE**（W6） | `compute_growth_advisor_day` 函数 + scoped_memory hook + ops dashboard endpoint + day-cohort 7 turn 序列测试 |

## 7. 风险

| 风险 | 应对 |
|---|---|
| onboarding_at 缺失（端用户从其他系统迁入） | fallback：取 first turn timestamp 作为 onboarding_at |
| 跨日期跑 day-counter 边界（晚上 23:59 vs 凌晨 00:01） | tz_offset 固定 +8h，按当地日期切；不在边界做特殊处理 |
| client request 时间戳不可信 | day_counter 用服务器 now_ms，不用 client 传的 |

## 变更日志

- 2026-05-13: v0.1 SHADOW spec。
