# Relationship Lab Gate 0 — Qwen2.5 1.5B development calibration

这是 2026-08-19 的 **development calibration evidence**，不是 formal
hidden-test evidence，也不是已经冻结的 preregistration。

- substrate：`Qwen/Qwen2.5-1.5B-Instruct`（本机缓存、同一冻结 weights）
- arm：`stateless`，模型只收到逐字节相同的 current input 与 closed action surface
- split：train + validation，共 4 个 mirrored pairs × 3 个 matched seeds × 2 users
- structured decisions：24/24 有效
- action accuracy：4/24（0.166667）
- context tokens：4554
- decision ledger sha256：`fff11045ea2302bbb2f218590449f1eaec9075110fe6fa3340e9f4bd574ca45b`
- Gate 0：六项检查全部 PASS，`machinery_ready=true / gate0_passed=true`

`baseline_attestation.json`、`decisions.jsonl` 与 `gate0/report.json` 是
content-addressed evidence。仓库场景包中的 `heldout` 仅是开发期结构分割；正式
实验必须在冻结 prereg 后另行生成并封存 secret heldout。
