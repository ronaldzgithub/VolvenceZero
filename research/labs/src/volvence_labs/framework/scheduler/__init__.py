"""Scheduler: sequential & parallel runner。

每个 unit = (probe, cell, seed)。流程：

    1. 从 CAS 读 inputs_sha（若首次，调用 probe.default_inputs 写入）
    2. 调用 probe.run_cell(ctx, knobs)
    3. 把 output / readouts / manifest 写入 CAS
    4. 写 RunRecord 到 RunLog
    5. 往 experiments/<run_id>/ 镜像一份人类可读 manifest.json（方便 grep）

无论 sequential 还是 parallel，单 unit 的处理流程完全一致 —— parallel 只是
把 unit list 分到子进程。
"""

from .runner import (
    ExperimentReport,
    ParallelRunner,
    SequentialRunner,
    UnitReport,
    run_experiment,
)

__all__ = [
    "ExperimentReport",
    "ParallelRunner",
    "SequentialRunner",
    "UnitReport",
    "run_experiment",
]
