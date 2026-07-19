"""Content-addressed snapshot store + run log.

设计底线（DESIGN.md §4）：
- 数据本体存 .labs/cas/<sha[:2]>/<sha>.bin；重复内容去重。
- 索引存 .labs/index.sqlite 两张表：snapshots + runs。
- 写入幂等；消费者只通过 sha 引用；rollback = 从 CAS + RunLog 重建。
"""

from .cas import CASStore, canonical_dumps, sha256_bytes
from .runlog import RunLog, RunRecord
from .paths import LabsPaths, default_paths

__all__ = [
    "CASStore",
    "RunLog",
    "RunRecord",
    "LabsPaths",
    "default_paths",
    "canonical_dumps",
    "sha256_bytes",
]
