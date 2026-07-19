"""Run log: 每次 unit run 一条记录，内容全是 sha 引用。

R15 的可回滚就靠这个：删掉 experiments/<run_id>/ 后，也能从 CAS + RunLog
重建。
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict, dataclass
from typing import Iterable, Optional

from .cas import CASStore, _connect, _ensure_schema
from .paths import LabsPaths


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    probe_id: str
    wiring: str            # WiringLevel value
    ablation_cell: str     # AblationCell value
    seed: int
    knobs_sha: str
    input_sha: str
    output_sha: str
    readouts_sha: str
    manifest_sha: str
    created_at: float

    @classmethod
    def from_row(cls, row: tuple) -> "RunRecord":
        return cls(
            run_id=row[0],
            probe_id=row[1],
            wiring=row[2],
            ablation_cell=row[3],
            seed=row[4],
            knobs_sha=row[5],
            input_sha=row[6],
            output_sha=row[7],
            readouts_sha=row[8],
            manifest_sha=row[9],
            created_at=row[10],
        )


class RunLog:
    def __init__(self, paths: LabsPaths, store: CASStore):
        self.paths = paths
        self.store = store
        # reuse a single connection; ensure schema is present.
        self._conn = _connect(self.paths.index_db)
        _ensure_schema(self._conn)

    def record(self, record: RunRecord) -> None:
        for attempt in range(5):
            try:
                self._conn.execute(
                    "INSERT OR REPLACE INTO runs("
                    "run_id, probe_id, wiring, ablation_cell, seed, knobs_sha, "
                    "input_sha, output_sha, readouts_sha, manifest_sha, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);",
                    (
                        record.run_id,
                        record.probe_id,
                        record.wiring,
                        record.ablation_cell,
                        record.seed,
                        record.knobs_sha,
                        record.input_sha,
                        record.output_sha,
                        record.readouts_sha,
                        record.manifest_sha,
                        record.created_at,
                    ),
                )
                return
            except sqlite3.OperationalError:
                if attempt < 4:
                    time.sleep(0.1 * (attempt + 1))
                else:
                    raise

    def get(self, run_id: str) -> RunRecord:
        cur = self._conn.execute(
            "SELECT run_id, probe_id, wiring, ablation_cell, seed, knobs_sha, "
            "input_sha, output_sha, readouts_sha, manifest_sha, created_at "
            "FROM runs WHERE run_id = ?;",
            (run_id,),
        )
        row = cur.fetchone()
        if row is None:
            raise KeyError(f"run_id not found: {run_id}")
        return RunRecord.from_row(row)

    def list(self, *, probe_id: Optional[str] = None, limit: int = 200) -> list[RunRecord]:
        cur = self._conn.execute(
            "SELECT run_id, probe_id, wiring, ablation_cell, seed, knobs_sha, "
            "input_sha, output_sha, readouts_sha, manifest_sha, created_at "
            "FROM runs WHERE (? IS NULL OR probe_id = ?) "
            "ORDER BY created_at DESC LIMIT ?;",
            (probe_id, probe_id, limit),
        )
        return [RunRecord.from_row(row) for row in cur.fetchall()]

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
