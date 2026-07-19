"""运行时路径。

默认 layout：
  <root>/
    .labs/
      cas/<sha[:2]>/<sha>.bin
      index.sqlite
    experiments/<run_id>/
      manifest.json
      readouts/
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LabsPaths:
    root: Path

    @property
    def labs_dir(self) -> Path:
        return self.root / ".labs"

    @property
    def cas_dir(self) -> Path:
        return self.labs_dir / "cas"

    @property
    def index_db(self) -> Path:
        return self.labs_dir / "index.sqlite"

    @property
    def experiments_dir(self) -> Path:
        return self.root / "experiments"

    def experiment_dir(self, run_id: str) -> Path:
        return self.experiments_dir / run_id

    def ensure(self) -> "LabsPaths":
        self.cas_dir.mkdir(parents=True, exist_ok=True)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        return self


def default_paths(root: os.PathLike | str | None = None) -> LabsPaths:
    """Resolve root from arg > env > cwd."""
    if root is None:
        root = os.environ.get("VOLVENCE_LABS_ROOT")
    if root is None:
        root = Path.cwd()
    return LabsPaths(root=Path(root).resolve()).ensure()
