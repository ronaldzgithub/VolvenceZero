"""WiringLevel / AblationCell / WiringProfile + 4 内置 profile.

设计原则（DESIGN.md §1, §5）：
- WiringLevel 与 AblationCell 是正交的两个轴。
- 业务代码读 WiringLevel 用方法（is_active / is_shadow），不用 == 比较。
- 4 内置 profile：dev / shadow / canary / active。
- 阶段 1+：profile 可从 YAML 文件加载（load_profile）。
"""

from __future__ import annotations

import enum
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional


class WiringLevel(str, enum.Enum):
    DISABLED = "disabled"
    SHADOW = "shadow"
    ACTIVE = "active"

    def is_active(self) -> bool:
        return self is WiringLevel.ACTIVE

    def is_shadow(self) -> bool:
        return self is WiringLevel.SHADOW

    def participates(self) -> bool:
        """probe is mounted (SHADOW or ACTIVE), but not necessarily affecting behavior."""
        return self is not WiringLevel.DISABLED


class AblationCell(str, enum.Enum):
    BASELINE = "baseline"
    PROBE_ON = "probe_on"
    PROBE_OFF = "probe_off"
    COUNTERFACTUAL = "counterfactual"


@dataclass(frozen=True)
class WiringProfile:
    name: str
    default_level: WiringLevel
    seeds: tuple[int, ...]
    cells: tuple[AblationCell, ...]
    probe_overrides: Mapping[str, WiringLevel] = field(default_factory=dict)

    def level_for(self, probe_id: str) -> WiringLevel:
        return self.probe_overrides.get(probe_id, self.default_level)


def _profile(
    name: str,
    *,
    default_level: WiringLevel,
    seeds: Iterable[int],
    cells: Iterable[AblationCell],
    overrides: Optional[Mapping[str, WiringLevel]] = None,
) -> WiringProfile:
    return WiringProfile(
        name=name,
        default_level=default_level,
        seeds=tuple(seeds),
        cells=tuple(cells),
        probe_overrides=dict(overrides or {}),
    )


_DEV = _profile(
    "dev",
    default_level=WiringLevel.SHADOW,
    seeds=(0,),
    cells=(AblationCell.BASELINE, AblationCell.PROBE_ON),
)

_SHADOW = _profile(
    "shadow",
    default_level=WiringLevel.SHADOW,
    seeds=(0, 1, 2, 3, 4, 5, 6, 7),
    cells=(
        AblationCell.BASELINE,
        AblationCell.PROBE_ON,
        AblationCell.PROBE_OFF,
        AblationCell.COUNTERFACTUAL,
    ),
)

_CANARY = _profile(
    "canary",
    default_level=WiringLevel.SHADOW,
    seeds=(0, 1, 2, 3),
    cells=(
        AblationCell.BASELINE,
        AblationCell.PROBE_ON,
    ),
    overrides={},
)

_ACTIVE = _profile(
    "active",
    default_level=WiringLevel.ACTIVE,
    seeds=(0, 1, 2, 3),
    cells=(AblationCell.PROBE_ON,),
)


_BUILTIN: dict[str, WiringProfile] = {
    p.name: p for p in (_DEV, _SHADOW, _CANARY, _ACTIVE)
}


def builtin_profiles() -> dict[str, WiringProfile]:
    return dict(_BUILTIN)


def get_profile(name: str) -> WiringProfile:
    """Get a profile by name. Checks builtins first, then configs/wiring_profiles/."""
    if name in _BUILTIN:
        return _BUILTIN[name]
    # Try loading from YAML in configs/wiring_profiles/
    configs_dir = _find_configs_dir()
    if configs_dir is not None:
        yaml_path = configs_dir / "wiring_profiles" / f"{name}.yaml"
        if yaml_path.exists():
            return load_profile(yaml_path)
    raise KeyError(f"unknown wiring profile: {name!r} (have {sorted(_BUILTIN)})")


def load_profile(path: os.PathLike | str) -> WiringProfile:
    """Load a WiringProfile from a YAML file.

    Expected YAML keys: name, default_level, seeds, cells, probe_overrides (optional).
    """
    import yaml  # lazy import; pyyaml is a stage 1 dependency

    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f)

    return WiringProfile(
        name=data["name"],
        default_level=WiringLevel(data["default_level"]),
        seeds=tuple(int(s) for s in data.get("seeds", [0])),
        cells=tuple(AblationCell(c) for c in data.get("cells", ["baseline", "probe_on"])),
        probe_overrides={
            k: WiringLevel(v)
            for k, v in (data.get("probe_overrides") or {}).items()
        },
    )


def _find_configs_dir() -> Optional[Path]:
    """Walk up from CWD or VOLVENCE_LABS_ROOT to find configs/."""
    root = os.environ.get("VOLVENCE_LABS_ROOT")
    if root:
        p = Path(root) / "configs"
        if p.is_dir():
            return p
    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents]:
        candidate = parent / "configs"
        if (candidate / "wiring_profiles").is_dir():
            return candidate
        if parent == parent.parent:
            break
    return None
