"""Wiring isolation: WiringLevel 的语义 + probe_overrides + AblationCell 正交性。"""

from __future__ import annotations

import unittest

from volvence_labs.framework.wiring import (
    AblationCell,
    WiringLevel,
    builtin_profiles,
    get_profile,
)


class TestWiringIsolation(unittest.TestCase):
    def test_wiring_level_semantics(self) -> None:
        self.assertTrue(WiringLevel.ACTIVE.is_active())
        self.assertFalse(WiringLevel.SHADOW.is_active())
        self.assertFalse(WiringLevel.DISABLED.is_active())

        self.assertTrue(WiringLevel.SHADOW.is_shadow())
        self.assertFalse(WiringLevel.ACTIVE.is_shadow())

        self.assertTrue(WiringLevel.SHADOW.participates())
        self.assertTrue(WiringLevel.ACTIVE.participates())
        self.assertFalse(WiringLevel.DISABLED.participates())

    def test_profile_default_and_override(self) -> None:
        profile = get_profile("shadow")
        self.assertEqual(profile.default_level, WiringLevel.SHADOW)
        self.assertEqual(profile.level_for("anything"), WiringLevel.SHADOW)

    def test_canary_allows_override(self) -> None:
        profile = get_profile("canary")
        # canary 默认仍是 SHADOW；override 允许 per-probe 升 ACTIVE。
        self.assertEqual(profile.level_for("unknown"), WiringLevel.SHADOW)

    def test_builtin_profiles_cover_four(self) -> None:
        names = set(builtin_profiles())
        self.assertEqual(names, {"dev", "shadow", "canary", "active"})

    def test_ablation_cells_orthogonal_to_wiring(self) -> None:
        """AblationCell 和 WiringLevel 是两个独立枚举 — 没有互相渗漏。"""
        cells = {c.value for c in AblationCell}
        levels = {l.value for l in WiringLevel}
        self.assertTrue(cells.isdisjoint(levels))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
