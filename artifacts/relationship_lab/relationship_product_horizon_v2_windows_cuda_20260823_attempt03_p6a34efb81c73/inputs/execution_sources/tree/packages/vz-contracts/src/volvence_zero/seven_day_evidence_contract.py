"""Cross-wheel constants for the frozen seven-day evidence protocol."""

from __future__ import annotations


# Source archive used after days 1..6 by the shuffled-history intervention.
# This lives in vz-contracts so runtime evidence code does not reverse-import
# a lifeform implementation wheel to recover a preregistered constant.
SEVEN_DAY_SHUFFLED_SOURCE_DAYS = (1, 1, 2, 1, 4, 3)


__all__ = ("SEVEN_DAY_SHUFFLED_SOURCE_DAYS",)
