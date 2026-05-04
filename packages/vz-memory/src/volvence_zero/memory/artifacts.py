"""Audit-friendly explicit artifact store for memory entries."""

from __future__ import annotations

from dataclasses import replace

from volvence_zero.memory.contracts import MemoryEntry, MemoryStratum
from volvence_zero.memory.retrieval import _clamp_strength


class ArtifactStore:
    """Audit-friendly durable artifact layer.

    This layer stores explicit cards, beliefs, and other rollback-friendly
    artifacts. It is not the primary memory substrate; learned multi-timescale
    state lives in the CMS core.
    """

    def __init__(self) -> None:
        self._entries: dict[str, MemoryEntry] = {}
        self._by_stratum: dict[MemoryStratum, list[str]] = {
            MemoryStratum.TRANSIENT: [],
            MemoryStratum.EPISODIC: [],
            MemoryStratum.DURABLE: [],
            MemoryStratum.DERIVED: [],
        }
        self._pending_promotions: list[str] = []
        self._pending_decays: list[str] = []

    @property
    def pending_promotions(self) -> tuple[str, ...]:
        return tuple(self._pending_promotions)

    @property
    def pending_decays(self) -> tuple[str, ...]:
        return tuple(self._pending_decays)

    def write(self, entry: MemoryEntry) -> MemoryEntry:
        self._entries[entry.entry_id] = entry
        stratum = MemoryStratum(entry.stratum)
        if entry.entry_id not in self._by_stratum[stratum]:
            self._by_stratum[stratum].append(entry.entry_id)
        if stratum in {MemoryStratum.TRANSIENT, MemoryStratum.EPISODIC} and entry.entry_id not in self._pending_promotions:
            self._pending_promotions.append(entry.entry_id)
        if stratum is MemoryStratum.TRANSIENT and entry.strength < 0.4 and entry.entry_id not in self._pending_decays:
            self._pending_decays.append(entry.entry_id)
        return entry

    def get(self, entry_id: str) -> MemoryEntry | None:
        return self._entries.get(entry_id)

    def touch(self, entry_id: str, *, timestamp_ms: int) -> MemoryEntry | None:
        entry = self._entries.get(entry_id)
        if entry is None:
            return None
        updated = replace(entry, last_accessed_ms=timestamp_ms)
        self._entries[entry_id] = updated
        return updated

    def replace_entry(self, entry: MemoryEntry) -> None:
        self._entries[entry.entry_id] = entry
        stratum = MemoryStratum(entry.stratum)
        for candidate in MemoryStratum:
            bucket = self._by_stratum[candidate]
            if candidate is stratum:
                if entry.entry_id not in bucket:
                    bucket.append(entry.entry_id)
            elif entry.entry_id in bucket:
                bucket.remove(entry.entry_id)

    def entries_for(self, stratum: MemoryStratum) -> tuple[MemoryEntry, ...]:
        return tuple(self._entries[entry_id] for entry_id in self._by_stratum[stratum])

    def entries_in(self, strata: tuple[MemoryStratum, ...]) -> tuple[MemoryEntry, ...]:
        return tuple(
            self._entries[entry_id]
            for stratum in strata
            for entry_id in self._by_stratum[stratum]
        )

    def total_entries_by_stratum(self) -> tuple[tuple[str, int], ...]:
        return tuple((stratum.value, len(self._by_stratum[stratum])) for stratum in MemoryStratum)

    def entry_count(self) -> int:
        return len(self._entries)

    def promote(
        self,
        *,
        entry_id: str,
        promotion_threshold: float,
        promotion_boost: float,
        timestamp_ms: int,
    ) -> MemoryEntry | None:
        entry = self._entries.get(entry_id)
        if entry is None or entry.strength < promotion_threshold:
            return None
        updated = replace(
            entry,
            stratum=MemoryStratum.DURABLE.value,
            strength=_clamp_strength(max(entry.strength, 0.55 + promotion_boost * 0.35)),
            last_accessed_ms=timestamp_ms,
        )
        self.replace_entry(updated)
        return updated

    def decay(
        self,
        *,
        entry_id: str,
        decay_scale: float,
        timestamp_ms: int,
    ) -> MemoryEntry | None:
        entry = self._entries.get(entry_id)
        if entry is None:
            return None
        updated = replace(
            entry,
            strength=_clamp_strength(entry.strength * max(0.55, 1.0 - decay_scale * 0.35)),
            last_accessed_ms=timestamp_ms,
        )
        self._entries[entry_id] = updated
        return updated

    def export_entries(self) -> tuple[MemoryEntry, ...]:
        return tuple(self._entries.values())

    def restore(
        self,
        *,
        entries: tuple[MemoryEntry, ...],
        pending_promotions: tuple[str, ...],
        pending_decays: tuple[str, ...],
    ) -> None:
        self._entries = {entry.entry_id: entry for entry in entries}
        self._by_stratum = {
            stratum: [entry.entry_id for entry in entries if entry.stratum == stratum.value]
            for stratum in MemoryStratum
        }
        self._pending_promotions = list(pending_promotions)
        self._pending_decays = list(pending_decays)

