"""P1 context and persistence surfaces for Relationship Lab.

This module is an offline orchestrator. ``vz-memory.MemoryStore`` remains the
only owner of structured Volvence memory, while ``companion-ref-harness`` owns
the reference RAG index. The orchestrator writes public observations through
their formal APIs and only quotes the records they publish. It never reads
generator truth while constructing model context.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from dataclasses import dataclass, replace
from enum import Enum

from companion_ref_harness import ComponentSet, HarnessComponent, HarnessPolicy
from companion_ref_harness.embed import EmbedEntry, Embedder, top_k
from companion_ref_harness.store import StoreMode, open_store
from volvence_zero.memory import (
    FileSystemPersistenceBackend,
    MemoryEntry,
    MemoryStore,
    MemoryStratum,
    MemoryWriteRequest,
    RetrievalQuery,
    Track,
)

from lifeform_domain_emogpt.lab import (
    RelationshipDatasetSplit,
    RelationshipHistoryEvent,
    RelationshipObservation,
    RelationshipTransferDataset,
    load_relationship_transfer_dataset,
    relationship_transfer_package_dir,
    sha256_json,
)


RELATIONSHIP_P1_CONTEXT_SCHEMA_VERSION = "relationship-p1-context.v1"
RELATIONSHIP_P1_STATE_DIGEST_SCHEMA_VERSION = "relationship-p1-state-digest.v1"
RELATIONSHIP_P1_CONSOLE_PROBE_SCHEMA_VERSION = "relationship-p1-console-probe.v1"
RELATIONSHIP_P1_BACKGROUND_SCHEMA_VERSION = "relationship-p1-background-templates.v1"
RELATIONSHIP_P1_DEFAULT_DEPTHS: tuple[int, ...] = (0, 8, 32)
RELATIONSHIP_P1_RAG_TOP_K = 4
RELATIONSHIP_P1_RAG_MIN_SCORE = -1.0
_PERSISTENCE_KEY = "memory/store"
_RAG_QUERY_PREFIX = (
    "检索这个用户过去明确的关系回应结果：哪些 assistant action 让用户感到 "
    "FELT_HEARD 或 HELPED，哪些导致 MISSED 或 OVER_DIRECTIVE。当前新消息："
)
_HEX_DIGITS = frozenset("0123456789abcdef")


class RelationshipP1Arm(str, Enum):
    STATELESS = "stateless"
    PROMPT_STEELMAN = "prompt-steelman"
    RAG_STEELMAN = "rag-steelman"
    STRUCTURED_STATE = "structured-state"


RELATIONSHIP_P1_ARMS: tuple[RelationshipP1Arm, ...] = tuple(RelationshipP1Arm)


class RelationshipP1RagCandidateSurface(str, Enum):
    """Typed owner-record surface admitted to semantic top-k retrieval."""

    ALL_PUBLIC_RECORDS = "all_public_records"
    RELATIONSHIP_OUTCOMES_ONLY = "relationship_outcomes_only"


def relationship_p1_background_template_path(
    package_name: str | None = None,
) -> pathlib.Path:
    root = (
        relationship_transfer_package_dir()
        if package_name is None
        else relationship_transfer_package_dir(package_name)
    )
    return root / "p1_background_templates.json"


def _require_scope_hash(scope_hash: str) -> None:
    if len(scope_hash) != 64 or any(char not in _HEX_DIGITS for char in scope_hash):
        raise ValueError("scope_hash must be a lowercase sha256 digest")


def _require_sha256(value: object, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in _HEX_DIGITS for char in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase sha256 digest")
    return value


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_background_templates(
    package_name: str | None = None,
) -> tuple[tuple[str, str], ...]:
    path = relationship_p1_background_template_path(package_name)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {"schema_version", "templates"}:
        raise ValueError("P1 background template file has an invalid shape")
    if raw["schema_version"] != RELATIONSHIP_P1_BACKGROUND_SCHEMA_VERSION:
        raise ValueError("P1 background template schema_version mismatch")
    templates = raw["templates"]
    if not isinstance(templates, list) or len(templates) < 4:
        raise ValueError("P1 requires at least four background templates")
    parsed: list[tuple[str, str]] = []
    for index, item in enumerate(templates):
        if not isinstance(item, dict) or set(item) != {"user", "assistant"}:
            raise ValueError(f"background template {index} has an invalid shape")
        user = item["user"]
        assistant = item["assistant"]
        if not isinstance(user, str) or not user.strip():
            raise ValueError(f"background template {index}.user must be non-empty")
        if not isinstance(assistant, str) or not assistant.strip():
            raise ValueError(f"background template {index}.assistant must be non-empty")
        parsed.append((user, assistant))
    return tuple(parsed)


@dataclass(frozen=True)
class PublicRelationshipContextItem:
    item_id: str
    kind: str
    content: str
    tags: tuple[str, ...]
    timestamp_ms: int
    timestamp_iso: str
    background_index: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"relationship_outcome", "background"}:
            raise ValueError("context item kind is unsupported")
        if not self.item_id.strip() or not self.content.strip():
            raise ValueError("context item id/content must be non-empty")
        if self.tags != tuple(sorted(set(self.tags))):
            raise ValueError("context item tags must be sorted and unique")
        if self.kind == "background" and self.background_index is None:
            raise ValueError("background item requires background_index")
        if self.kind == "relationship_outcome" and self.background_index is not None:
            raise ValueError("relationship outcome cannot have background_index")


def _signal_content(event: RelationshipHistoryEvent) -> str:
    return "\n".join(
        (
            "[public relationship outcome evidence]",
            f"surface_family: {event.surface_family}",
            f"user: {event.user_utterance}",
            f"assistant_action: {event.assistant_action.value}",
            f"typed_external_outcome: {event.typed_outcome.value}",
            f"user_reaction: {event.user_reaction}",
        )
    )


def _background_content(*, index: int, user: str, assistant: str) -> str:
    return "\n".join(
        (
            f"[ordinary prior session {index + 1}]",
            f"user: {user}",
            f"assistant: {assistant}",
        )
    )


def _context_items(
    observation: RelationshipObservation,
    *,
    background_depth: int,
    templates: tuple[tuple[str, str], ...],
) -> tuple[PublicRelationshipContextItem, ...]:
    if background_depth < 0:
        raise ValueError("background_depth must be non-negative")
    background: list[PublicRelationshipContextItem] = []
    for index in range(background_depth):
        user, assistant = templates[index % len(templates)]
        background.append(
            PublicRelationshipContextItem(
                item_id=f"background-{index:03d}",
                kind="background",
                content=_background_content(
                    index=index,
                    user=user,
                    assistant=assistant,
                ),
                tags=("background", "relationship-lab"),
                timestamp_ms=1_800_000_000_000 + index * 10,
                timestamp_iso=f"2027-01-15T00:00:{index:02d}+00:00",
                background_index=index,
            )
        )
    signal_items = tuple(
        PublicRelationshipContextItem(
            item_id=event.event_id,
            kind="relationship_outcome",
            content=_signal_content(event),
            tags=(
                f"event:{event.event_id}",
                "relationship-lab",
                "relationship-outcome",
            ),
            timestamp_ms=1_800_000_000_005 + signal_index * 10_000,
            timestamp_iso=f"2027-01-15T00:01:0{signal_index}+00:00",
        )
        for signal_index, event in enumerate(observation.histories)
    )
    if len(signal_items) not in {2, 4}:
        raise ValueError("P1 supports exactly two v1 or four v2 relationship histories")
    if len(signal_items) == 2:
        # Preserve the byte-for-byte v1 evaluated context surface.
        first_cut = background_depth // 3
        second_cut = (background_depth * 2) // 3
        ordered = (
            *background[:first_cut],
            signal_items[0],
            *background[first_cut:second_cut],
            signal_items[1],
            *background[second_cut:],
        )
    else:
        cuts = tuple(
            (background_depth * (index + 1)) // (len(signal_items) + 1)
            for index in range(len(signal_items))
        )
        assembled: list[PublicRelationshipContextItem] = []
        previous_cut = 0
        for signal, cut in zip(signal_items, cuts, strict=True):
            assembled.extend(background[previous_cut:cut])
            assembled.append(signal)
            previous_cut = cut
        assembled.extend(background[previous_cut:])
        ordered = tuple(assembled)
    return tuple(
        replace(
            item,
            timestamp_ms=1_800_000_000_000 + index,
            timestamp_iso=f"2027-01-15T00:{index // 60:02d}:{index % 60:02d}+00:00",
        )
        for index, item in enumerate(ordered)
    )


def _memory_entry_payload(entry: MemoryEntry) -> dict[str, object]:
    return {
        "entry_id": entry.entry_id,
        "content": entry.content,
        "track": entry.track.value,
        "stratum": entry.stratum,
        "created_at_ms": entry.created_at_ms,
        "last_accessed_ms": entry.last_accessed_ms,
        "strength": entry.strength,
        "tags": list(entry.tags),
        "subject_ids": list(entry.subject_ids),
        "audience_ids": list(entry.audience_ids),
    }


class StructuredRelationshipStateStore:
    """Thin adapter over the existing MemoryStore owner API."""

    def __init__(
        self,
        *,
        root: pathlib.Path,
        scope_hash: str,
        load_existing: bool,
    ) -> None:
        _require_scope_hash(scope_hash)
        self._scope_hash = scope_hash
        self._subject_id = f"relationship-user:{scope_hash}"
        scope_root = pathlib.Path(root) / scope_hash
        backend = FileSystemPersistenceBackend(base_dir=str(scope_root))
        self._store = MemoryStore(persistence_backend=backend)
        loaded = self._store.load_from_backend(key=_PERSISTENCE_KEY)
        if load_existing and not loaded:
            raise FileNotFoundError(f"no persisted MemoryStore for {scope_hash}")
        if not load_existing and loaded:
            raise FileExistsError(f"MemoryStore already exists for {scope_hash}")

    def append_items(self, items: tuple[PublicRelationshipContextItem, ...]) -> None:
        if self._store.entries_for(MemoryStratum.EPISODIC):
            raise RuntimeError("append_items requires a fresh user store")
        for item in items:
            self._write_item(item)

    def _write_item(self, item: PublicRelationshipContextItem) -> MemoryEntry:
        return self._store.write(
            MemoryWriteRequest(
                content=item.content,
                track=Track.WORLD,
                stratum=MemoryStratum.EPISODIC,
                tags=item.tags,
                strength=0.85 if item.kind == "relationship_outcome" else 0.35,
                subject_ids=(self._subject_id,),
            ),
            timestamp_ms=item.timestamp_ms,
        )

    def persist(self) -> None:
        if not self._store.save_to_backend(key=_PERSISTENCE_KEY):
            raise RuntimeError("MemoryStore did not persist to its configured backend")

    def entries(self) -> tuple[MemoryEntry, ...]:
        return self._store.entries_for(MemoryStratum.EPISODIC)

    def state_digest(self) -> str:
        payload = tuple(
            _memory_entry_payload(entry) for entry in sorted(self.entries(), key=lambda item: item.entry_id)
        )
        return sha256_json(payload)

    def relationship_evidence(
        self,
        *,
        expected_records: int = 2,
    ) -> tuple[MemoryEntry, ...]:
        if expected_records not in {2, 4}:
            raise ValueError("structured relationship recall supports two or four records")
        result = self._store.retrieve(
            RetrievalQuery(
                text="relationship outcome evidence",
                track=Track.WORLD,
                strata=(MemoryStratum.EPISODIC,),
                facets=("relationship-outcome",),
                limit=expected_records,
            ),
            timestamp_ms=1_800_000_100_000,
            active_subject_ids=(self._subject_id,),
            record_access=False,
        )
        entries = tuple(sorted(result.entries, key=lambda item: item.created_at_ms))
        if len(entries) != expected_records or any(
            "relationship-outcome" not in entry.tags for entry in entries
        ):
            raise RuntimeError(
                "MemoryStore did not recover the expected typed relationship records"
            )
        if result.suppressed_cross_scope_entries:
            raise RuntimeError("structured relationship recall crossed user scope")
        return entries

    def render_context(
        self,
        *,
        expected_records: int = 2,
    ) -> tuple[str, tuple[str, ...]]:
        entries = self.relationship_evidence(expected_records=expected_records)
        return (
            "\n\n".join(f"[memory-owner:{entry.stratum}] {entry.content}" for entry in entries),
            tuple(entry.entry_id for entry in entries),
        )

    def _entry_for_event(self, event_id: str) -> MemoryEntry:
        tag = f"event:{event_id}"
        matches = tuple(entry for entry in self.entries() if tag in entry.tags)
        if len(matches) != 1:
            raise KeyError(f"expected one entry for event {event_id!r}, got {len(matches)}")
        return matches[0]

    def correct_event(
        self,
        *,
        original_event_id: str,
        replacement_event: RelationshipHistoryEvent,
        timestamp_ms: int,
    ) -> MemoryEntry:
        original = self._entry_for_event(original_event_id)
        removed = self._store.delete_artifact_entry(original.entry_id)
        if removed is None:
            raise RuntimeError("MemoryStore lost the event selected for correction")
        replacement_item = PublicRelationshipContextItem(
            item_id=replacement_event.event_id,
            kind="relationship_outcome",
            content=_signal_content(replacement_event),
            tags=(
                f"event:{replacement_event.event_id}",
                "relationship-lab",
                "relationship-outcome",
            ),
            timestamp_ms=timestamp_ms,
            timestamp_iso="2027-01-15T01:00:00+00:00",
        )
        return self._write_item(replacement_item)

    def delete_event(self, event_id: str) -> MemoryEntry:
        entry = self._entry_for_event(event_id)
        removed = self._store.delete_artifact_entry(entry.entry_id)
        if removed is None:
            raise RuntimeError("MemoryStore lost the event selected for deletion")
        return removed


def _rag_entry_payload(entry: EmbedEntry) -> dict[str, object]:
    return {
        "scope_key": entry.scope_key,
        "turn_id": entry.turn_id,
        "role": entry.role,
        "content": entry.content,
        "embedding": list(entry.embedding),
        "ts": entry.ts,
    }


@dataclass(frozen=True)
class RelationshipP1Context:
    arm: RelationshipP1Arm
    scene_id: str
    background_depth: int
    context_text: str
    source_evidence_refs: tuple[str, ...]
    schema_version: str = RELATIONSHIP_P1_CONTEXT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RELATIONSHIP_P1_CONTEXT_SCHEMA_VERSION:
            raise ValueError("P1 context schema_version mismatch")
        if not self.scene_id.strip() or self.background_depth < 0:
            raise ValueError("P1 context scene/depth is invalid")
        if self.arm is RelationshipP1Arm.STATELESS:
            if self.context_text or self.source_evidence_refs:
                raise ValueError("stateless context must be exactly empty")
        elif not self.context_text.strip() or not self.source_evidence_refs:
            raise ValueError("contextual arm requires text and evidence refs")

    @property
    def context_sha256(self) -> str:
        return hashlib.sha256(self.context_text.encode("utf-8")).hexdigest()

    def render_user_message(self, current_input: str) -> str:
        if self.arm is RelationshipP1Arm.STATELESS:
            return current_input
        return (
            "<public_history_evidence>\n"
            f"{self.context_text}\n"
            "</public_history_evidence>\n\n"
            "<current_user_message>\n"
            f"{current_input}\n"
            "</current_user_message>"
        )


@dataclass(frozen=True)
class PersistedRelationshipP1StateDigest:
    structured_scope_digests: tuple[tuple[str, str, int], ...]
    rag_scope_digests: tuple[tuple[str, str, int], ...]
    schema_version: str = RELATIONSHIP_P1_STATE_DIGEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RELATIONSHIP_P1_STATE_DIGEST_SCHEMA_VERSION:
            raise ValueError("P1 state digest schema_version mismatch")
        if not self.structured_scope_digests or not self.rag_scope_digests:
            raise ValueError("P1 state digest requires both persistence surfaces")

    @property
    def artifact_id(self) -> str:
        return sha256_json(self.to_payload())

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "structured_scope_digests": [
                {"scope_hash": scope, "sha256": digest, "records": records}
                for scope, digest, records in self.structured_scope_digests
            ],
            "rag_scope_digests": [
                {"scope_hash": scope, "sha256": digest, "records": records}
                for scope, digest, records in self.rag_scope_digests
            ],
        }


@dataclass(frozen=True)
class RelationshipP1ContextBundle:
    dataset_fingerprint: str
    background_depths: tuple[int, ...]
    background_templates_sha256: str
    rag_config_sha256: str
    contexts: tuple[RelationshipP1Context, ...]
    persisted_state: PersistedRelationshipP1StateDigest

    def __post_init__(self) -> None:
        if self.background_depths != tuple(sorted(set(self.background_depths))):
            raise ValueError("background_depths must be sorted and unique")
        expected = {(context.scene_id, context.background_depth, context.arm) for context in self.contexts}
        if len(expected) != len(self.contexts):
            raise ValueError("P1 contexts must have unique scene/depth/arm keys")

    @property
    def max_background_depth(self) -> int:
        return self.background_depths[-1]

    @property
    def artifact_id(self) -> str:
        return sha256_json(self.to_summary_payload())

    def to_summary_payload(self) -> dict[str, object]:
        return {
            "dataset_fingerprint": self.dataset_fingerprint,
            "background_depths": list(self.background_depths),
            "background_templates_sha256": self.background_templates_sha256,
            "rag_config_sha256": self.rag_config_sha256,
            "contexts": [
                {
                    "scene_id": item.scene_id,
                    "background_depth": item.background_depth,
                    "arm": item.arm.value,
                    "context_sha256": item.context_sha256,
                    "source_evidence_refs": list(item.source_evidence_refs),
                }
                for item in self.contexts
            ],
            "persisted_state_artifact_id": self.persisted_state.artifact_id,
        }

    def context(
        self,
        *,
        scene_id: str,
        arm: RelationshipP1Arm,
        background_depth: int | None = None,
    ) -> RelationshipP1Context:
        depth = self.max_background_depth if background_depth is None else background_depth
        matches = tuple(
            item
            for item in self.contexts
            if item.scene_id == scene_id and item.arm is arm and item.background_depth == depth
        )
        if len(matches) != 1:
            raise KeyError((scene_id, arm.value, depth))
        return matches[0]


@dataclass(frozen=True)
class RelationshipP1RagReplayOrder:
    scene_id: str
    background_depth: int
    turn_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.scene_id.strip() or self.background_depth < 0:
            raise ValueError("P1 RAG replay scene/depth is invalid")
        if not self.turn_ids or len(set(self.turn_ids)) != len(self.turn_ids):
            raise ValueError("P1 RAG replay turn ids must be unique and non-empty")


@dataclass(frozen=True)
class RelationshipP1ContextReplayManifest:
    artifact_id: str
    dataset_fingerprint: str
    background_depths: tuple[int, ...]
    background_templates_sha256: str
    rag_config_sha256: str
    persisted_state_artifact_id: str
    context_hashes: tuple[tuple[str, int, RelationshipP1Arm, str], ...]
    rag_orders: tuple[RelationshipP1RagReplayOrder, ...]

    def __post_init__(self) -> None:
        for field_name, value in (
            ("artifact_id", self.artifact_id),
            ("dataset_fingerprint", self.dataset_fingerprint),
            ("background_templates_sha256", self.background_templates_sha256),
            ("rag_config_sha256", self.rag_config_sha256),
            ("persisted_state_artifact_id", self.persisted_state_artifact_id),
        ):
            _require_sha256(value, f"P1 replay manifest {field_name}")
        if self.background_depths != tuple(sorted(set(self.background_depths))):
            raise ValueError("P1 replay manifest depths must be sorted and unique")
        context_keys = tuple(
            (scene_id, depth, arm)
            for scene_id, depth, arm, _digest in self.context_hashes
        )
        if not context_keys or len(set(context_keys)) != len(context_keys):
            raise ValueError("P1 replay manifest context keys must be unique")
        rag_keys = tuple(
            (item.scene_id, item.background_depth) for item in self.rag_orders
        )
        if not rag_keys or len(set(rag_keys)) != len(rag_keys):
            raise ValueError("P1 replay manifest RAG keys must be unique")

    def validate_model_inputs(self, bundle: RelationshipP1ContextBundle) -> None:
        if (
            bundle.dataset_fingerprint != self.dataset_fingerprint
            or bundle.background_depths != self.background_depths
            or bundle.background_templates_sha256
            != self.background_templates_sha256
            or bundle.rag_config_sha256 != self.rag_config_sha256
        ):
            raise ValueError("P1 replayed bundle lineage diverges from manifest")
        actual_hashes = tuple(
            (
                context.scene_id,
                context.background_depth,
                context.arm,
                context.context_sha256,
            )
            for context in bundle.contexts
        )
        if actual_hashes != self.context_hashes:
            raise ValueError("P1 replayed model-input hashes diverge from manifest")
        actual_rag_orders = tuple(
            RelationshipP1RagReplayOrder(
                scene_id=context.scene_id,
                background_depth=context.background_depth,
                turn_ids=context.source_evidence_refs,
            )
            for context in bundle.contexts
            if context.arm is RelationshipP1Arm.RAG_STEELMAN
        )
        if actual_rag_orders != self.rag_orders:
            raise ValueError("P1 replayed RAG order diverges from manifest")


def load_relationship_p1_context_replay_manifest(
    path: pathlib.Path,
    *,
    dataset: RelationshipTransferDataset,
) -> RelationshipP1ContextReplayManifest:
    file_path = pathlib.Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"P1 context replay manifest is missing: {file_path}")
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("P1 context replay manifest is not valid JSON") from exc
    expected_top_level = {
        "artifact_id",
        "dataset_fingerprint",
        "background_depths",
        "background_templates_sha256",
        "rag_config_sha256",
        "contexts",
        "persisted_state_artifact_id",
    }
    if not isinstance(payload, dict) or set(payload) != expected_top_level:
        raise ValueError("P1 context replay manifest fields do not match schema")
    artifact_id = _require_sha256(payload["artifact_id"], "P1 context artifact")
    identity = {key: value for key, value in payload.items() if key != "artifact_id"}
    if sha256_json(identity) != artifact_id:
        raise ValueError("P1 context replay manifest artifact id mismatch")
    depths_raw = payload["background_depths"]
    if not isinstance(depths_raw, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in depths_raw
    ):
        raise ValueError("P1 context replay depths must be an integer array")
    contexts_raw = payload["contexts"]
    if not isinstance(contexts_raw, list) or not contexts_raw:
        raise ValueError("P1 context replay records must be non-empty")
    context_hashes: list[tuple[str, int, RelationshipP1Arm, str]] = []
    rag_orders: list[RelationshipP1RagReplayOrder] = []
    for index, item in enumerate(contexts_raw):
        expected_fields = {
            "scene_id",
            "background_depth",
            "arm",
            "context_sha256",
            "source_evidence_refs",
        }
        if not isinstance(item, dict) or set(item) != expected_fields:
            raise ValueError(f"P1 context replay record {index} fields mismatch")
        scene_id = item["scene_id"]
        depth = item["background_depth"]
        refs = item["source_evidence_refs"]
        if not isinstance(scene_id, str) or not scene_id.strip():
            raise ValueError("P1 context replay scene id must be non-empty")
        if isinstance(depth, bool) or not isinstance(depth, int) or depth < 0:
            raise ValueError("P1 context replay depth must be non-negative")
        if not isinstance(refs, list) or any(
            not isinstance(ref, str) or not ref.strip() for ref in refs
        ):
            raise ValueError("P1 context replay refs must be a string array")
        arm = RelationshipP1Arm(item["arm"])
        context_hashes.append(
            (
                scene_id,
                depth,
                arm,
                _require_sha256(
                    item["context_sha256"], "P1 replay context hash"
                ),
            )
        )
        if arm is RelationshipP1Arm.RAG_STEELMAN:
            rag_orders.append(
                RelationshipP1RagReplayOrder(
                    scene_id=scene_id,
                    background_depth=depth,
                    turn_ids=tuple(refs),
                )
            )
    expected_keys = {
        (observation.scene_id, depth, arm)
        for observation in dataset.observations
        for depth in depths_raw
        for arm in RELATIONSHIP_P1_ARMS
    }
    if {
        (scene_id, depth, arm)
        for scene_id, depth, arm, _digest in context_hashes
    } != expected_keys:
        raise ValueError("P1 context replay coverage diverges from dataset")
    history_ids_by_scene = {
        observation.scene_id: {event.event_id for event in observation.histories}
        for observation in dataset.observations
    }
    for order in rag_orders:
        if set(order.turn_ids) != history_ids_by_scene[order.scene_id]:
            raise ValueError("P1 context replay RAG refs diverge from public histories")
    dataset_fingerprint = _require_sha256(
        payload["dataset_fingerprint"], "P1 replay dataset fingerprint"
    )
    if dataset_fingerprint != dataset.dataset_fingerprint:
        raise ValueError("P1 context replay dataset fingerprint mismatch")
    return RelationshipP1ContextReplayManifest(
        artifact_id=artifact_id,
        dataset_fingerprint=dataset_fingerprint,
        background_depths=tuple(depths_raw),
        background_templates_sha256=_require_sha256(
            payload["background_templates_sha256"],
            "P1 replay background templates",
        ),
        rag_config_sha256=_require_sha256(
            payload["rag_config_sha256"], "P1 replay RAG config"
        ),
        persisted_state_artifact_id=_require_sha256(
            payload["persisted_state_artifact_id"],
            "P1 replay persisted state",
        ),
        context_hashes=tuple(context_hashes),
        rag_orders=tuple(rag_orders),
    )


def _state_root_must_be_fresh(state_root: pathlib.Path) -> None:
    root = pathlib.Path(state_root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"P1 state root is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)


def _rag_context_from_entries(
    *,
    policy: HarnessPolicy,
    scope_hash: str,
    current_input: str,
    entries: tuple[EmbedEntry, ...],
    query_embedding: tuple[float, ...],
    top_k_count: int,
    replay_turn_ids: tuple[str, ...] | None = None,
) -> tuple[str, tuple[str, ...]]:
    if top_k_count <= 0:
        raise ValueError("RAG top_k_count must be positive")
    retrieved = top_k(
        query=query_embedding,
        entries=entries,
        k=top_k_count,
        min_score=RELATIONSHIP_P1_RAG_MIN_SCORE,
    )
    if replay_turn_ids is not None:
        retrieved_by_id = {entry.turn_id: entry for entry in retrieved}
        if (
            len(retrieved_by_id) != len(retrieved)
            or set(retrieved_by_id) != set(replay_turn_ids)
        ):
            raise ValueError("P1 frozen RAG order diverges from semantic retrieval set")
        retrieved = tuple(retrieved_by_id[turn_id] for turn_id in replay_turn_ids)
    blended = policy.blend(
        scope_key=scope_hash,
        session_id="relationship-p1-probe",
        messages=[{"role": "user", "content": current_input}],
        retrieved_turns=retrieved,
    )
    if not blended.blended or not blended.messages:
        raise RuntimeError("ref-harness RAG produced no blended context")
    system_messages = tuple(item["content"] for item in blended.messages if item["role"] == "system")
    if len(system_messages) != 1:
        raise RuntimeError("ref-harness must publish exactly one context prefix")
    return system_messages[0], tuple(entry.turn_id for entry in retrieved)


def _audit_context_truth_leakage(
    *,
    dataset: RelationshipTransferDataset,
    contexts: tuple[RelationshipP1Context, ...],
) -> None:
    sealed_tokens = (
        {dynamic.dynamic_id for dynamic in dataset.dynamics}
        | {dynamic.mirror_pair_id for dynamic in dataset.dynamics}
        | {dynamic.outcome_profile_id for dynamic in dataset.dynamics}
        | {condition.condition_id for condition in dataset.abstract_conditions}
        | {policy.policy_id for policy in dataset.policy_profiles}
    )
    forbidden_keys = (
        "preferred_action",
        "sealed_latent_dynamic_id",
        "future_outcome",
        "generator_truth",
        "condition_id",
        "policy_id",
        "probe_condition_id",
        "history_condition_bindings",
    )
    for context in contexts:
        encoded = context.context_text
        leaked = sorted(token for token in sealed_tokens if token in encoded)
        if leaked:
            raise ValueError(f"P1 context leaks sealed ids: {leaked}")
        if any(key in encoded for key in forbidden_keys):
            raise ValueError("P1 context leaks a forbidden truth key")


def build_relationship_p1_context_bundle(
    *,
    state_root: pathlib.Path,
    rag_embedder: Embedder,
    dataset: RelationshipTransferDataset | None = None,
    background_template_package_name: str | None = None,
    background_depths: tuple[int, ...] = RELATIONSHIP_P1_DEFAULT_DEPTHS,
    rag_top_k: int = RELATIONSHIP_P1_RAG_TOP_K,
    rag_candidate_surface: RelationshipP1RagCandidateSurface = (
        RelationshipP1RagCandidateSurface.ALL_PUBLIC_RECORDS
    ),
    rag_replay_orders: tuple[RelationshipP1RagReplayOrder, ...] = (),
) -> RelationshipP1ContextBundle:
    """Write public histories, recover them, and build four matched contexts."""

    effective_dataset = dataset or load_relationship_transfer_dataset()
    if not background_depths or background_depths != tuple(sorted(set(background_depths))) or background_depths[0] != 0:
        raise ValueError("background_depths must be sorted, unique, and start at 0")
    if background_depths[-1] < 8:
        raise ValueError("P1 scaling requires a maximum background depth >= 8")
    if rag_top_k <= 0:
        raise ValueError("rag_top_k must be positive")
    if not isinstance(rag_candidate_surface, RelationshipP1RagCandidateSurface):
        raise ValueError("rag_candidate_surface must be typed")
    replay_order_map = {
        (item.scene_id, item.background_depth): item.turn_ids
        for item in rag_replay_orders
    }
    if len(replay_order_map) != len(rag_replay_orders):
        raise ValueError("P1 RAG replay orders must have unique scene/depth keys")
    if rag_replay_orders:
        expected_replay_keys = {
            (observation.scene_id, depth)
            for observation in effective_dataset.observations
            for depth in background_depths
        }
        if set(replay_order_map) != expected_replay_keys:
            raise ValueError("P1 RAG replay orders must cover the complete context surface")
    history_counts = {len(item.histories) for item in effective_dataset.observations}
    if len(history_counts) != 1:
        raise ValueError("P1 dataset users must expose one fixed relationship-history count")
    relationship_history_count = next(iter(history_counts))
    if relationship_history_count not in {2, 4}:
        raise ValueError("P1 dataset must expose two v1 or four v2 histories per user")
    if (
        rag_candidate_surface is RelationshipP1RagCandidateSurface.RELATIONSHIP_OUTCOMES_ONLY
        and rag_top_k < relationship_history_count
    ):
        raise ValueError("relationship-outcome RAG must admit every signal record")
    _state_root_must_be_fresh(state_root)
    template_package_name = (
        effective_dataset.package_name
        if background_template_package_name is None
        else background_template_package_name
    )
    if not template_package_name.strip():
        raise ValueError("background_template_package_name must be non-empty")
    templates = _load_background_templates(template_package_name)
    max_depth = background_depths[-1]
    memory_root = pathlib.Path(state_root) / "memory_store"
    rag_root = pathlib.Path(state_root) / "ref_harness"
    rag_root.mkdir(parents=True, exist_ok=True)
    rag_path = rag_root / "relationship-p1-rag.sqlite3"
    rag_store = open_store(StoreMode.SQLITE, sqlite_path=rag_path)

    max_items_by_scope: dict[str, tuple[PublicRelationshipContextItem, ...]] = {}
    for observation in effective_dataset.observations:
        items = _context_items(
            observation,
            background_depth=max_depth,
            templates=templates,
        )
        max_items_by_scope[observation.user_scope_hash] = items
        structured = StructuredRelationshipStateStore(
            root=memory_root,
            scope_hash=observation.user_scope_hash,
            load_existing=False,
        )
        structured.append_items(items)
        structured.persist()
        for item in items:
            rag_store.embed_index_put(
                EmbedEntry(
                    scope_key=observation.user_scope_hash,
                    turn_id=item.item_id,
                    role="memory",
                    content=item.content,
                    embedding=rag_embedder.embed(item.content),
                    ts=item.timestamp_iso,
                )
            )
    rag_store.close()

    persisted_state = probe_relationship_p1_persisted_state(
        state_root=state_root,
        dataset=effective_dataset,
    )
    policy = HarnessPolicy(ComponentSet(frozenset({HarnessComponent.EMBED})))
    rag_store = open_store(StoreMode.SQLITE, sqlite_path=rag_path)
    contexts: list[RelationshipP1Context] = []
    for observation in effective_dataset.observations:
        structured = StructuredRelationshipStateStore(
            root=memory_root,
            scope_hash=observation.user_scope_hash,
            load_existing=True,
        )
        structured_text, structured_refs = structured.render_context(
            expected_records=len(observation.histories)
        )
        all_rag_entries = rag_store.embed_index_list_for_scope(scope_key=observation.user_scope_hash)
        query_embedding = rag_embedder.embed(_RAG_QUERY_PREFIX + observation.current_input)
        for depth in background_depths:
            items_at_depth = _context_items(
                observation,
                background_depth=depth,
                templates=templates,
            )
            allowed_ids = {item.item_id for item in items_at_depth}
            rag_entries = tuple(entry for entry in all_rag_entries if entry.turn_id in allowed_ids)
            if (
                rag_candidate_surface
                is RelationshipP1RagCandidateSurface.RELATIONSHIP_OUTCOMES_ONLY
            ):
                signal_ids = {item.event_id for item in observation.histories}
                rag_entries = tuple(
                    entry for entry in rag_entries if entry.turn_id in signal_ids
                )
            full_text = "\n\n".join(item.content for item in items_at_depth)
            rag_text, rag_refs = _rag_context_from_entries(
                policy=policy,
                scope_hash=observation.user_scope_hash,
                current_input=observation.current_input,
                entries=rag_entries,
                query_embedding=query_embedding,
                top_k_count=rag_top_k,
                replay_turn_ids=replay_order_map.get(
                    (observation.scene_id, depth)
                ),
            )
            if (
                rag_candidate_surface
                is RelationshipP1RagCandidateSurface.RELATIONSHIP_OUTCOMES_ONLY
                and set(rag_refs) != {item.event_id for item in observation.histories}
            ):
                raise RuntimeError("strong RAG did not publish every relationship outcome")
            contexts.extend(
                (
                    RelationshipP1Context(
                        arm=RelationshipP1Arm.STATELESS,
                        scene_id=observation.scene_id,
                        background_depth=depth,
                        context_text="",
                        source_evidence_refs=(),
                    ),
                    RelationshipP1Context(
                        arm=RelationshipP1Arm.PROMPT_STEELMAN,
                        scene_id=observation.scene_id,
                        background_depth=depth,
                        context_text=full_text,
                        source_evidence_refs=tuple(item.item_id for item in items_at_depth),
                    ),
                    RelationshipP1Context(
                        arm=RelationshipP1Arm.RAG_STEELMAN,
                        scene_id=observation.scene_id,
                        background_depth=depth,
                        context_text=rag_text,
                        source_evidence_refs=rag_refs,
                    ),
                    RelationshipP1Context(
                        arm=RelationshipP1Arm.STRUCTURED_STATE,
                        scene_id=observation.scene_id,
                        background_depth=depth,
                        context_text=structured_text,
                        source_evidence_refs=structured_refs,
                    ),
                )
            )
    rag_store.close()
    frozen_contexts = tuple(contexts)
    _audit_context_truth_leakage(
        dataset=effective_dataset,
        contexts=frozen_contexts,
    )
    rag_config: dict[str, object] = {
        "wrapper": "companion-ref-harness",
        "components": [HarnessComponent.EMBED.value],
        "embedder": rag_embedder.name,
        "embedding_dim": rag_embedder.dim,
        "top_k": rag_top_k,
        "minimum_score": RELATIONSHIP_P1_RAG_MIN_SCORE,
        "query_prefix_sha256": hashlib.sha256(
            _RAG_QUERY_PREFIX.encode("utf-8")
        ).hexdigest(),
    }
    if rag_candidate_surface is not RelationshipP1RagCandidateSurface.ALL_PUBLIC_RECORDS:
        rag_config["candidate_surface"] = rag_candidate_surface.value
    rag_config_sha256 = sha256_json(rag_config)
    return RelationshipP1ContextBundle(
        dataset_fingerprint=effective_dataset.dataset_fingerprint,
        background_depths=background_depths,
        background_templates_sha256=_sha256_file(
            relationship_p1_background_template_path(template_package_name)
        ),
        rag_config_sha256=rag_config_sha256,
        contexts=frozen_contexts,
        persisted_state=persisted_state,
    )


def probe_relationship_p1_persisted_state(
    *,
    state_root: pathlib.Path,
    dataset: RelationshipTransferDataset | None = None,
) -> PersistedRelationshipP1StateDigest:
    """Open brand-new owner instances and hash their public persisted records."""

    effective_dataset = dataset or load_relationship_transfer_dataset()
    root = pathlib.Path(state_root)
    structured_rows: list[tuple[str, str, int]] = []
    rag_rows: list[tuple[str, str, int]] = []
    rag_path = root / "ref_harness" / "relationship-p1-rag.sqlite3"
    rag_store = open_store(StoreMode.SQLITE, sqlite_path=rag_path)
    for observation in effective_dataset.observations:
        structured = StructuredRelationshipStateStore(
            root=root / "memory_store",
            scope_hash=observation.user_scope_hash,
            load_existing=True,
        )
        entries = structured.entries()
        structured_rows.append(
            (
                observation.user_scope_hash,
                structured.state_digest(),
                len(entries),
            )
        )
        rag_entries = rag_store.embed_index_list_for_scope(scope_key=observation.user_scope_hash)
        rag_rows.append(
            (
                observation.user_scope_hash,
                sha256_json(tuple(_rag_entry_payload(item) for item in rag_entries)),
                len(rag_entries),
            )
        )
    rag_store.close()
    return PersistedRelationshipP1StateDigest(
        structured_scope_digests=tuple(sorted(structured_rows)),
        rag_scope_digests=tuple(sorted(rag_rows)),
    )


@dataclass(frozen=True)
class RelationshipP1ConsoleControlEvidence:
    rewrite_persisted: bool
    delete_persisted: bool
    sibling_scope_unchanged: bool
    original_entry_sha256: str
    rewritten_entry_sha256: str
    final_target_state_sha256: str
    sibling_state_sha256: str
    schema_version: str = RELATIONSHIP_P1_CONSOLE_PROBE_SCHEMA_VERSION

    @property
    def passed(self) -> bool:
        return self.rewrite_persisted and self.delete_persisted and self.sibling_scope_unchanged

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "rewrite_persisted": self.rewrite_persisted,
            "delete_persisted": self.delete_persisted,
            "sibling_scope_unchanged": self.sibling_scope_unchanged,
            "original_entry_sha256": self.original_entry_sha256,
            "rewritten_entry_sha256": self.rewritten_entry_sha256,
            "final_target_state_sha256": self.final_target_state_sha256,
            "sibling_state_sha256": self.sibling_state_sha256,
            "passed": self.passed,
        }


def run_relationship_p1_console_control_probe(
    *,
    root: pathlib.Path,
    dataset: RelationshipTransferDataset | None = None,
) -> RelationshipP1ConsoleControlEvidence:
    """Exercise owner-side rewrite/delete without feeding either into learning."""

    effective_dataset = dataset or load_relationship_transfer_dataset()
    pair = effective_dataset.mirrored_pairs()[0][1]
    target_observation = pair[0][0]
    sibling_observation = pair[1][0]
    templates = _load_background_templates(effective_dataset.package_name)
    control_root = pathlib.Path(root)
    _state_root_must_be_fresh(control_root)
    for observation in (target_observation, sibling_observation):
        store = StructuredRelationshipStateStore(
            root=control_root,
            scope_hash=observation.user_scope_hash,
            load_existing=False,
        )
        store.append_items(
            _context_items(
                observation,
                background_depth=0,
                templates=templates,
            )
        )
        store.persist()

    target = StructuredRelationshipStateStore(
        root=control_root,
        scope_hash=target_observation.user_scope_hash,
        load_existing=True,
    )
    sibling = StructuredRelationshipStateStore(
        root=control_root,
        scope_hash=sibling_observation.user_scope_hash,
        load_existing=True,
    )
    sibling_before = sibling.state_digest()
    original_event = target_observation.histories[0]
    original_entry = target._entry_for_event(original_event.event_id)
    original_hash = hashlib.sha256(original_entry.content.encode("utf-8")).hexdigest()
    replacement_event = replace(
        original_event,
        user_reaction=(original_event.user_reaction + " 用户更正：这条反应描述以当前版本为准。"),
    )
    rewritten = target.correct_event(
        original_event_id=original_event.event_id,
        replacement_event=replacement_event,
        timestamp_ms=1_800_000_200_000,
    )
    target.persist()
    rewritten_hash = hashlib.sha256(rewritten.content.encode("utf-8")).hexdigest()

    target_after_rewrite = StructuredRelationshipStateStore(
        root=control_root,
        scope_hash=target_observation.user_scope_hash,
        load_existing=True,
    )
    recovered_rewrite = target_after_rewrite._entry_for_event(original_event.event_id)
    rewrite_persisted = (
        recovered_rewrite.content == rewritten.content and original_entry.content != recovered_rewrite.content
    )
    target_after_rewrite.delete_event(original_event.event_id)
    target_after_rewrite.persist()
    target_after_delete = StructuredRelationshipStateStore(
        root=control_root,
        scope_hash=target_observation.user_scope_hash,
        load_existing=True,
    )
    delete_persisted = not any(
        f"event:{original_event.event_id}" in entry.tags for entry in target_after_delete.entries()
    )
    sibling_after = StructuredRelationshipStateStore(
        root=control_root,
        scope_hash=sibling_observation.user_scope_hash,
        load_existing=True,
    )
    sibling_after_digest = sibling_after.state_digest()
    return RelationshipP1ConsoleControlEvidence(
        rewrite_persisted=rewrite_persisted,
        delete_persisted=delete_persisted,
        sibling_scope_unchanged=sibling_before == sibling_after_digest,
        original_entry_sha256=original_hash,
        rewritten_entry_sha256=rewritten_hash,
        final_target_state_sha256=target_after_delete.state_digest(),
        sibling_state_sha256=sibling_after_digest,
    )


def relationship_p1_structural_metrics(
    *,
    bundle: RelationshipP1ContextBundle,
    dataset: RelationshipTransferDataset | None = None,
) -> dict[str, object]:
    effective_dataset = dataset or load_relationship_transfer_dataset()
    pair_count = 0
    evaluated_context_pairs = 0
    current_inputs_identical = True
    contextual_histories_distinct = True
    scope_isolation = True
    for _pair_id, members in effective_dataset.mirrored_pairs():
        pair_count += 1
        left, right = members
        current_inputs_identical = current_inputs_identical and (
            left[0].current_input.encode("utf-8") == right[0].current_input.encode("utf-8")
        )
        if left[1].split in {
            RelationshipDatasetSplit.TRAIN,
            RelationshipDatasetSplit.VALIDATION,
        }:
            evaluated_context_pairs += 1
            for arm in (
                RelationshipP1Arm.PROMPT_STEELMAN,
                RelationshipP1Arm.RAG_STEELMAN,
                RelationshipP1Arm.STRUCTURED_STATE,
            ):
                left_context = bundle.context(scene_id=left[0].scene_id, arm=arm)
                right_context = bundle.context(scene_id=right[0].scene_id, arm=arm)
                contextual_histories_distinct = contextual_histories_distinct and (
                    left_context.context_sha256 != right_context.context_sha256
                )
        left_signals = {_signal_content(item) for item in left[0].histories}
        right_signals = {_signal_content(item) for item in right[0].histories}
        left_structured = bundle.context(
            scene_id=left[0].scene_id,
            arm=RelationshipP1Arm.STRUCTURED_STATE,
        ).context_text
        right_structured = bundle.context(
            scene_id=right[0].scene_id,
            arm=RelationshipP1Arm.STRUCTURED_STATE,
        ).context_text
        scope_isolation = (
            scope_isolation
            and all(signal not in right_structured for signal in left_signals)
            and all(signal not in left_structured for signal in right_signals)
        )
    return {
        "mirrored_pairs": pair_count,
        "evaluated_context_pairs": evaluated_context_pairs,
        "current_inputs_byte_identical": current_inputs_identical,
        "contextual_histories_distinct": contextual_histories_distinct,
        "scope_isolation": scope_isolation,
        "passed": (
            pair_count >= 6
            and evaluated_context_pairs > 0
            and current_inputs_identical
            and contextual_histories_distinct
            and scope_isolation
        ),
    }


def relationship_p1_evaluated_context_surface_sha256(
    *,
    bundle: RelationshipP1ContextBundle,
    dataset: RelationshipTransferDataset | None = None,
) -> str:
    """Hash stable model-input bytes for the development-evaluated splits."""

    effective_dataset = dataset or load_relationship_transfer_dataset()
    evaluated_scene_ids = {
        observation.scene_id
        for observation in effective_dataset.observations
        if effective_dataset.dynamic_for_scene(observation.scene_id).split
        in {
            RelationshipDatasetSplit.TRAIN,
            RelationshipDatasetSplit.VALIDATION,
        }
    }
    rows = tuple(
        sorted(
            (
                context.scene_id,
                context.background_depth,
                context.arm.value,
                context.context_sha256,
            )
            for context in bundle.contexts
            if context.scene_id in evaluated_scene_ids
        )
    )
    if not rows:
        raise ValueError("P1 evaluated context surface must be non-empty")
    return sha256_json(
        {
            "dataset_fingerprint": bundle.dataset_fingerprint,
            "background_depths": list(bundle.background_depths),
            "background_templates_sha256": bundle.background_templates_sha256,
            "rag_config_sha256": bundle.rag_config_sha256,
            "evaluated_splits": [
                RelationshipDatasetSplit.TRAIN.value,
                RelationshipDatasetSplit.VALIDATION.value,
            ],
            "contexts": rows,
        }
    )


__all__ = [
    "RELATIONSHIP_P1_ARMS",
    "RELATIONSHIP_P1_DEFAULT_DEPTHS",
    "PersistedRelationshipP1StateDigest",
    "RelationshipP1Arm",
    "RelationshipP1RagCandidateSurface",
    "RelationshipP1RagReplayOrder",
    "RelationshipP1ConsoleControlEvidence",
    "RelationshipP1Context",
    "RelationshipP1ContextBundle",
    "RelationshipP1ContextReplayManifest",
    "StructuredRelationshipStateStore",
    "build_relationship_p1_context_bundle",
    "load_relationship_p1_context_replay_manifest",
    "probe_relationship_p1_persisted_state",
    "relationship_p1_evaluated_context_surface_sha256",
    "relationship_p1_background_template_path",
    "relationship_p1_structural_metrics",
    "run_relationship_p1_console_control_probe",
]
