"""Validate and analyse Gate 8/11 human-anchor pilot ratings.

This module is evaluation-only.  It binds the blinded packet, internal key,
rating rows, and L4-A preregistration before computing reliability, effect
estimates, and a frozen formal-sample recommendation.  Human ratings remain
readouts and are never exposed as reward, credit, or runtime state.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
import random
from statistics import NormalDist, fmean, stdev
from typing import Mapping, Sequence


GATE811_ANALYSIS_PREREG_SCHEMA_VERSION = (
    "gate811-human-anchor-analysis-prereg.v1"
)
GATE811_PILOT_ANALYSIS_SCHEMA_VERSION = (
    "gate811-human-anchor-pilot-analysis.v1"
)
GATE811_RATER_ROSTER_SCHEMA_VERSION = "gate811-human-anchor-rater-roster.v1"
GATE811_ANALYSIS_CODE_PATHS = (
    (
        "packages/vz-runtime/src/volvence_zero/agent/"
        "gate811_human_anchor_analysis.py"
    ),
    "scripts/preregister_gate811_human_anchor_analysis.py",
    "scripts/analyze_gate811_human_anchor_pilot.py",
)
GATE811_ANALYSIS_BOOTSTRAP_REPLICATES = 10_000
GATE811_ANALYSIS_BOOTSTRAP_SEED = 1549

_DIMENSIONS = (
    "rememberedness",
    "relationship_continuity",
    "boundary_respect",
)
_RATING_COLUMNS = (
    "rater_slot",
    "rater_id",
    "pair_id",
    "a_rememberedness",
    "b_rememberedness",
    "a_relationship_continuity",
    "b_relationship_continuity",
    "a_boundary_respect",
    "b_boundary_respect",
    "forced_preference",
    "malformed_reason",
)


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{field} must be a SHA-256 digest") from exc
    return value


def _require_mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _require_positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def build_gate811_analysis_preregistration(
    *,
    repo_root: str | Path,
    human_anchor_preregistration_path: str | Path,
    created_at_unix_ms: int,
) -> dict[str, object]:
    """Freeze analysis implementation details before pilot ratings exist."""

    _require_positive_int(created_at_unix_ms, field="created_at_unix_ms")
    root = Path(repo_root)
    source_path = Path(human_anchor_preregistration_path)
    absolute_source = source_path
    if not source_path.is_absolute():
        absolute_source = root / source_path
    source = json.loads(absolute_source.read_text(encoding="utf-8"))
    if source.get("schema_version") != "gate811-human-anchor-prereg.v1":
        raise ValueError("human-anchor preregistration schema drift")
    code_manifest = {
        relative: _sha256_file(root / relative)
        for relative in GATE811_ANALYSIS_CODE_PATHS
    }
    return {
        "schema_version": GATE811_ANALYSIS_PREREG_SCHEMA_VERSION,
        "created_at_unix_ms": created_at_unix_ms,
        "source_preregistration": {
            "path": str(source_path),
            "sha256": _sha256_file(absolute_source),
        },
        "code_manifest": code_manifest,
        "code_tree_sha256": _sha256_bytes(_canonical_bytes(code_manifest)),
        "rating_validation": {
            "exact_packet_manifest_hashes_required": True,
            "exact_three_distinct_raters_per_pair_required": True,
            "minimum_unique_non_project_raters": source["pilot"][
                "minimum_unique_raters"
            ],
            "typed_human_non_project_roster_attestation_required": True,
            "integer_likert_range": [
                source["rating"]["scale_min"],
                source["rating"]["scale_max"],
            ],
            "forced_preference_values": ["A", "B", "MALFORMED"],
            "malformed_requires_reason": True,
            "any_malformed_row_blocks_power_freeze": True,
        },
        "reliability": {
            "method": "krippendorff-alpha-ordinal-pooled-side-dimension",
            "minimum": source["pilot"][
                "minimum_krippendorff_alpha_ordinal"
            ],
            "unit": "pair-side-dimension",
        },
        "uncertainty": {
            "preference_interval": "two-sided-95%-wilson",
            "likert_interval": "rater-cluster-bootstrap-percentile-95%",
            "bootstrap_replicates": GATE811_ANALYSIS_BOOTSTRAP_REPLICATES,
            "bootstrap_seed": GATE811_ANALYSIS_BOOTSTRAP_SEED,
            "cluster": "rater_id",
        },
        "power": {
            "unit": "matched-pair-mean-over-three-raters",
            "target": source["formal"]["target_power"],
            "familywise_alpha": source["formal"]["familywise_alpha"],
            "planning_alpha": "bonferroni-two-sided-two-contrasts",
            "preference_method": (
                "normal-approximation-one-sample-proportion-vs-0.5"
            ),
            "likert_method": "normal-approximation-paired-mean",
            "rounding": "ceil-max-across-preference-composite-boundary",
            "minimum_pairs_per_contrast": source["formal"][
                "minimum_pairs_per_contrast"
            ],
            "maximum_pairs_per_contrast": source["formal"][
                "maximum_pairs_per_contrast"
            ],
        },
        "authorization": {
            "pilot_only": True,
            "human_anchor_claim_allowed": False,
            "rating_may_enter_reward_or_credit": False,
            "production_promotion_authorized": False,
            "formal_capture_requires_successful_pilot_analysis": True,
        },
    }


def validate_gate811_analysis_preregistration(
    payload: Mapping[str, object],
    *,
    repo_root: str | Path,
) -> None:
    """Fail if the frozen source or analysis implementation has drifted."""

    created_at = _require_positive_int(
        payload.get("created_at_unix_ms"), field="created_at_unix_ms"
    )
    source = _require_mapping(
        payload.get("source_preregistration"),
        field="source_preregistration",
    )
    source_path = source.get("path")
    if not isinstance(source_path, str) or not source_path:
        raise ValueError("source_preregistration.path must be non-empty")
    expected = build_gate811_analysis_preregistration(
        repo_root=repo_root,
        human_anchor_preregistration_path=source_path,
        created_at_unix_ms=created_at,
    )
    if dict(payload) != expected:
        raise ValueError("Gate 8/11 analysis preregistration drift")


def write_gate811_analysis_preregistration(
    *,
    payload: Mapping[str, object],
    output_path: str | Path,
) -> dict[str, object]:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = _canonical_bytes(dict(payload))
    output.write_bytes(serialized)
    manifest = {
        "schema_version": GATE811_ANALYSIS_PREREG_SCHEMA_VERSION,
        "preregistration_path": str(output),
        "preregistration_sha256": _sha256_bytes(serialized),
        "code_tree_sha256": payload["code_tree_sha256"],
        "production_promotion_authorized": False,
    }
    manifest_path = output.with_name(f"{output.stem}.manifest.json")
    manifest_path.write_bytes(_canonical_bytes(manifest))
    return manifest


def _validate_packet_bundle(
    *,
    packet: Mapping[str, object],
    internal_key: Mapping[str, object],
    packet_manifest: Mapping[str, object],
    packet_bytes: bytes,
    internal_key_bytes: bytes,
    rating_template_bytes: bytes,
    human_anchor_preregistration_sha256: str,
) -> tuple[dict[str, Mapping[str, object]], int]:
    if packet.get("schema_version") != "gate811-human-anchor-packet.v1":
        raise ValueError("blinded packet schema drift")
    if internal_key.get("schema_version") != packet.get("schema_version"):
        raise ValueError("internal key schema drift")
    for payload in (packet, internal_key):
        if payload.get("preregistration_sha256") != (
            human_anchor_preregistration_sha256
        ):
            raise ValueError("packet preregistration binding drift")
    if packet.get("pilot_only") is not True:
        raise ValueError("analysis accepts pilot packets only")
    if packet.get("human_anchor_claim_allowed") is not False:
        raise ValueError("pilot packet claim authorization drift")
    if internal_key.get("do_not_distribute_to_raters") is not True:
        raise ValueError("internal key distribution guard drift")
    hashes = _require_mapping(packet_manifest.get("sha256"), field="sha256")
    observed_hashes = {
        "pilot_packet_blinded.json": _sha256_bytes(packet_bytes),
        "pilot_key_internal.json": _sha256_bytes(internal_key_bytes),
        "pilot_rating_template.csv": _sha256_bytes(rating_template_bytes),
    }
    if dict(hashes) != observed_hashes:
        raise ValueError("pilot packet manifest hash drift")
    pairs = packet.get("pairs")
    entries = internal_key.get("entries")
    if not isinstance(pairs, list) or not isinstance(entries, list):
        raise ValueError("pilot packet pairs or key entries missing")
    pair_ids = [pair.get("pair_id") for pair in pairs if isinstance(pair, Mapping)]
    if len(pair_ids) != len(pairs) or len(set(pair_ids)) != len(pair_ids):
        raise ValueError("pilot packet pair_id drift")
    key_by_pair = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("internal key entry must be an object")
        pair_id = entry.get("pair_id")
        if pair_id in key_by_pair:
            raise ValueError("duplicate internal key pair_id")
        key_by_pair[pair_id] = entry
    if set(pair_ids) != set(key_by_pair):
        raise ValueError("packet and internal key pair coverage drift")
    return key_by_pair, len(pair_ids)


def _validate_rating_template_layout(
    *,
    rating_template_csv: str,
    rating_csv: str,
) -> None:
    template_reader = csv.DictReader(io.StringIO(rating_template_csv))
    rating_reader = csv.DictReader(io.StringIO(rating_csv))
    if tuple(template_reader.fieldnames or ()) != _RATING_COLUMNS:
        raise ValueError("rating template CSV columns drift")
    if tuple(rating_reader.fieldnames or ()) != _RATING_COLUMNS:
        raise ValueError("rating CSV columns drift")
    expected = [
        (row["rater_slot"], row["pair_id"]) for row in template_reader
    ]
    observed = [
        (row["rater_slot"], row["pair_id"]) for row in rating_reader
    ]
    if observed != expected:
        raise ValueError("completed rating rows drift from frozen template")


def _parse_likert(value: str, *, field: str, scale_min: int, scale_max: int) -> int:
    try:
        score = int(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be an integer Likert score") from exc
    if str(score) != value.strip() or not scale_min <= score <= scale_max:
        raise ValueError(f"{field} must be within the frozen Likert range")
    return score


def _parse_rating_rows(
    *,
    rating_csv: str,
    key_by_pair: Mapping[str, Mapping[str, object]],
    human_anchor_preregistration: Mapping[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    reader = csv.DictReader(io.StringIO(rating_csv))
    if tuple(reader.fieldnames or ()) != _RATING_COLUMNS:
        raise ValueError("rating CSV columns drift")
    rating = _require_mapping(
        human_anchor_preregistration.get("rating"), field="rating"
    )
    scale_min = _require_positive_int(rating.get("scale_min"), field="scale_min")
    scale_max = _require_positive_int(rating.get("scale_max"), field="scale_max")
    rows = []
    malformed = []
    seen_pair_raters = set()
    for row_index, raw in enumerate(reader, start=2):
        pair_id = raw["pair_id"].strip()
        rater_slot = raw["rater_slot"].strip()
        rater_id = raw["rater_id"].strip()
        if pair_id not in key_by_pair:
            raise ValueError(f"rating row {row_index} has unknown pair_id")
        if not rater_slot or not rater_id:
            raise ValueError(f"rating row {row_index} lacks rater identity")
        identity = (pair_id, rater_id)
        if identity in seen_pair_raters:
            raise ValueError("same rater may not rate a pair twice")
        seen_pair_raters.add(identity)
        preference = raw["forced_preference"].strip().upper()
        malformed_reason = raw["malformed_reason"].strip()
        if preference == "MALFORMED":
            if not malformed_reason:
                raise ValueError("MALFORMED preference requires a reason")
            malformed.append(
                {
                    "pair_id": pair_id,
                    "rater_id": rater_id,
                    "reason": malformed_reason,
                }
            )
            continue
        if preference not in ("A", "B"):
            raise ValueError("forced_preference must be A, B, or MALFORMED")
        if malformed_reason:
            raise ValueError("valid preference may not carry malformed_reason")
        scores = {}
        for side in ("a", "b"):
            for dimension in _DIMENSIONS:
                field = f"{side}_{dimension}"
                scores[field] = _parse_likert(
                    raw[field].strip(),
                    field=field,
                    scale_min=scale_min,
                    scale_max=scale_max,
                )
        key = key_by_pair[pair_id]
        experimental_side = None
        if key.get("side_a_arm") in (
            "correct-user-state",
            "sleep-consolidation",
        ):
            experimental_side = "A"
        elif key.get("side_b_arm") in (
            "correct-user-state",
            "sleep-consolidation",
        ):
            experimental_side = "B"
        if experimental_side is None:
            raise ValueError("internal key lacks registered experimental arm")
        differences = {}
        for dimension in _DIMENSIONS:
            a_score = scores[f"a_{dimension}"]
            b_score = scores[f"b_{dimension}"]
            differences[dimension] = (
                a_score - b_score
                if experimental_side == "A"
                else b_score - a_score
            )
        rows.append(
            {
                "pair_id": pair_id,
                "contrast_id": key["contrast_id"],
                "rater_slot": rater_slot,
                "rater_id": rater_id,
                "preference_success": preference == experimental_side,
                "differences": differences,
                "side_scores": scores,
            }
        )
    return rows, malformed


def _validate_rating_coverage(
    *,
    rows: Sequence[Mapping[str, object]],
    malformed: Sequence[Mapping[str, str]],
    key_by_pair: Mapping[str, Mapping[str, object]],
    human_anchor_preregistration: Mapping[str, object],
) -> set[str]:
    pilot = _require_mapping(
        human_anchor_preregistration.get("pilot"), field="pilot"
    )
    ratings_per_pair = _require_positive_int(
        pilot.get("ratings_per_pair"), field="ratings_per_pair"
    )
    expected_rows = len(key_by_pair) * ratings_per_pair
    if len(rows) + len(malformed) != expected_rows:
        raise ValueError("rating row count does not match frozen pilot")
    if malformed:
        raise ValueError("malformed pilot rows block power freeze")
    by_pair: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        by_pair.setdefault(str(row["pair_id"]), []).append(row)
    if set(by_pair) != set(key_by_pair):
        raise ValueError("rating pair coverage drift")
    for pair_rows in by_pair.values():
        if len(pair_rows) != ratings_per_pair:
            raise ValueError("pair rating count drift")
        if len({row["rater_id"] for row in pair_rows}) != ratings_per_pair:
            raise ValueError("pair ratings must use distinct raters")
        if len({row["rater_slot"] for row in pair_rows}) != ratings_per_pair:
            raise ValueError("pair ratings must use distinct rater slots")
    raters = {str(row["rater_id"]) for row in rows}
    minimum_raters = _require_positive_int(
        pilot.get("minimum_unique_raters"), field="minimum_unique_raters"
    )
    if len(raters) < minimum_raters:
        raise ValueError("pilot has too few unique external raters")
    return raters


def _validate_rater_roster(
    *,
    roster: Mapping[str, object],
    raters: set[str],
    human_anchor_preregistration_sha256: str,
    analysis_preregistration_sha256: str,
) -> None:
    if roster.get("schema_version") != GATE811_RATER_ROSTER_SCHEMA_VERSION:
        raise ValueError("rater roster schema drift")
    if roster.get("human_anchor_preregistration_sha256") != (
        human_anchor_preregistration_sha256
    ):
        raise ValueError("rater roster human-anchor binding drift")
    if roster.get("analysis_preregistration_sha256") != (
        analysis_preregistration_sha256
    ):
        raise ValueError("rater roster analysis binding drift")
    entries = roster.get("entries")
    if not isinstance(entries, list):
        raise ValueError("rater roster entries must be a list")
    roster_ids = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("rater roster entry must be an object")
        rater_id = entry.get("rater_id")
        if not isinstance(rater_id, str) or not rater_id.strip():
            raise ValueError("rater roster rater_id must be non-empty")
        if rater_id in roster_ids:
            raise ValueError("duplicate rater roster rater_id")
        roster_ids.add(rater_id)
        if entry.get("human_rater_attested") is not True:
            raise ValueError("human_rater_attested must be true")
        if entry.get("non_project_member_attested") is not True:
            raise ValueError("non_project_member_attested must be true")
        _require_sha256(
            entry.get("eligibility_review_artifact_sha256"),
            field="eligibility_review_artifact_sha256",
        )
        attested_by = entry.get("attested_by")
        if not isinstance(attested_by, str) or not attested_by.strip():
            raise ValueError("rater roster attested_by must be non-empty")
    if roster_ids != raters:
        raise ValueError("rater roster must exactly cover completed ratings")


def _wilson_interval(successes: int, count: int) -> tuple[float, float]:
    if count <= 0:
        raise ValueError("Wilson interval requires observations")
    z = NormalDist().inv_cdf(0.975)
    estimate = successes / count
    denominator = 1.0 + z * z / count
    center = (estimate + z * z / (2.0 * count)) / denominator
    radius = (
        z
        * math.sqrt(
            estimate * (1.0 - estimate) / count
            + z * z / (4.0 * count * count)
        )
        / denominator
    )
    return center - radius, center + radius


def _ordinal_alpha(
    rows: Sequence[Mapping[str, object]],
    *,
    scale_min: int,
    scale_max: int,
) -> float:
    units: dict[tuple[str, str, str], list[int]] = {}
    for row in rows:
        side_scores = row["side_scores"]
        assert isinstance(side_scores, Mapping)
        for side in ("a", "b"):
            for dimension in _DIMENSIONS:
                units.setdefault(
                    (str(row["pair_id"]), side, dimension), []
                ).append(int(side_scores[f"{side}_{dimension}"]))
    categories = tuple(range(scale_min, scale_max + 1))
    coincidence = {
        (left, right): 0.0 for left in categories for right in categories
    }
    for values in units.values():
        if len(values) < 2:
            continue
        weight = 1.0 / (len(values) - 1)
        for left_index, left in enumerate(values):
            for right_index, right in enumerate(values):
                if left_index != right_index:
                    coincidence[(left, right)] += weight
    marginals = {
        category: sum(coincidence[(category, other)] for other in categories)
        for category in categories
    }
    total = sum(marginals.values())
    if total <= 1.0:
        raise ValueError("ordinal alpha requires pairable ratings")

    def distance(left: int, right: int) -> float:
        lower, upper = sorted((left, right))
        cumulative = sum(marginals[value] for value in range(lower, upper + 1))
        cumulative -= (marginals[lower] + marginals[upper]) / 2.0
        return cumulative * cumulative

    observed = sum(
        coincidence[(left, right)] * distance(left, right)
        for left in categories
        for right in categories
    )
    expected = sum(
        marginals[left] * marginals[right] * distance(left, right)
        for left in categories
        for right in categories
    )
    if expected == 0.0:
        raise ValueError("ordinal alpha undefined for constant ratings")
    return 1.0 - observed * (total - 1.0) / expected


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _cluster_bootstrap_intervals(
    rows: Sequence[Mapping[str, object]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    by_rater: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        by_rater.setdefault(str(row["rater_id"]), []).append(row)
    rater_ids = sorted(by_rater)
    if len(rater_ids) < 2:
        raise ValueError("cluster bootstrap requires at least two raters")
    rng = random.Random(seed)
    composite_samples = []
    boundary_samples = []
    for _ in range(replicates):
        sampled_rows = []
        for _cluster in rater_ids:
            sampled_rows.extend(by_rater[rng.choice(rater_ids)])
        composites = []
        boundaries = []
        for row in sampled_rows:
            differences = row["differences"]
            assert isinstance(differences, Mapping)
            composites.append(
                fmean(float(differences[name]) for name in _DIMENSIONS)
            )
            boundaries.append(float(differences["boundary_respect"]))
        composite_samples.append(fmean(composites))
        boundary_samples.append(fmean(boundaries))
    return {
        "composite": (
            _percentile(composite_samples, 0.025),
            _percentile(composite_samples, 0.975),
        ),
        "boundary": (
            _percentile(boundary_samples, 0.025),
            _percentile(boundary_samples, 0.975),
        ),
    }


def _sample_size_for_proportion(
    *,
    alternative: float,
    null: float,
    z_alpha: float,
    z_power: float,
) -> int | None:
    effect = alternative - null
    if effect <= 0.0:
        return None
    numerator = (
        z_alpha * math.sqrt(null * (1.0 - null))
        + z_power * math.sqrt(alternative * (1.0 - alternative))
    ) ** 2
    return math.ceil(numerator / (effect * effect))


def _sample_size_for_mean(
    *,
    standard_deviation: float,
    effect: float,
    z_alpha: float,
    z_power: float,
) -> int | None:
    if effect <= 0.0:
        return None
    return math.ceil(
        ((z_alpha + z_power) * standard_deviation / effect) ** 2
    )


def _pair_level_values(
    rows: Sequence[Mapping[str, object]],
) -> tuple[list[float], list[float], list[float]]:
    by_pair: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        by_pair.setdefault(str(row["pair_id"]), []).append(row)
    preferences = []
    composites = []
    boundaries = []
    for pair_rows in by_pair.values():
        preference_count = sum(
            bool(row["preference_success"]) for row in pair_rows
        )
        preferences.append(float(preference_count > len(pair_rows) / 2.0))
        row_composites = []
        row_boundaries = []
        for row in pair_rows:
            differences = row["differences"]
            assert isinstance(differences, Mapping)
            row_composites.append(
                fmean(float(differences[name]) for name in _DIMENSIONS)
            )
            row_boundaries.append(float(differences["boundary_respect"]))
        composites.append(fmean(row_composites))
        boundaries.append(fmean(row_boundaries))
    return preferences, composites, boundaries


def _formal_pair_recommendation(
    *,
    rows: Sequence[Mapping[str, object]],
    formal: Mapping[str, object],
    contrast_count: int,
) -> dict[str, object]:
    preferences, composites, boundaries = _pair_level_values(rows)
    target_power = float(formal["target_power"])
    familywise_alpha = float(formal["familywise_alpha"])
    alpha_per_contrast = familywise_alpha / contrast_count
    z_alpha = NormalDist().inv_cdf(1.0 - alpha_per_contrast / 2.0)
    z_power = NormalDist().inv_cdf(target_power)
    observed_preference = fmean(preferences)
    planning_preference = max(
        observed_preference,
        float(formal["minimum_preference_win_rate"]),
    )
    preference_n = _sample_size_for_proportion(
        alternative=planning_preference,
        null=0.5,
        z_alpha=z_alpha,
        z_power=z_power,
    )
    composite_mean = fmean(composites)
    composite_effect = max(
        composite_mean,
        float(formal["minimum_composite_likert_delta"]),
    )
    composite_n = _sample_size_for_mean(
        standard_deviation=stdev(composites),
        effect=composite_effect,
        z_alpha=z_alpha,
        z_power=z_power,
    )
    boundary_mean = fmean(boundaries)
    boundary_margin = float(formal["boundary_noninferiority_margin"])
    boundary_n = _sample_size_for_mean(
        standard_deviation=stdev(boundaries),
        effect=boundary_mean - boundary_margin,
        z_alpha=z_alpha,
        z_power=z_power,
    )
    requirements = (preference_n, composite_n, boundary_n)
    minimum = int(formal["minimum_pairs_per_contrast"])
    maximum = int(formal["maximum_pairs_per_contrast"])
    raw_required = (
        max(requirement for requirement in requirements if requirement is not None)
        if all(requirement is not None for requirement in requirements)
        else None
    )
    feasible = raw_required is not None and raw_required <= maximum
    recommended = max(minimum, raw_required) if feasible else maximum
    return {
        "observed_pair_majority_preference_rate": observed_preference,
        "observed_pair_composite_mean": composite_mean,
        "observed_pair_composite_sd": stdev(composites),
        "observed_pair_boundary_mean": boundary_mean,
        "observed_pair_boundary_sd": stdev(boundaries),
        "planning_alpha_per_contrast_two_sided": alpha_per_contrast,
        "preference_required_pairs": preference_n,
        "composite_required_pairs": composite_n,
        "boundary_required_pairs": boundary_n,
        "raw_required_pairs": raw_required,
        "recommended_pairs": recommended,
        "within_frozen_60_to_300_range": feasible,
    }


def analyze_gate811_pilot_ratings(
    *,
    human_anchor_preregistration: Mapping[str, object],
    human_anchor_preregistration_bytes: bytes,
    human_anchor_preregistration_sha256: str,
    analysis_preregistration: Mapping[str, object],
    analysis_preregistration_bytes: bytes,
    analysis_preregistration_sha256: str,
    packet: Mapping[str, object],
    packet_bytes: bytes,
    internal_key: Mapping[str, object],
    internal_key_bytes: bytes,
    packet_manifest: Mapping[str, object],
    rating_template_csv: str,
    rating_csv: str,
    rater_roster: Mapping[str, object],
    rater_roster_bytes: bytes,
) -> dict[str, object]:
    """Validate a completed pilot and produce a non-claim power report."""

    _require_sha256(
        human_anchor_preregistration_sha256,
        field="human_anchor_preregistration_sha256",
    )
    _require_sha256(
        analysis_preregistration_sha256,
        field="analysis_preregistration_sha256",
    )
    if _sha256_bytes(human_anchor_preregistration_bytes) != (
        human_anchor_preregistration_sha256
    ):
        raise ValueError("human-anchor preregistration hash drift")
    if json.loads(human_anchor_preregistration_bytes) != dict(
        human_anchor_preregistration
    ):
        raise ValueError("human-anchor preregistration bytes drift")
    if _sha256_bytes(analysis_preregistration_bytes) != (
        analysis_preregistration_sha256
    ):
        raise ValueError("analysis preregistration hash drift")
    if json.loads(analysis_preregistration_bytes) != dict(
        analysis_preregistration
    ):
        raise ValueError("analysis preregistration bytes drift")
    analysis_source = _require_mapping(
        analysis_preregistration.get("source_preregistration"),
        field="source_preregistration",
    )
    if analysis_source.get("sha256") != human_anchor_preregistration_sha256:
        raise ValueError("analysis preregistration source binding drift")
    rating_bytes = rating_csv.encode("utf-8")
    rating_template_bytes = rating_template_csv.encode("utf-8")
    if json.loads(packet_bytes) != dict(packet):
        raise ValueError("blinded packet bytes drift")
    if json.loads(internal_key_bytes) != dict(internal_key):
        raise ValueError("internal key bytes drift")
    if json.loads(rater_roster_bytes) != dict(rater_roster):
        raise ValueError("rater roster bytes drift")
    key_by_pair, pair_count = _validate_packet_bundle(
        packet=packet,
        internal_key=internal_key,
        packet_manifest=packet_manifest,
        packet_bytes=packet_bytes,
        internal_key_bytes=internal_key_bytes,
        rating_template_bytes=rating_template_bytes,
        human_anchor_preregistration_sha256=(
            human_anchor_preregistration_sha256
        ),
    )
    _validate_rating_template_layout(
        rating_template_csv=rating_template_csv,
        rating_csv=rating_csv,
    )
    rows, malformed = _parse_rating_rows(
        rating_csv=rating_csv,
        key_by_pair=key_by_pair,
        human_anchor_preregistration=human_anchor_preregistration,
    )
    raters = _validate_rating_coverage(
        rows=rows,
        malformed=malformed,
        key_by_pair=key_by_pair,
        human_anchor_preregistration=human_anchor_preregistration,
    )
    _validate_rater_roster(
        roster=rater_roster,
        raters=raters,
        human_anchor_preregistration_sha256=(
            human_anchor_preregistration_sha256
        ),
        analysis_preregistration_sha256=analysis_preregistration_sha256,
    )
    rating = _require_mapping(
        human_anchor_preregistration.get("rating"), field="rating"
    )
    pilot = _require_mapping(
        human_anchor_preregistration.get("pilot"), field="pilot"
    )
    formal = _require_mapping(
        human_anchor_preregistration.get("formal"), field="formal"
    )
    scale_min = int(rating["scale_min"])
    scale_max = int(rating["scale_max"])
    contrast_ids = sorted({str(row["contrast_id"]) for row in rows})
    contrast_results = {}
    reliability_floor = float(pilot["minimum_krippendorff_alpha_ordinal"])
    uncertainty = _require_mapping(
        analysis_preregistration.get("uncertainty"), field="uncertainty"
    )
    replicates = _require_positive_int(
        uncertainty.get("bootstrap_replicates"),
        field="bootstrap_replicates",
    )
    base_seed = _require_positive_int(
        uncertainty.get("bootstrap_seed"), field="bootstrap_seed"
    )
    all_ready = True
    for contrast_index, contrast_id in enumerate(contrast_ids):
        contrast_rows = [
            row for row in rows if row["contrast_id"] == contrast_id
        ]
        successes = sum(
            bool(row["preference_success"]) for row in contrast_rows
        )
        preference_rate = successes / len(contrast_rows)
        wilson_lower, wilson_upper = _wilson_interval(
            successes, len(contrast_rows)
        )
        differences = [
            row["differences"] for row in contrast_rows
        ]
        composites = [
            fmean(float(value[name]) for name in _DIMENSIONS)
            for value in differences
            if isinstance(value, Mapping)
        ]
        boundaries = [
            float(value["boundary_respect"])
            for value in differences
            if isinstance(value, Mapping)
        ]
        intervals = _cluster_bootstrap_intervals(
            contrast_rows,
            replicates=replicates,
            seed=base_seed + contrast_index,
        )
        alpha = _ordinal_alpha(
            contrast_rows,
            scale_min=scale_min,
            scale_max=scale_max,
        )
        power = _formal_pair_recommendation(
            rows=contrast_rows,
            formal=formal,
            contrast_count=len(contrast_ids),
        )
        reliability_passed = alpha >= reliability_floor
        formal_ready = (
            reliability_passed
            and bool(power["within_frozen_60_to_300_range"])
        )
        all_ready = all_ready and formal_ready
        contrast_results[contrast_id] = {
            "pair_count": len({row["pair_id"] for row in contrast_rows}),
            "rating_count": len(contrast_rows),
            "preference": {
                "experimental_successes": successes,
                "win_rate": preference_rate,
                "wilson_95": [wilson_lower, wilson_upper],
                "formal_gate_evaluated": False,
            },
            "likert": {
                "composite_delta": fmean(composites),
                "composite_cluster_bootstrap_95": list(
                    intervals["composite"]
                ),
                "boundary_delta": fmean(boundaries),
                "boundary_cluster_bootstrap_95": list(
                    intervals["boundary"]
                ),
                "formal_gate_evaluated": False,
            },
            "ordinal_krippendorff_alpha": alpha,
            "reliability_floor": reliability_floor,
            "reliability_passed": reliability_passed,
            "power": power,
            "formal_capture_ready": formal_ready,
            "failure_action": (
                None
                if formal_ready
                else (
                    "freeze-pilot-and-revise-rubric-under-new-schema"
                    if not reliability_passed
                    else "pilot-effect-outside-frozen-formal-range"
                )
            ),
        }
    return {
        "schema_version": GATE811_PILOT_ANALYSIS_SCHEMA_VERSION,
        "human_anchor_preregistration_sha256": (
            human_anchor_preregistration_sha256
        ),
        "analysis_preregistration_sha256": (
            analysis_preregistration_sha256
        ),
        "packet_sha256": _sha256_bytes(packet_bytes),
        "internal_key_sha256": _sha256_bytes(internal_key_bytes),
        "ratings_sha256": _sha256_bytes(rating_bytes),
        "rater_roster_sha256": _sha256_bytes(rater_roster_bytes),
        "pilot_only": True,
        "pilot_rows_excluded_from_formal": True,
        "pair_count": pair_count,
        "rating_count": len(rows),
        "unique_rater_count": len(raters),
        "contrasts": contrast_results,
        "formal_capture_authorized": all_ready,
        "human_anchor_claim_allowed": False,
        "rating_may_enter_reward_or_credit": False,
        "production_promotion_authorized": False,
    }


def export_gate811_pilot_analysis(
    *,
    report: Mapping[str, object],
    output_path: str | Path,
) -> dict[str, object]:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = _canonical_bytes(dict(report))
    output.write_bytes(serialized)
    return {
        "schema_version": GATE811_PILOT_ANALYSIS_SCHEMA_VERSION,
        "report_path": str(output),
        "report_sha256": _sha256_bytes(serialized),
        "formal_capture_authorized": report["formal_capture_authorized"],
        "human_anchor_claim_allowed": False,
        "production_promotion_authorized": False,
    }


__all__ = [
    "GATE811_ANALYSIS_BOOTSTRAP_REPLICATES",
    "GATE811_ANALYSIS_BOOTSTRAP_SEED",
    "GATE811_ANALYSIS_CODE_PATHS",
    "GATE811_ANALYSIS_PREREG_SCHEMA_VERSION",
    "GATE811_PILOT_ANALYSIS_SCHEMA_VERSION",
    "GATE811_RATER_ROSTER_SCHEMA_VERSION",
    "analyze_gate811_pilot_ratings",
    "build_gate811_analysis_preregistration",
    "export_gate811_pilot_analysis",
    "validate_gate811_analysis_preregistration",
    "write_gate811_analysis_preregistration",
]
