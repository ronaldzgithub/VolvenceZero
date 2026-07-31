"""Prefix-KV carrier injection on the frozen transformers runtime.

The load-bearing test here is ``test_greedy_prefix_loop_reproduces_generate``.
The prefix carrier cannot use ``model.generate`` (a pre-filled cache makes it
truncate the prompt and shift every position), so it runs its own greedy loop.
That only makes arm G comparable to arms A/E/B-prime if the loop is decode-path
identical to ``generate`` when no prefix is supplied. If that equivalence ever
breaks, a divergence between arms would be attributable to the decoder rather
than to the carrier under test.
"""

from __future__ import annotations

from dataclasses import replace
import random

import pytest

from volvence_zero.conditioning_bank_contracts import (
    CONDITIONING_BANK_LATENT_CARRIER_SCHEMA_VERSION,
    CONDITIONING_BANK_SCHEMA_VERSION,
    ConditioningBankLatentCarrier,
    ConditioningBankSnapshot,
    ConditioningBankType,
    ConditioningRevocationState,
    ConditioningScope,
)
from volvence_zero.personal_conditioning_contracts import (
    PERSONAL_CONDITIONING_SCHEMA_VERSION,
    PERSONAL_CONDITIONING_VECTOR_LABELS,
    PersonalConditioningSnapshot,
)
from volvence_zero.substrate.prefix_kv_artifact import (
    build_teacher_distilled_prefix_artifact,
)
from volvence_zero.substrate.residual_backend import (
    _banned_repeated_ngram_tokens,
)
from volvence_zero.substrate.relationship_prefix_kv_artifact import (
    bind_relationship_prefix_artifact,
)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
SYSTEM = "你是一个助手。"
MESSAGES = (("system", SYSTEM), ("user", "我又搞砸了"))
PROMPT = "我又搞砸了"


def test_prefix_decode_bans_repeated_trigrams() -> None:
    assert _banned_repeated_ngram_tokens([1, 2, 3, 2, 3], ngram_size=3) == (2,)
    assert _banned_repeated_ngram_tokens([1, 2], ngram_size=3) == ()


def _local_weights() -> str:
    hub = pytest.importorskip("huggingface_hub")
    try:
        return hub.snapshot_download(MODEL_ID, local_files_only=True)
    except Exception as exc:  # noqa: BLE001 - any resolution failure skips
        pytest.skip(f"frozen {MODEL_ID} snapshot unavailable: {exc}")


@pytest.fixture(scope="module")
def runtime():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from volvence_zero.substrate import TransformersOpenWeightResidualRuntime

    return TransformersOpenWeightResidualRuntime(
        model_id=MODEL_ID,
        pretrained_source=_local_weights(),
        device="cpu",
        hook_layer_selection="middle",
        local_files_only=True,
        runtime_origin="hf-local",
    )


def _artifact(
    *,
    seed: int,
    slots: int = 2,
    rank: int = 2,
    vector_labels: tuple[str, ...] = PERSONAL_CONDITIONING_VECTOR_LABELS,
):
    """A bounded, geometry-correct artifact; content is not trained."""

    rng = random.Random(seed)
    layers, kv_heads, head_dim = 24, 2, 64
    width = slots * kv_heads * head_dim
    coordinates = len(vector_labels)

    def block() -> list[list[float]]:
        return [
            [rng.uniform(-1.0, 1.0) for _ in range(rank)] for _ in range(width)
        ]

    return build_teacher_distilled_prefix_artifact(
        model_id=MODEL_ID,
        num_layers=layers,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        num_slots=slots,
        bottleneck_rank=rank,
        encoder_rows=[
            [rng.uniform(-1.0, 1.0) for _ in range(coordinates)]
            for _ in range(rank)
        ],
        encoder_bias=[0.0] * rank,
        key_projection=[block() for _ in range(layers)],
        key_bias=[[0.0] * width for _ in range(layers)],
        value_projection=[block() for _ in range(layers)],
        value_bias=[[0.0] * width for _ in range(layers)],
        reference_key_norms=[20.0] * layers,
        reference_value_norms=[4.0] * layers,
        norm_cap=0.1,
        source_fingerprint=f"unit-test-seed-{seed}",
        sample_count=8,
        vector_labels=vector_labels,
    )


RELATIONSHIP_LABELS = (
    "rel_trust",
    "rel_repair_pressure",
    "rel_emotional_load",
)


def _relationship_artifact():
    return bind_relationship_prefix_artifact(
        prefix_artifact=_artifact(
            seed=19,
            vector_labels=RELATIONSHIP_LABELS,
        ),
        owner_schema_version="relationship-conditioning.v2",
        readout_labels=RELATIONSHIP_LABELS,
        description="Relationship runtime injection test artifact.",
    )


def _relationship_carrier(*, artifact, labels=RELATIONSHIP_LABELS):
    bank = ConditioningBankSnapshot(
        schema_version=CONDITIONING_BANK_SCHEMA_VERSION,
        bank_type=ConditioningBankType.RELATIONSHIP,
        scope=ConditioningScope(
            tenant_scope="tenant-test",
            user_scope="user-test",
            session_scope="session-test",
        ),
        readout=(0.8, 0.7, 0.6),
        readout_labels=labels,
        source_versions=(("relationship_state", 2),),
        source_fingerprint="relationship-prefix-runtime-test",
        confidence=0.75,
        freshness=0.8,
        consent_version=1,
        provenance="owner:RelationshipConditioningModule/test",
        revocation_state=ConditioningRevocationState.ACTIVE,
        is_cold_start=False,
        description="Relationship Prefix-KV runtime test bank.",
    )
    return ConditioningBankLatentCarrier(
        schema_version=CONDITIONING_BANK_LATENT_CARRIER_SCHEMA_VERSION,
        bank=bank,
        carrier="prefix_kv",
        projector_version=artifact.carrier_version,
        scale=artifact.prefix_artifact.norm_cap,
        description="Relationship Prefix-KV runtime test carrier.",
    )


def _conditioning(*, value: float, cold_start: bool = False):
    return PersonalConditioningSnapshot(
        schema_version=PERSONAL_CONDITIONING_SCHEMA_VERSION,
        state_vector=tuple(
            0.0 if cold_start else value
            for _ in PERSONAL_CONDITIONING_VECTOR_LABELS
        ),
        vector_labels=PERSONAL_CONDITIONING_VECTOR_LABELS,
        source_versions=(("user_model", 1),),
        source_fingerprint=f"prefix-injection-{value}",
        confidence=0.0 if cold_start else 0.72,
        is_cold_start=cold_start,
        description="prefix carrier test state",
    )


def _generate(runtime, **kwargs):
    return runtime.generate(
        prompt=PROMPT,
        system_context=SYSTEM,
        chat_messages=MESSAGES,
        max_new_tokens=24,
        temperature=0.0,
        capture_residuals=False,
        **kwargs,
    )


def test_greedy_prefix_loop_reproduces_generate(runtime) -> None:
    baseline = _generate(runtime).text

    _, inputs = runtime._build_generation_inputs(
        prompt=PROMPT, system_context=SYSTEM, chat_messages=MESSAGES
    )
    prompt_length = int(inputs["input_ids"].shape[-1])
    output_ids = runtime._greedy_generate_with_prefix(
        model_inputs=inputs,
        prefix_pairs=None,
        max_new_tokens=24,
        repetition_penalty=1.08,
        temperature=0.0,
    )
    replayed = runtime._decode_generated_text(
        token_ids=output_ids[0, prompt_length:]
    )

    assert replayed == baseline


def test_unknown_carrier_is_rejected(runtime) -> None:
    with pytest.raises(ValueError, match="personal_conditioning_carrier"):
        _generate(runtime, personal_conditioning_carrier="soft-prompt")


def test_prefix_carrier_without_artifact_fails_loudly(runtime) -> None:
    # Falling back to the residual carrier here would publish an arm labelled
    # prefix-KV whose evidence came from a different channel.
    with pytest.raises(ValueError, match="requires a prefix artifact"):
        _generate(
            runtime,
            personal_conditioning_carrier="prefix_kv",
            personal_conditioning=_conditioning(value=0.8),
        )


@pytest.fixture(scope="module")
def prefix_runtime():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from volvence_zero.substrate import TransformersOpenWeightResidualRuntime

    return TransformersOpenWeightResidualRuntime(
        model_id=MODEL_ID,
        pretrained_source=_local_weights(),
        device="cpu",
        hook_layer_selection="middle",
        local_files_only=True,
        runtime_origin="hf-local",
        personal_conditioning_prefix=_artifact(seed=7),
    )


def test_prefix_runtime_reports_the_loaded_artifact(prefix_runtime) -> None:
    assert prefix_runtime.supports_prefix_kv is True
    assert len(prefix_runtime.personal_conditioning_prefix_id) == 64


def test_prefix_carrier_reports_injection_only_when_state_is_admitted(
    prefix_runtime,
) -> None:
    applied = _generate(
        prefix_runtime,
        personal_conditioning_carrier="prefix_kv",
        personal_conditioning=_conditioning(value=0.8),
    )
    cold = _generate(
        prefix_runtime,
        personal_conditioning_carrier="prefix_kv",
        personal_conditioning=_conditioning(value=0.0, cold_start=True),
    )
    absent = _generate(
        prefix_runtime,
        personal_conditioning_carrier="prefix_kv",
        personal_conditioning=None,
    )

    assert applied.personal_conditioning_applied is True
    # Same gating as the residual carrier: cold-start and absent snapshots
    # inject nothing, so the two carriers admit exactly the same states.
    assert cold.personal_conditioning_applied is False
    assert absent.personal_conditioning_applied is False


def test_prefix_carrier_conditions_prefill_capture(prefix_runtime) -> None:
    baseline = prefix_runtime.capture(source_text=PROMPT)
    conditioned_a = prefix_runtime.capture_conditioned(
        source_text=PROMPT,
        personal_conditioning=_conditioning(value=0.8),
        personal_conditioning_carrier="prefix_kv",
    )
    conditioned_b = prefix_runtime.capture_conditioned(
        source_text=PROMPT,
        personal_conditioning=_conditioning(value=0.2),
        personal_conditioning_carrier="prefix_kv",
    )

    assert baseline.personal_conditioning_applied is False
    assert conditioned_a.personal_conditioning_applied is True
    assert conditioned_b.personal_conditioning_applied is True
    assert conditioned_a.residual_activations != baseline.residual_activations
    assert conditioned_b.residual_activations != baseline.residual_activations
    assert conditioned_a.residual_activations != conditioned_b.residual_activations


def test_prefix_capture_rejects_missing_artifact(runtime) -> None:
    with pytest.raises(ValueError, match="requires a prefix artifact"):
        runtime.capture_conditioned(
            source_text=PROMPT,
            personal_conditioning=_conditioning(value=0.8),
            personal_conditioning_carrier="prefix_kv",
        )


def test_prefix_carrier_refuses_unseeded_sampling(prefix_runtime) -> None:
    with pytest.raises(ValueError, match="requires sampling_seed"):
        prefix_runtime.generate(
            prompt=PROMPT,
            system_context=SYSTEM,
            chat_messages=MESSAGES,
            max_new_tokens=8,
            temperature=0.7,
            capture_residuals=False,
            personal_conditioning_carrier="prefix_kv",
            personal_conditioning=_conditioning(value=0.8),
        )


def test_prefix_carrier_seeded_sampling_is_reproducible(prefix_runtime) -> None:
    first = prefix_runtime.generate(
        prompt=PROMPT,
        system_context=SYSTEM,
        chat_messages=MESSAGES,
        max_new_tokens=4,
        temperature=0.7,
        capture_residuals=False,
        personal_conditioning_carrier="prefix_kv",
        personal_conditioning=_conditioning(value=0.8),
        sampling_seed=1701,
    )
    second = prefix_runtime.generate(
        prompt=PROMPT,
        system_context=SYSTEM,
        chat_messages=MESSAGES,
        max_new_tokens=4,
        temperature=0.7,
        capture_residuals=False,
        personal_conditioning_carrier="prefix_kv",
        personal_conditioning=_conditioning(value=0.8),
        sampling_seed=1701,
    )

    assert first.text == second.text


def test_residual_carrier_is_unchanged_by_a_loaded_prefix(
    runtime, prefix_runtime
) -> None:
    # Loading a prefix artifact must not perturb the default carrier: the
    # rollback story for this package is "omit the artifact", and that is only
    # true if a loaded-but-unused artifact is inert.
    conditioning = _conditioning(value=0.8)
    plain = _generate(runtime, personal_conditioning=conditioning).text
    with_artifact = _generate(
        prefix_runtime, personal_conditioning=conditioning
    ).text

    assert plain == with_artifact


@pytest.fixture(scope="module")
def relationship_prefix_runtime():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from volvence_zero.substrate import TransformersOpenWeightResidualRuntime

    artifact = _relationship_artifact()
    runtime = TransformersOpenWeightResidualRuntime(
        model_id=MODEL_ID,
        pretrained_source=_local_weights(),
        device="cpu",
        hook_layer_selection="middle",
        local_files_only=True,
        runtime_origin="hf-local",
        relationship_conditioning_prefix=artifact,
    )
    return runtime, artifact


def test_relationship_prefix_reports_only_bank_attestation(
    relationship_prefix_runtime,
) -> None:
    runtime, artifact = relationship_prefix_runtime
    carrier = _relationship_carrier(artifact=artifact)

    result = _generate(
        runtime,
        conditioning_bank_carriers=(carrier,),
    )

    assert result.personal_conditioning_applied is False
    assert result.conditioning_bank_carriers_applied == (
        (ConditioningBankType.RELATIONSHIP.value, artifact.carrier_version),
    )
    assert runtime.relationship_conditioning_prefix_id == artifact.artifact_id


def test_loaded_relationship_prefix_is_inert_without_carrier(
    runtime,
    relationship_prefix_runtime,
) -> None:
    loaded_runtime, _ = relationship_prefix_runtime

    assert _generate(loaded_runtime).text == _generate(runtime).text


def test_relationship_prefix_fails_on_missing_artifact_and_label_drift(
    runtime,
    relationship_prefix_runtime,
) -> None:
    _, artifact = relationship_prefix_runtime
    carrier = _relationship_carrier(artifact=artifact)
    with pytest.raises(ValueError, match="requires a Relationship"):
        _generate(runtime, conditioning_bank_carriers=(carrier,))

    wrong = _relationship_carrier(
        artifact=artifact,
        labels=("wrong_a", "wrong_b", "wrong_c"),
    )
    with pytest.raises(ValueError, match="readout_labels"):
        _generate(
            relationship_prefix_runtime[0],
            conditioning_bank_carriers=(wrong,),
        )


@pytest.mark.parametrize(
    ("field_name", "field_value", "message"),
    (
        ("projector_version", "relationship-prefix-kv-carrier.v1:wrong", "version"),
        ("scale", 0.05, "scale"),
    ),
)
def test_relationship_prefix_rejects_carrier_artifact_drift(
    relationship_prefix_runtime,
    field_name: str,
    field_value: str | float,
    message: str,
) -> None:
    runtime, artifact = relationship_prefix_runtime
    carrier = replace(
        _relationship_carrier(artifact=artifact),
        **{field_name: field_value},
    )

    with pytest.raises(ValueError, match=message):
        _generate(runtime, conditioning_bank_carriers=(carrier,))
