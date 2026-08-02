"""Cross-family blind matching judge for carrier-identification evidence.

Supplies the missing half of claims 3 and 4 in
``docs/specs/state-kv-identification-evidence.md``: a judge that sees a
response and two candidate user descriptions, and nothing else.

Four properties make its votes worth putting in an artifact:

1. **Blind by signature.** :meth:`LocalTransformersBlindJudge.match` accepts a
   response string and two user ids. There is no parameter through which an arm
   label, a prompt, a fingerprint, or an internal state vector could reach it,
   so blindness is structural rather than a reviewer's promise.
2. **Cross-family, enforced.** Construction compares the judge's and the
   substrate's HF ``config.model_type`` and refuses to run when they match
   (spec §关键不变量 5). ``Qwen2.5`` is ``qwen2``; ``TinyLlama`` is ``llama``.
   A same-family judge would be scoring its own family's idiolect.
3. **Order-symmetrized, so position bias cannot manufacture a signal.** Each
   decision runs both candidate orderings and subtracts the two log-prob
   differences. A judge with a constant "always pick the first option" bias
   scores exactly zero and lands at chance -- which is what claim 4 requires of
   the control arm. Greedy scoring throughout: no sampling, no seed, same votes
   on a re-run.
4. **The judge's material is named in the artifact.** :class:`JudgeMaterial`
   carries a ``material_kind`` that ends up in the verdict, because "which
   candidate produced this reply" means something different when the candidates
   are described by session history than by a rendered state readout.

What this judge cannot do: it is a small local model, so a near-chance result is
as likely to mean "the judge is too weak" as "the state did not reach the
output". That ambiguity is why its size and id are recorded rather than
summarized away.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from importlib.resources import files
from dataclasses import dataclass
from typing import Any

__all__ = [
    "JUDGE_PROMPT_TEMPLATE_NAME",
    "JudgeMaterial",
    "JudgeMaterialKind",
    "LocalEmbeddingBlindJudge",
    "LocalTransformersBlindJudge",
    "load_judge_prompt_template",
    "resolve_model_family",
]


class JudgeMaterialKind:
    """What the judge was shown about each candidate user.

    Not an enum with a default: the two kinds support different claims, and a
    reader of the artifact must be able to tell which experiment was run.

    ``SESSION_HISTORY`` -- prose summarising the user's prior sessions, the
    material the spec's twin-session design assumes.
    ``RENDERED_STATE`` -- the owner-rendered natural-language form of the same
    typed readout the residual carries. Used when the arms differ only by state
    and there is no history to summarise; it makes claim 3 a test of "is this
    state legible in the output", which is narrower than "is this person
    recognisable".
    """

    SESSION_HISTORY = "session-history-summary"
    RENDERED_STATE = "rendered-state-statement"
    MATCHED_OUTCOME_RUBRIC = "matched-outcome-rubric"

    ALL = (SESSION_HISTORY, RENDERED_STATE, MATCHED_OUTCOME_RUBRIC)


@dataclass(frozen=True)
class JudgeMaterial:
    """One candidate user as the judge sees them."""

    user_id: str
    summary: str
    material_kind: str

    def __post_init__(self) -> None:
        if not self.user_id:
            raise ValueError("JudgeMaterial requires a user_id.")
        if not self.summary.strip():
            raise ValueError(
                f"JudgeMaterial for {self.user_id!r} has an empty summary: a "
                "candidate the judge cannot read is not a candidate."
            )
        if self.material_kind not in JudgeMaterialKind.ALL:
            raise ValueError(
                f"unknown judge material kind {self.material_kind!r}; "
                f"expected one of {JudgeMaterialKind.ALL}"
            )


JUDGE_PROMPT_TEMPLATE_NAME = "state_kv_blind_match.md"


def load_judge_prompt_template(name: str = JUDGE_PROMPT_TEMPLATE_NAME) -> str:
    """Load the blind-match prompt from ``prompts/`` (AGENTS §7).

    The judge's prompt is part of the published method: a reviewer re-running
    the matching task must be able to read the exact wording without reading
    Python, and a change to it must show up as a change to a template file.
    """

    return (
        files("volvence_zero")
        .joinpath("prompts", name)
        .read_text(encoding="utf-8")
    )


def resolve_model_family(*, model_id: str, local_files_only: bool = True) -> str:
    """Return the HF ``model_type`` for a model id or local snapshot path.

    Architecture, not vendor string: two checkpoints from different orgs can
    share an architecture, and the cross-family rule is about the architecture
    that produced the idiolect being judged.
    """

    transformers = importlib.import_module("transformers")
    config = transformers.AutoConfig.from_pretrained(
        model_id, local_files_only=local_files_only
    )
    family = str(getattr(config, "model_type", "")).strip().lower()
    if not family:
        raise ValueError(
            f"model {model_id!r} declares no config.model_type, so the "
            "cross-family judge rule cannot be checked."
        )
    return family


class LocalTransformersBlindJudge:
    """Two-alternative blind matcher backed by a local causal LM."""

    def __init__(
        self,
        *,
        judge_model_id: str,
        substrate_model_id: str,
        materials: Sequence[JudgeMaterial],
        judge_source: str | None = None,
        substrate_source: str | None = None,
        device: str = "cpu",
        local_files_only: bool = True,
        model: object | None = None,
        tokenizer: object | None = None,
        judge_family: str | None = None,
        substrate_family: str | None = None,
    ) -> None:
        """Construct the judge.

        ``judge_family`` / ``substrate_family`` override the HF-config lookup.
        They exist so the cross-family rule can be exercised without loading
        two checkpoints; passing a family that contradicts the checkpoint would
        only fool this guard, not the reader of the artifact, since both
        families are recorded in :meth:`as_json_dict`.
        """

        if len(materials) != 2:
            raise ValueError(
                "two-alternative matching needs exactly two candidate "
                f"materials, got {len(materials)}"
            )
        kinds = {material.material_kind for material in materials}
        if len(kinds) != 1:
            raise ValueError(
                "both candidates must be described with the same kind of "
                f"material, got {sorted(kinds)}: otherwise the judge is "
                "comparing unlike descriptions and any gap is an artefact of "
                "the material, not of the state."
            )
        self._materials = {material.user_id: material for material in materials}
        if len(self._materials) != 2:
            raise ValueError("candidate materials must have distinct user ids.")
        self.material_kind = materials[0].material_kind

        judge_family = judge_family or resolve_model_family(
            model_id=judge_source or judge_model_id,
            local_files_only=local_files_only,
        )
        substrate_family = substrate_family or resolve_model_family(
            model_id=substrate_source or substrate_model_id,
            local_files_only=local_files_only,
        )
        if judge_family == substrate_family:
            raise ValueError(
                "cross-family judge rule violated (spec §关键不变量 5): judge "
                f"{judge_model_id!r} and substrate {substrate_model_id!r} are "
                f"both {judge_family!r}. A substrate family may not judge its "
                "own outputs."
            )
        self.judge_family = judge_family
        self.substrate_family = substrate_family

        self._torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
        source = judge_source or judge_model_id
        self._tokenizer = tokenizer or transformers.AutoTokenizer.from_pretrained(
            source, local_files_only=local_files_only
        )
        self._model = model or transformers.AutoModelForCausalLM.from_pretrained(
            source, local_files_only=local_files_only
        )
        self._device = device
        self._model.to(device)
        self._model.eval()
        self._judge_model_id = judge_model_id
        self._template = load_judge_prompt_template()
        # Vote bookkeeping so a degenerate judge is visible in the artifact
        # rather than hidden inside an accuracy number.
        self.tie_count = 0
        self.decision_count = 0

    @property
    def judge_model_id(self) -> str:
        return self._judge_model_id

    def _letter_logprobs(self, prompt: str) -> tuple[float, float]:
        """Log-probabilities of "A" and "B" as the next token."""

        encoded = self._tokenizer(prompt, return_tensors="pt")
        encoded = {key: value.to(self._device) for key, value in encoded.items()}
        with self._torch.no_grad():
            logits = self._model(**encoded).logits[0, -1, :]
        log_probs = self._torch.log_softmax(logits.float(), dim=-1)
        scores: list[float] = []
        for letter in ("A", "B"):
            # Score the best-scoring surface form of the letter: tokenizers
            # differ on whether the leading space is part of the token, and
            # picking the wrong variant would measure tokenizer trivia.
            candidate_ids = {
                token_id
                for variant in (letter, f" {letter}")
                for token_id in self._tokenizer.encode(
                    variant, add_special_tokens=False
                )[:1]
            }
            if not candidate_ids:
                raise ValueError(
                    f"judge tokenizer produced no token for {letter!r}"
                )
            scores.append(max(float(log_probs[token_id]) for token_id in candidate_ids))
        return scores[0], scores[1]

    def match(self, *, response_text: str, candidate_user_ids: Sequence[str]) -> str:
        """Attribute one response to one of two candidate users.

        Runs both orderings and subtracts, so a constant positional preference
        cancels instead of becoming a signal.
        """

        ids = tuple(candidate_user_ids)
        if len(ids) != 2 or len(set(ids)) != 2:
            raise ValueError(
                f"two-alternative matching needs two distinct users, got {ids}"
            )
        missing = [user_id for user_id in ids if user_id not in self._materials]
        if missing:
            raise ValueError(
                f"no judge material for candidate(s): {', '.join(missing)}"
            )
        first, second = ids
        text = response_text.strip()
        if not text:
            raise ValueError(
                "cannot judge an empty response; an empty generation is a "
                "substrate failure to report, not a coin flip to record."
            )

        forward_a, forward_b = self._letter_logprobs(
            self._template.format(
                summary_a=self._materials[first].summary,
                summary_b=self._materials[second].summary,
                response=text,
            )
        )
        swapped_a, swapped_b = self._letter_logprobs(
            self._template.format(
                summary_a=self._materials[second].summary,
                summary_b=self._materials[first].summary,
                response=text,
            )
        )
        # Positive favours `first` in both orderings; the subtraction removes
        # any fixed preference for whichever option is presented first.
        score = (forward_a - forward_b) - (swapped_a - swapped_b)
        self.decision_count += 1
        if score == 0.0:
            self.tie_count += 1
        return first if score > 0.0 else second

    def as_json_dict(self) -> dict[str, Any]:
        """Judge provenance for the artifact."""

        return {
            "judge_model_id": self._judge_model_id,
            "judge_family": self.judge_family,
            "substrate_family": self.substrate_family,
            "material_kind": self.material_kind,
            "scoring_method": "causal-lm-order-symmetrized-letter-logprob-v1",
            "order_symmetrized": True,
            "greedy": True,
            "decision_count": self.decision_count,
            "tie_count": self.tie_count,
        }


class LocalEmbeddingBlindJudge:
    """Two-alternative blind matcher backed by a local embedding model."""

    def __init__(
        self,
        *,
        judge_model_id: str,
        substrate_model_id: str,
        materials: Sequence[JudgeMaterial],
        judge_source: str | None = None,
        substrate_source: str | None = None,
        device: str = "cpu",
        local_files_only: bool = True,
        model: object | None = None,
        tokenizer: object | None = None,
        judge_family: str | None = None,
        substrate_family: str | None = None,
    ) -> None:
        if len(materials) != 2:
            raise ValueError(
                "two-alternative matching needs exactly two candidate "
                f"materials, got {len(materials)}"
            )
        kinds = {material.material_kind for material in materials}
        if len(kinds) != 1:
            raise ValueError(
                "both candidates must be described with the same kind of "
                f"material, got {sorted(kinds)}"
            )
        self._materials = {material.user_id: material for material in materials}
        if len(self._materials) != 2:
            raise ValueError("candidate materials must have distinct user ids.")
        self.material_kind = materials[0].material_kind

        judge_family = judge_family or resolve_model_family(
            model_id=judge_source or judge_model_id,
            local_files_only=local_files_only,
        )
        substrate_family = substrate_family or resolve_model_family(
            model_id=substrate_source or substrate_model_id,
            local_files_only=local_files_only,
        )
        if judge_family == substrate_family:
            raise ValueError(
                "cross-family judge rule violated (spec §关键不变量 5): judge "
                f"{judge_model_id!r} and substrate {substrate_model_id!r} are "
                f"both {judge_family!r}. A substrate family may not judge its "
                "own outputs."
            )
        self.judge_family = judge_family
        self.substrate_family = substrate_family

        self._torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
        source = judge_source or judge_model_id
        self._tokenizer = tokenizer or transformers.AutoTokenizer.from_pretrained(
            source, local_files_only=local_files_only
        )
        self._model = model or transformers.AutoModel.from_pretrained(
            source, local_files_only=local_files_only
        )
        self._device = device
        self._model.to(device)
        self._model.eval()
        self._judge_model_id = judge_model_id
        self._material_embeddings = {
            user_id: self._embed(material.summary)
            for user_id, material in self._materials.items()
        }
        self.tie_count = 0
        self.decision_count = 0

    @property
    def judge_model_id(self) -> str:
        return self._judge_model_id

    def _embed(self, text: str):
        encoded = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        encoded = {key: value.to(self._device) for key, value in encoded.items()}
        with self._torch.no_grad():
            hidden = self._model(**encoded).last_hidden_state.to(
                self._torch.float32
            )
        mask = encoded["attention_mask"].unsqueeze(-1).to(self._torch.float32)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self._torch.nn.functional.normalize(pooled, p=2, dim=1)[0]

    def match(self, *, response_text: str, candidate_user_ids: Sequence[str]) -> str:
        ids = tuple(candidate_user_ids)
        if len(ids) != 2 or len(set(ids)) != 2:
            raise ValueError(
                f"two-alternative matching needs two distinct users, got {ids}"
            )
        missing = [user_id for user_id in ids if user_id not in self._materials]
        if missing:
            raise ValueError(
                f"no judge material for candidate(s): {', '.join(missing)}"
            )
        text = response_text.strip()
        if not text:
            raise ValueError(
                "cannot judge an empty response; an empty generation is a "
                "substrate failure to report, not a coin flip to record."
            )
        response = self._embed(text)
        scores = [
            float((response * self._material_embeddings[user_id]).sum())
            for user_id in ids
        ]
        self.decision_count += 1
        if scores[0] == scores[1]:
            self.tie_count += 1
        return ids[0] if scores[0] > scores[1] else ids[1]

    def as_json_dict(self) -> dict[str, Any]:
        return {
            "judge_model_id": self._judge_model_id,
            "judge_family": self.judge_family,
            "substrate_family": self.substrate_family,
            "material_kind": self.material_kind,
            "scoring_method": "embedding-cosine-mean-pool-v1",
            "order_symmetrized": False,
            "greedy": True,
            "decision_count": self.decision_count,
            "tie_count": self.tie_count,
        }
