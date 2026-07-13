"""Social cognition owners (R16-R20).

This subpackage holds the runtime ``RuntimeModule`` owners for the
social cognition learning layer. It collapses the previously
plain-flat ``social_*.py`` and ``social_*_runtime.py`` files into
five capability-domain files:

* ``identity`` — multi-party identity scope + memory-PE lifters
* ``role`` — conversational role assignment
* ``tom`` — four Theory-of-Mind owners + their LLM proposal runtime
* ``common_ground`` — dyad/group common-ground owner + its LLM
  proposal runtime
* ``group`` — group / joint commitment owner

The companion immutable contract types (``SocialPrediction`` /
``SocialPredictionError`` / ``MultiPartyIdentitySnapshot`` ...) live
in ``volvence_zero.social_cognition`` (the ``vz-contracts`` wheel) and
are imported from there.

Public API is re-exported from this package so consumers can write
``from volvence_zero.social import CommonGroundModule`` without caring
about the file split.
"""

from __future__ import annotations

from volvence_zero.social.common_ground import (
    CommonGroundModule,
    CommonGroundProposal,
    CommonGroundProposalBatch,
    LLMCommonGroundProposalRuntime,
)
from volvence_zero.social.group import GroupModule
from volvence_zero.social.identity import (
    MultiPartyIdentityModule,
    SocialPredictionAggregateModule,
    SocialPredictionErrorModule,
)
from volvence_zero.social.record_store import (
    TOM_SLOTS,
    PendingSocialPrediction,
    SocialRecordStore,
    apply_outcome_to_record,
    default_summary_similarity,
    settle_pending_predictions,
)
from volvence_zero.social.role import ConversationalRoleModule
from volvence_zero.social.tom import (
    BeliefAboutOtherModule,
    FeelingAboutOtherModule,
    IntentAboutOtherModule,
    LLMToMProposalRuntime,
    PreferenceAboutOtherModule,
)

__all__ = [
    "TOM_SLOTS",
    "BeliefAboutOtherModule",
    "CommonGroundModule",
    "CommonGroundProposal",
    "CommonGroundProposalBatch",
    "ConversationalRoleModule",
    "FeelingAboutOtherModule",
    "GroupModule",
    "IntentAboutOtherModule",
    "LLMCommonGroundProposalRuntime",
    "LLMToMProposalRuntime",
    "MultiPartyIdentityModule",
    "PendingSocialPrediction",
    "PreferenceAboutOtherModule",
    "SocialPredictionAggregateModule",
    "SocialPredictionErrorModule",
    "SocialRecordStore",
    "apply_outcome_to_record",
    "default_summary_similarity",
    "settle_pending_predictions",
]
