# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Companion Ref-Harness — vendor-neutral agent wrapper for CompanionBench.

Public re-exports are intentionally small. Heavy modules (upstream
client, server, store) stay importable on demand so that pure-data
consumers (tests, typing, ablation analysers) can ``import
companion_ref_harness`` without pulling aiohttp into their import
graph.

For a tour of why this wheel exists and how it composes the four
components, read the README and
``docs/moving forward/companion-ref-harness-packet.md``.
"""

from __future__ import annotations

from companion_ref_harness.embed import Embedder, HashingEmbedder
from companion_ref_harness.episodic import (
    EpisodicEvent,
    EpisodicExtractor,
    StubEpisodicExtractor,
)
from companion_ref_harness.policy import (
    ComponentSet,
    HarnessComponent,
    HarnessPolicy,
    parse_component_set,
)
from companion_ref_harness.session_summary import (
    SessionSummary,
    SummaryExtractor,
    StubSummaryExtractor,
)
from companion_ref_harness.user_model import (
    StubUserFactExtractor,
    UserFact,
    UserFactExtractor,
)

__all__ = (
    "ComponentSet",
    "HarnessComponent",
    "HarnessPolicy",
    "parse_component_set",
    "SessionSummary",
    "SummaryExtractor",
    "StubSummaryExtractor",
    "Embedder",
    "HashingEmbedder",
    "UserFact",
    "UserFactExtractor",
    "StubUserFactExtractor",
    "EpisodicEvent",
    "EpisodicExtractor",
    "StubEpisodicExtractor",
)

__version__ = "0.1.0a0"
