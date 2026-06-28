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

"""Companion CAMEL Baseline — standard agent-framework baseline for CompanionBench.

Public re-exports are intentionally small. Heavy modules (server, the real
CAMEL backend) stay importable on demand so pure-data consumers (tests, typing,
ablation analysers) can ``import companion_camel_baseline`` without pulling
aiohttp or camel-ai into their import graph.

Why this wheel exists and how it composes the agent + memory: read the README
and ``docs/specs/companion-ablation.md``.
"""

from __future__ import annotations

from companion_camel_baseline.backend import (
    AgentReply,
    CamelBackend,
    EchoCamelBackend,
)
from companion_camel_baseline.memory_store import (
    SessionMemoryRecord,
    StoreMode,
    open_store,
)

__all__ = (
    "AgentReply",
    "CamelBackend",
    "EchoCamelBackend",
    "SessionMemoryRecord",
    "StoreMode",
    "open_store",
)

__version__ = "0.1.0a0"
