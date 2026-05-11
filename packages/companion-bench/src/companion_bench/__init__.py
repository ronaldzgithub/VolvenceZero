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

"""Companion Bench — Long-Session Companion Benchmark reference implementation.

Public surface re-exports the most-used building blocks. Heavy modules
(arc_runner, judge_*) stay importable on demand to keep ``import companion_bench``
fast for tooling that only needs the spec / aggregator surfaces.

Previously circulated as LSCB; the wheel ships under ``companion-bench`` from
v1.0 onward.
"""

from __future__ import annotations

from companion_bench.spec import (
    AxisId,
    ExpectedAxes,
    FamilyId,
    ScenarioDisqualifier,
    ScenarioSpec,
    UserSimulatorSpec,
    load_scenario_yaml,
    load_scenarios_dir,
    scenario_hash,
)

__all__ = (
    "AxisId",
    "ExpectedAxes",
    "FamilyId",
    "ScenarioDisqualifier",
    "ScenarioSpec",
    "UserSimulatorSpec",
    "load_scenario_yaml",
    "load_scenarios_dir",
    "scenario_hash",
)

__version__ = "1.0.0a0"
