from __future__ import annotations

from dataclasses import dataclass

from volvence_zero.internal_rl.environment import InternalRLProofEpisode, InternalRLProofSubgoal


@dataclass(frozen=True)
class HierarchicalLocation:
    location_id: str
    role: str
    target_signature: tuple[float, ...] = ()
    completion_threshold: float = 0.72
    min_persistence: int = 2
    credit_horizon: int = 2
    observation_weight: float = 0.24
    effect_weight: float = 0.56
    control_weight: float = 0.20
    description: str = ""

    @property
    def is_objective(self) -> bool:
        return bool(self.target_signature)


@dataclass(frozen=True)
class HierarchicalTransition:
    source_id: str
    target_id: str
    structural_role: str = "corridor"
    description: str = ""


@dataclass(frozen=True)
class HierarchicalRouteSpec:
    case_id: str
    split: str
    source_text: str
    waypoints: tuple[str, ...]
    distractor_ids: tuple[str, ...] = ()
    description: str = ""


@dataclass(frozen=True)
class MiniHierarchicalCase:
    case_id: str
    split: str
    source_text: str
    environment_id: str
    route_signature: tuple[str, ...]
    branch_depth: int
    proof_episode: InternalRLProofEpisode
    description: str


@dataclass(frozen=True)
class HierarchicalEpisodeState:
    case_id: str
    split: str
    source_text: str
    route_waypoints: tuple[str, ...]
    distractor_ids: tuple[str, ...]
    current_location_id: str
    step_index: int = 0
    visited_locations: tuple[str, ...] = ()
    completed_objective_ids: tuple[str, ...] = ()
    done: bool = False
    success: bool = False


@dataclass(frozen=True)
class HierarchicalObservation:
    case_id: str
    current_location_id: str
    available_targets: tuple[str, ...]
    completed_objective_ids: tuple[str, ...]
    remaining_route: tuple[str, ...]
    done: bool
    description: str


@dataclass(frozen=True)
class HierarchicalStepFeedback:
    source_id: str
    target_id: str
    reached_location_id: str
    structural_role: str
    objective_completed: bool
    distractor_hit: bool
    route_advanced: bool
    done: bool
    success: bool
    description: str


@dataclass(frozen=True)
class HierarchicalStepResult:
    next_state: HierarchicalEpisodeState
    observation: HierarchicalObservation
    feedback: HierarchicalStepFeedback


@dataclass(frozen=True)
class MiniHierarchicalEnvironment:
    env_id: str
    entry_location_id: str
    locations: tuple[HierarchicalLocation, ...]
    transitions: tuple[HierarchicalTransition, ...]
    description: str = ""

    def _location_map(self) -> dict[str, HierarchicalLocation]:
        mapping = {location.location_id: location for location in self.locations}
        if len(mapping) != len(self.locations):
            raise ValueError("MiniHierarchicalEnvironment requires unique location ids.")
        if self.entry_location_id not in mapping:
            raise ValueError(f"Unknown entry location {self.entry_location_id!r}.")
        return mapping

    def _adjacency(self) -> dict[str, tuple[HierarchicalTransition, ...]]:
        location_map = self._location_map()
        adjacency: dict[str, list[HierarchicalTransition]] = {location_id: [] for location_id in location_map}
        for transition in self.transitions:
            if transition.source_id not in location_map or transition.target_id not in location_map:
                raise ValueError(
                    f"Unknown transition edge {transition.source_id!r}->{transition.target_id!r} in environment {self.env_id!r}."
                )
            adjacency[transition.source_id].append(transition)
        return {location_id: tuple(edges) for location_id, edges in adjacency.items()}

    def location(self, location_id: str) -> HierarchicalLocation:
        location = self._location_map().get(location_id)
        if location is None:
            raise ValueError(f"Unknown location {location_id!r} in environment {self.env_id!r}.")
        return location

    def objective_locations(self) -> tuple[HierarchicalLocation, ...]:
        return tuple(location for location in self.locations if location.is_objective)

    def reset(self, route: HierarchicalRouteSpec) -> HierarchicalEpisodeState:
        self.validate_route(route)
        return HierarchicalEpisodeState(
            case_id=route.case_id,
            split=route.split,
            source_text=route.source_text,
            route_waypoints=route.waypoints,
            distractor_ids=route.distractor_ids,
            current_location_id=self.entry_location_id,
            step_index=0,
            visited_locations=(self.entry_location_id,),
            completed_objective_ids=(),
            done=False,
            success=False,
        )

    def observe(self, state: HierarchicalEpisodeState) -> HierarchicalObservation:
        adjacency = self._adjacency()
        next_targets = tuple(
            transition.target_id for transition in adjacency.get(state.current_location_id, ())
        )
        remaining_route = state.route_waypoints[state.step_index + 1 :] if state.step_index + 1 < len(state.route_waypoints) else ()
        return HierarchicalObservation(
            case_id=state.case_id,
            current_location_id=state.current_location_id,
            available_targets=next_targets,
            completed_objective_ids=state.completed_objective_ids,
            remaining_route=remaining_route,
            done=state.done,
            description=(
                f"Episode {state.case_id} at {state.current_location_id} with "
                f"{len(next_targets)} outgoing transitions and {len(remaining_route)} route steps remaining."
            ),
        )

    def step(
        self,
        state: HierarchicalEpisodeState,
        *,
        target_id: str,
    ) -> HierarchicalStepResult:
        if state.done:
            raise ValueError(f"Episode {state.case_id!r} is already done.")
        adjacency = self._adjacency()
        outgoing = adjacency.get(state.current_location_id, ())
        transition = next((edge for edge in outgoing if edge.target_id == target_id), None)
        if transition is None:
            available = tuple(edge.target_id for edge in outgoing)
            raise ValueError(
                f"Invalid transition {state.current_location_id!r}->{target_id!r} in environment {self.env_id!r}; "
                f"available targets are {available}."
            )
        next_location = self.location(target_id)
        expected_index = min(state.step_index + 1, len(state.route_waypoints) - 1)
        expected_target = state.route_waypoints[expected_index]
        route_advanced = target_id == expected_target
        completed_objectives = state.completed_objective_ids
        objective_completed = next_location.is_objective and next_location.location_id not in completed_objectives
        if objective_completed:
            completed_objectives = completed_objectives + (next_location.location_id,)
        distractor_hit = next_location.location_id in state.distractor_ids
        next_step_index = state.step_index + 1 if route_advanced else state.step_index
        done = next_step_index >= len(state.route_waypoints) - 1
        target_objectives = tuple(
            location.location_id
            for location in self._route_objectives(
                HierarchicalRouteSpec(
                    case_id=state.case_id,
                    split=state.split,
                    source_text=state.source_text,
                    waypoints=state.route_waypoints,
                    distractor_ids=state.distractor_ids,
                )
            )
        )
        success = done and all(objective_id in completed_objectives for objective_id in target_objectives)
        next_state = HierarchicalEpisodeState(
            case_id=state.case_id,
            split=state.split,
            source_text=state.source_text,
            route_waypoints=state.route_waypoints,
            distractor_ids=state.distractor_ids,
            current_location_id=next_location.location_id,
            step_index=next_step_index,
            visited_locations=state.visited_locations + (next_location.location_id,),
            completed_objective_ids=completed_objectives,
            done=done,
            success=success,
        )
        feedback = HierarchicalStepFeedback(
            source_id=state.current_location_id,
            target_id=target_id,
            reached_location_id=next_location.location_id,
            structural_role=transition.structural_role,
            objective_completed=objective_completed,
            distractor_hit=distractor_hit,
            route_advanced=route_advanced,
            done=done,
            success=success,
            description=(
                f"{state.current_location_id}->{target_id} via {transition.structural_role}; "
                f"objective_completed={objective_completed} distractor_hit={distractor_hit} "
                f"route_advanced={route_advanced} done={done}."
            ),
        )
        return HierarchicalStepResult(
            next_state=next_state,
            observation=self.observe(next_state),
            feedback=feedback,
        )

    def route_branch_depth(self, waypoints: tuple[str, ...]) -> int:
        adjacency = self._adjacency()
        depth = 0
        visited: set[str] = set()
        for location_id in waypoints:
            location = self.location(location_id)
            outgoing = adjacency.get(location_id, ())
            if location.role in {"junction", "hub", "branch"}:
                depth += 1
            if len(outgoing) > 1:
                depth += 1
            if location_id in visited:
                depth += 1
            visited.add(location_id)
        return depth

    def validate_route(self, route: HierarchicalRouteSpec) -> None:
        if not route.waypoints:
            raise ValueError("Hierarchical route requires at least one waypoint.")
        if route.waypoints[0] != self.entry_location_id:
            raise ValueError(
                f"Route {route.case_id!r} must start at entry {self.entry_location_id!r}, got {route.waypoints[0]!r}."
            )
        adjacency = self._adjacency()
        for location_id in route.waypoints:
            self.location(location_id)
        for source_id, target_id in zip(route.waypoints, route.waypoints[1:]):
            if target_id not in {transition.target_id for transition in adjacency.get(source_id, ())}:
                raise ValueError(
                    f"Route {route.case_id!r} uses missing transition {source_id!r}->{target_id!r} "
                    f"in environment {self.env_id!r}."
                )
        for distractor_id in route.distractor_ids:
            self.location(distractor_id)

    def _route_objectives(self, route: HierarchicalRouteSpec) -> tuple[HierarchicalLocation, ...]:
        return tuple(
            location
            for location_id in route.waypoints
            for location in (self.location(location_id),)
            if location.is_objective
        )

    def _route_distractors(self, route: HierarchicalRouteSpec) -> tuple[HierarchicalLocation, ...]:
        if route.distractor_ids:
            return tuple(self.location(location_id) for location_id in route.distractor_ids if self.location(location_id).is_objective)
        route_ids = set(route.waypoints)
        return tuple(
            location
            for location in self.objective_locations()
            if location.location_id not in route_ids
        )

    def build_proof_episode(self, route: HierarchicalRouteSpec) -> InternalRLProofEpisode:
        self.validate_route(route)
        subgoals = tuple(
            InternalRLProofSubgoal(
                subgoal_id=location.location_id,
                target_signature=location.target_signature,
                completion_threshold=location.completion_threshold,
                min_persistence=location.min_persistence,
                credit_horizon=location.credit_horizon,
                observation_weight=location.observation_weight,
                effect_weight=location.effect_weight,
                control_weight=location.control_weight,
                description=location.description or f"Objective location {location.location_id}.",
            )
            for location in self._route_objectives(route)
        )
        distractors = tuple(location.target_signature for location in self._route_distractors(route))
        return InternalRLProofEpisode(
            episode_id=route.case_id,
            subgoals=subgoals,
            distractor_signatures=distractors,
            subgoal_reward=0.28,
            terminal_reward=1.15,
            distractor_penalty=0.14 if self.route_branch_depth(route.waypoints) >= 3 else 0.10,
            failure_penalty=0.32 if len(subgoals) >= 3 else 0.26,
            description=(
                route.description
                or f"Mini hierarchical episode in {self.env_id} over route {route.waypoints} with {len(distractors)} distractors."
            ),
        )

    def build_case(self, route: HierarchicalRouteSpec) -> MiniHierarchicalCase:
        state = self.reset(route)
        for target_id in route.waypoints[1:]:
            state = self.step(state, target_id=target_id).next_state
        proof_episode = self.build_proof_episode(route)
        return MiniHierarchicalCase(
            case_id=route.case_id,
            split=route.split,
            source_text=route.source_text,
            environment_id=self.env_id,
            route_signature=state.visited_locations,
            branch_depth=self.route_branch_depth(state.visited_locations),
            proof_episode=proof_episode,
            description=route.description or proof_episode.description,
        )
