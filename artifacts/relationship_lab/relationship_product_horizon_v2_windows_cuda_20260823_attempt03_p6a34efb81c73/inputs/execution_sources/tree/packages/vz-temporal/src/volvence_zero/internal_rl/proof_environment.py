from __future__ import annotations

import hashlib
import random
from collections import deque
from dataclasses import dataclass

from volvence_zero.internal_rl.environment import (
    InternalRLProofEpisode,
    InternalRLProofSubgoal,
    sparse_proof_reward_taxonomy,
)


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
    split_detail: str = ""
    description: str = ""


@dataclass(frozen=True)
class MiniHierarchicalCase:
    case_id: str
    split: str
    split_detail: str
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
                    "Unknown transition edge "
                    f"{transition.source_id!r}->{transition.target_id!r} "
                    f"in environment {self.env_id!r}."
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
        remaining_route = (
            state.route_waypoints[state.step_index + 1 :]
            if state.step_index + 1 < len(state.route_waypoints)
            else ()
        )
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
                    split_detail=state.split,
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

    def shortest_path(self, source_id: str, target_id: str) -> tuple[str, ...]:
        """BFS shortest directed path source->target, inclusive of both ends.

        Raises when the two nodes are unknown or unreachable. Ties are broken
        deterministically by transition declaration order so a generated
        corpus is reproducible from its seed alone.
        """

        location_map = self._location_map()
        if source_id not in location_map:
            raise ValueError(f"Unknown path source {source_id!r} in {self.env_id!r}.")
        if target_id not in location_map:
            raise ValueError(f"Unknown path target {target_id!r} in {self.env_id!r}.")
        if source_id == target_id:
            return (source_id,)
        adjacency = self._adjacency()
        predecessor: dict[str, str] = {source_id: source_id}
        queue: deque[str] = deque((source_id,))
        while queue:
            current = queue.popleft()
            for transition in adjacency.get(current, ()):  # declaration order
                nxt = transition.target_id
                if nxt in predecessor:
                    continue
                predecessor[nxt] = current
                if nxt == target_id:
                    queue.clear()
                    break
                queue.append(nxt)
        if target_id not in predecessor:
            raise ValueError(
                f"No directed path {source_id!r}->{target_id!r} in {self.env_id!r}."
            )
        reversed_path = [target_id]
        while reversed_path[-1] != source_id:
            reversed_path.append(predecessor[reversed_path[-1]])
        return tuple(reversed(reversed_path))

    def stitch_waypoints(self, objective_order: tuple[str, ...]) -> tuple[str, ...]:
        """Expand an ordered objective visit list into a full waypoint path.

        Starts at the entry, then walks the shortest path to each objective in
        turn, dropping the duplicated junction node at each join so the result
        is a valid step-by-step route consumable by ``reset``/``step``.
        """

        if not objective_order:
            raise ValueError("objective_order must contain at least one objective.")
        waypoints: list[str] = [self.entry_location_id]
        for objective_id in objective_order:
            leg = self.shortest_path(waypoints[-1], objective_id)
            waypoints.extend(leg[1:])
        if len(waypoints) < 2:
            raise ValueError(
                f"Stitched route for {objective_order!r} collapsed to a single node."
            )
        return tuple(waypoints)

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
        for source_id, target_id in zip(
            route.waypoints, route.waypoints[1:], strict=False
        ):
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
            return tuple(
                self.location(location_id)
                for location_id in route.distractor_ids
                if self.location(location_id).is_objective
            )
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
                nominal_completion_threshold=location.completion_threshold,
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
            reward_profile="proof-sparse-terminal-delayed",
            split_detail=route.split_detail or route.split,
            reward_taxonomy=sparse_proof_reward_taxonomy(),
            description=(
                route.description
                or (
                    f"Mini hierarchical episode in {self.env_id} over route "
                    f"{route.waypoints} with {len(distractors)} distractors."
                )
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
            split_detail=route.split_detail or route.split,
            source_text=route.source_text,
            environment_id=self.env_id,
            route_signature=state.visited_locations,
            branch_depth=self.route_branch_depth(state.visited_locations),
            proof_episode=proof_episode,
            description=route.description or proof_episode.description,
        )


# ---------------------------------------------------------------------------
# Seeded procedural generation (ETA LLM-transfer ladder, Stage 1).
#
# The 7 hardcoded default routes were too few for the rate-distortion criterion:
# a controller could memorise them, so the KL rate axis never had to trade
# information for action accuracy. These generators build a larger, seeded
# environment and a compositionally train/heldout-disjoint route corpus so the
# rate axis can be tested against genuine data richness. They are the
# environment owner's own construction path (no consumer rebuilds this state).
# ---------------------------------------------------------------------------

# Objective labels must (a) be highly common single BPE tokens so the steered
# action scorer's distinct-first-token contract holds, and (b) carry no route
# ordering information. Colour words satisfy both and mirror the paper's
# coloured-location tasks.
_GENERATOR_OBJECTIVE_WORDS: tuple[str, ...] = (
    "red",
    "blue",
    "green",
    "yellow",
    "orange",
    "purple",
    "black",
    "white",
    "brown",
    "silver",
)

# Relay-corridor labels. Corridors are also action targets (a route may step
# through one), so like objectives they must be distinct single tokens; these
# compass words never collide with the colour objectives or "hub".
_GENERATOR_CORRIDOR_WORDS: tuple[str, ...] = (
    "north",
    "south",
    "east",
    "west",
    "center",
    "edge",
)

# Neutral thematic vocabulary for the abstract task-context sentence. It is
# route-correlated (seeded by the objective ordering) but never names an
# objective, so route identity stays a latent the controller must recover.
_GENERATOR_CONTEXT_WORDS: tuple[str, ...] = (
    "steady",
    "guidance",
    "alignment",
    "planning",
    "support",
    "corridor",
    "branch",
    "anchor",
    "careful",
    "repair",
    "warmth",
    "continuity",
    "reflective",
    "memory",
    "return",
    "loop",
    "dense",
    "reordering",
    "horizon",
    "patient",
    "grounded",
    "attentive",
    "resilient",
    "deliberate",
)


def _stable_hash(text: str) -> int:
    """Process-independent hash (Python's ``hash`` is salted per process)."""

    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def _seeded_signature(rng: random.Random) -> tuple[float, ...]:
    return tuple(round(rng.uniform(0.12, 0.98), 4) for _ in range(3))


def generate_hierarchical_environment(
    *,
    seed: int,
    objective_count: int = 8,
    corridor_count: int = 2,
    extra_edge_probability: float = 0.35,
    env_id: str = "eta-generated-hierarchy",
) -> MiniHierarchicalEnvironment:
    """Build a seeded hierarchical environment with a hub-relay backbone.

    The hub guarantees every objective ordering is realizable (obj->hub->obj),
    while seeded direct objective->objective corridors and entry shortcuts give
    genuine branch structure and shared subprograms across routes. Increasing
    ``objective_count`` widens the route space; ``extra_edge_probability`` tunes
    how many routes share direct corridors versus relaying through the hub.
    """

    if not 2 <= objective_count <= len(_GENERATOR_OBJECTIVE_WORDS):
        raise ValueError(
            "objective_count must be in "
            f"[2, {len(_GENERATOR_OBJECTIVE_WORDS)}], got {objective_count}."
        )
    if not 0 <= corridor_count <= len(_GENERATOR_CORRIDOR_WORDS):
        raise ValueError(
            "corridor_count must be in "
            f"[0, {len(_GENERATOR_CORRIDOR_WORDS)}], got {corridor_count}."
        )
    if not 0.0 <= extra_edge_probability <= 1.0:
        raise ValueError("extra_edge_probability must be in [0, 1].")
    rng = random.Random(seed)
    objectives = _GENERATOR_OBJECTIVE_WORDS[:objective_count]
    corridors = _GENERATOR_CORRIDOR_WORDS[:corridor_count]

    locations: list[HierarchicalLocation] = [
        HierarchicalLocation(
            location_id="entry",
            role="entry",
            description="Generated environment entry anchor.",
        ),
        HierarchicalLocation(
            location_id="hub",
            role="hub",
            description="Central relay hub shared by every route.",
        ),
    ]
    for objective_id in objectives:
        locations.append(
            HierarchicalLocation(
                location_id=objective_id,
                role="objective",
                target_signature=_seeded_signature(rng),
                completion_threshold=round(rng.uniform(0.70, 0.80), 3),
                min_persistence=rng.randint(1, 3),
                credit_horizon=rng.randint(2, 3),
                observation_weight=0.22,
                effect_weight=0.56,
                control_weight=0.22,
                description=f"Generated objective location {objective_id}.",
            )
        )
    for corridor_id in corridors:
        locations.append(
            HierarchicalLocation(
                location_id=corridor_id,
                role="corridor",
                description=f"Generated relay corridor {corridor_id}.",
            )
        )

    edges: list[HierarchicalTransition] = [
        HierarchicalTransition("entry", "hub", structural_role="branch"),
    ]
    # Hub relay backbone: guarantees reachability for any objective ordering.
    for objective_id in objectives:
        edges.append(
            HierarchicalTransition("hub", objective_id, structural_role="corridor")
        )
        edges.append(
            HierarchicalTransition(objective_id, "hub", structural_role="return")
        )
    # Corridors hang off the hub to add optional depth.
    for corridor_id in corridors:
        edges.append(
            HierarchicalTransition("hub", corridor_id, structural_role="loop")
        )
        edges.append(
            HierarchicalTransition(corridor_id, "hub", structural_role="loop")
        )
    # A direct entry shortcut so not every route opens through the hub.
    edges.append(
        HierarchicalTransition("entry", objectives[0], structural_role="corridor")
    )
    # Seeded direct objective->objective corridors for shared subprograms.
    seen_edges = {(edge.source_id, edge.target_id) for edge in edges}
    for source_id in objectives:
        for target_id in objectives:
            if source_id == target_id:
                continue
            if (source_id, target_id) in seen_edges:
                continue
            if rng.random() < extra_edge_probability:
                edges.append(
                    HierarchicalTransition(
                        source_id, target_id, structural_role="branch"
                    )
                )
                seen_edges.add((source_id, target_id))

    return MiniHierarchicalEnvironment(
        env_id=env_id,
        entry_location_id="entry",
        locations=tuple(locations),
        transitions=tuple(edges),
        description=(
            f"Seeded hierarchical environment (seed={seed}) with "
            f"{objective_count} objectives, {corridor_count} corridors, and a "
            "hub relay backbone for the ETA rate-distortion corpus."
        ),
    )


def _context_sentence(objective_order: tuple[str, ...], *, word_count: int = 8) -> str:
    """Deterministic non-leaking thematic sentence keyed to the route order."""

    rng = random.Random(_stable_hash("|".join(objective_order)))
    picks = rng.sample(
        _GENERATOR_CONTEXT_WORDS,
        k=min(word_count, len(_GENERATOR_CONTEXT_WORDS)),
    )
    return " ".join(picks)


def _distinct_orderings(
    objectives: tuple[str, ...],
    *,
    length: int,
    rng: random.Random,
) -> list[tuple[str, ...]]:
    from itertools import permutations

    pool = [tuple(order) for order in permutations(objectives, length)]
    rng.shuffle(pool)
    return pool


def generate_hierarchical_routes(
    *,
    environment: MiniHierarchicalEnvironment,
    seed: int,
    train_count: int,
    heldout_count: int,
    train_lengths: tuple[int, ...] = (2, 3),
    heldout_lengths: tuple[int, ...] = (3, 4),
    distractor_count: int = 2,
) -> tuple[HierarchicalRouteSpec, ...]:
    """Sample a compositionally train/heldout-disjoint route corpus.

    Routes are built from ordered objective visit lists stitched into full
    waypoint paths via the environment's hub relay. Train and heldout objective
    orderings are guaranteed set-disjoint: lengths present in only one split go
    wholly to that split, and any shared length is hash-partitioned so no
    ordering appears in both. This operationalises the paper's compositional /
    length-generalization OOD split without leaking any evaluation label.
    """

    if train_count < 1 or heldout_count < 1:
        raise ValueError("train_count and heldout_count must both be >= 1.")
    objectives = tuple(
        location.location_id
        for location in environment.objective_locations()
    )
    if len(objectives) < max((*train_lengths, *heldout_lengths)):
        raise ValueError(
            "environment has too few objectives for the requested route "
            f"lengths: {len(objectives)} objectives vs max length "
            f"{max((*train_lengths, *heldout_lengths))}."
        )
    rng = random.Random(seed * 2_654_435_761 % (2**63))
    shared_lengths = set(train_lengths) & set(heldout_lengths)

    train_pool: list[tuple[str, ...]] = []
    heldout_pool: list[tuple[str, ...]] = []
    for length in sorted(set(train_lengths) | set(heldout_lengths)):
        orderings = _distinct_orderings(objectives, length=length, rng=rng)
        in_train = length in set(train_lengths)
        in_heldout = length in set(heldout_lengths)
        for ordering in orderings:
            if length in shared_lengths:
                # Hash-partition shared lengths so the same ordering can never
                # land in both splits.
                bucket = _stable_hash("split|" + "|".join(ordering)) % 2
                if bucket == 0:
                    train_pool.append(ordering)
                else:
                    heldout_pool.append(ordering)
            elif in_train:
                train_pool.append(ordering)
            elif in_heldout:
                heldout_pool.append(ordering)

    if len(train_pool) < train_count:
        raise ValueError(
            f"train pool has only {len(train_pool)} distinct orderings for "
            f"{train_count} requested train routes; raise objective_count or "
            "lengths."
        )
    if len(heldout_pool) < heldout_count:
        raise ValueError(
            f"heldout pool has only {len(heldout_pool)} distinct orderings for "
            f"{heldout_count} requested heldout routes; raise objective_count "
            "or lengths."
        )
    rng.shuffle(train_pool)
    rng.shuffle(heldout_pool)
    train_orderings = train_pool[:train_count]
    heldout_orderings = heldout_pool[:heldout_count]

    overlap = set(train_orderings) & set(heldout_orderings)
    if overlap:
        raise RuntimeError(
            f"train/heldout ordering overlap detected: {sorted(overlap)[:3]}."
        )

    routes: list[HierarchicalRouteSpec] = []
    for split, orderings in (
        ("train", train_orderings),
        ("heldout", heldout_orderings),
    ):
        for index, ordering in enumerate(orderings):
            waypoints = environment.stitch_waypoints(ordering)
            ordering_set = set(ordering)
            distractors = tuple(
                objective_id
                for objective_id in objectives
                if objective_id not in ordering_set
            )[:distractor_count]
            routes.append(
                HierarchicalRouteSpec(
                    case_id=f"gen-{split}-{index:04d}-{'-'.join(ordering)}",
                    split=split,
                    source_text=_context_sentence(ordering),
                    waypoints=waypoints,
                    distractor_ids=distractors,
                    split_detail=f"generated-{split}-len{len(ordering)}",
                    description=(
                        f"Generated {split} route visiting {', '.join(ordering)} "
                        f"over {len(waypoints)} waypoints."
                    ),
                )
            )
    return tuple(routes)
