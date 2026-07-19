"""F6 AlphaEvolve — Evolutionary search for PE-optimal algorithms probe.

Hypothesis: Evolutionary search in algorithm space (code mutations + fitness
evaluation) discovers better solutions than random search. When fitness is
aligned with the true objective, evolution outperforms; when fitness is
misaligned (Goodhart), evolution optimizes the wrong thing.

Based on: DeepMind AlphaEvolve (arXiv 2506.13131) — LLM-guided evolution.

Cells:
- baseline (random_search): random algorithm sampling (no evolution)
- probe_on (evolutionary_search): mutation + selection with correct fitness
- probe_off (fixed_algorithm): no search at all (single fixed algorithm)
- counterfactual (goodhart_evolution): evolution with wrong fitness (Goodhart)

Eval: Toy optimization landscape (Rastrigin-style).
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ...framework.probe import (
    BaseProbe,
    GateReport,
    PrimitiveTag,
    ProbeContext,
    ReadoutBundle,
    RunOutcome,
    register_probe,
)
from ...framework.wiring import AblationCell


def _rastrigin(x: np.ndarray) -> float:
    """Rastrigin function (multimodal optimization landscape)."""
    n = len(x)
    return float(10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))


def _ackley(x: np.ndarray) -> float:
    """Ackley function (another multimodal landscape)."""
    n = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    return float(-20 * np.exp(-0.2 * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + 20 + np.e)


def _wrong_fitness(x: np.ndarray) -> float:
    """Wrong fitness: optimizes norm instead of Rastrigin (Goodhart)."""
    return float(np.sum(x**2))  # Just minimizes norm, ignores multimodal structure


def _generate_optimization_task(seed: int, dim: int = 5, n_evals: int = 100) -> dict:
    """Generate a toy optimization task."""
    rng = np.random.default_rng(seed)

    # Search bounds
    bounds = (-5.12, 5.12)

    # Initial population
    pop_size = 20
    initial_pop = rng.uniform(bounds[0], bounds[1], size=(pop_size, dim)).astype(np.float32)

    return {
        "dim": dim,
        "bounds": list(bounds),
        "n_evals": n_evals,
        "pop_size": pop_size,
        "initial_pop": initial_pop.tolist(),
        "source": "synthetic",
    }


def _evolutionary_search(
    initial_pop: np.ndarray,
    fitness_fn,
    n_evals: int,
    bounds: tuple,
    mutation_scale: float = 0.5,
    seed: int = 0,
) -> dict:
    """Simple (μ+λ) evolutionary strategy.

    Returns best fitness found and search trajectory.
    """
    rng = np.random.default_rng(seed)
    pop = initial_pop.copy()
    pop_size, dim = pop.shape

    # Evaluate initial population
    fitnesses = np.array([fitness_fn(x) for x in pop])
    best_fitness = float(fitnesses.min())
    trajectory = [best_fitness]

    evals_used = pop_size
    generation = 0

    while evals_used < n_evals:
        # Select top 50% as parents
        n_parents = max(2, pop_size // 2)
        parent_idx = np.argsort(fitnesses)[:n_parents]
        parents = pop[parent_idx]

        # Generate offspring via mutation
        n_offspring = pop_size - n_parents
        offspring = []
        for _ in range(n_offspring):
            parent = parents[rng.integers(0, n_parents)]
            # Gaussian mutation with adaptive scale
            child = parent + rng.standard_normal(dim).astype(np.float32) * mutation_scale
            child = np.clip(child, bounds[0], bounds[1])
            offspring.append(child)

        offspring = np.array(offspring, dtype=np.float32)
        offspring_fit = np.array([fitness_fn(x) for x in offspring])
        evals_used += n_offspring

        # Combine and select
        pop = np.concatenate([parents, offspring], axis=0)
        fitnesses = np.concatenate([fitnesses[parent_idx], offspring_fit])

        best_fitness = float(fitnesses.min())
        trajectory.append(best_fitness)
        generation += 1

        # Adaptive mutation
        mutation_scale *= 0.99

    best_idx = np.argmin(fitnesses)
    return {
        "best_fitness": best_fitness,
        "best_solution": pop[best_idx].tolist(),
        "trajectory": trajectory,
        "generations": generation,
        "evals_used": evals_used,
    }


def _random_search(
    dim: int,
    fitness_fn,
    n_evals: int,
    bounds: tuple,
    seed: int = 0,
) -> dict:
    """Pure random search baseline."""
    rng = np.random.default_rng(seed)
    best_fitness = float("inf")
    best_solution = None
    trajectory = []

    for i in range(n_evals):
        x = rng.uniform(bounds[0], bounds[1], size=dim).astype(np.float32)
        f = fitness_fn(x)
        if f < best_fitness:
            best_fitness = f
            best_solution = x.tolist()
        trajectory.append(best_fitness)

    return {
        "best_fitness": best_fitness,
        "best_solution": best_solution,
        "trajectory": trajectory,
        "generations": 0,
        "evals_used": n_evals,
    }


@register_probe
class AlphaEvolveProbe(BaseProbe):
    id = "alpha-evolve-v1"
    hypothesis = (
        "Evolutionary search with correct fitness outperforms random search. "
        "Evolution with wrong fitness (Goodhart) optimizes the proxy, not the true objective."
    )
    primitive = PrimitiveTag.F5_R15_FORMALIZATION  # Meta-optimization
    r_ids = ("R13", "R9")

    def knobs(self) -> dict[str, list]:
        return {
            "dim": [5, 10],
            "n_evals": [100, 200],
            "mutation_scale": [0.3, 0.5, 1.0],
        }

    def default_inputs(self, seed: int) -> Any:
        return _generate_optimization_task(seed=seed)

    def run_cell(self, ctx: ProbeContext, knobs: Mapping[str, Any]) -> RunOutcome:
        inputs = ctx.inputs
        dim = inputs["dim"]
        bounds = tuple(inputs["bounds"])
        n_evals = inputs["n_evals"]
        initial_pop = np.array(inputs["initial_pop"], dtype=np.float32)
        mutation_scale = knobs.get("mutation_scale", 0.5)

        if ctx.cell == AblationCell.BASELINE:
            # Random search
            result = _random_search(dim, _rastrigin, n_evals, bounds, seed=ctx.seed)

        elif ctx.cell == AblationCell.PROBE_ON:
            # Evolutionary search with correct fitness
            result = _evolutionary_search(
                initial_pop, _rastrigin, n_evals, bounds,
                mutation_scale=mutation_scale, seed=ctx.seed,
            )

        elif ctx.cell == AblationCell.PROBE_OFF:
            # Fixed algorithm: just evaluate the initial population mean
            mean_solution = initial_pop.mean(axis=0)
            fitness = _rastrigin(mean_solution)
            result = {
                "best_fitness": fitness,
                "best_solution": mean_solution.tolist(),
                "trajectory": [fitness],
                "generations": 0,
                "evals_used": 1,
            }

        elif ctx.cell == AblationCell.COUNTERFACTUAL:
            # Goodhart evolution: optimize wrong fitness
            result = _evolutionary_search(
                initial_pop, _wrong_fitness, n_evals, bounds,
                mutation_scale=mutation_scale, seed=ctx.seed,
            )
            # But evaluate on TRUE fitness
            best_solution = np.array(result["best_solution"], dtype=np.float32)
            result["true_fitness"] = _rastrigin(best_solution)
            result["proxy_fitness"] = result["best_fitness"]
            result["best_fitness"] = result["true_fitness"]  # report true fitness

        else:
            raise ValueError(f"unknown cell: {ctx.cell!r}")

        # Normalize fitness: lower is better for Rastrigin (global min = 0)
        # Convert to "score" where higher is better
        max_possible = _rastrigin(np.full(dim, bounds[1]))  # worst case
        score = 1.0 - result["best_fitness"] / (max_possible + 1e-8)
        score = max(0.0, min(1.0, score))

        readouts = ReadoutBundle(
            metrics={
                "best_fitness": result["best_fitness"],
                "score": score,
                "generations": float(result.get("generations", 0)),
                "evals_used": float(result["evals_used"]),
            },
            artifacts={
                "trajectory_tail": result["trajectory"][-10:],
                "best_solution": result["best_solution"][:3],  # first 3 dims
            },
            tags={"cell": ctx.cell.value, "seed": ctx.seed},
        )

        return RunOutcome(
            readouts=readouts,
            output={"cell": ctx.cell.value, "best_fitness": result["best_fitness"], "score": score},
        )

    def gate(self, outcomes: list[RunOutcome]) -> GateReport:
        if not outcomes:
            return GateReport(passed=False, reason="no outcomes", stats={})

        probe_on = [o for o in outcomes if o.readouts.tags.get("cell") == "probe_on"]
        baseline = [o for o in outcomes if o.readouts.tags.get("cell") == "baseline"]
        counterfactual = [o for o in outcomes if o.readouts.tags.get("cell") == "counterfactual"]

        if not probe_on or not baseline:
            return GateReport(passed=False, reason="missing cells", stats={})

        # Evolution should beat random search
        p_score = sum(o.readouts.metrics["score"] for o in probe_on) / len(probe_on)
        b_score = sum(o.readouts.metrics["score"] for o in baseline) / len(baseline)
        evolution_wins = p_score > b_score

        # Goodhart: evolution with wrong fitness should be WORSE than correct evolution
        if counterfactual:
            cf_score = sum(o.readouts.metrics["score"] for o in counterfactual) / len(counterfactual)
            goodhart_detected = cf_score < p_score
        else:
            goodhart_detected = True

        passed = evolution_wins and goodhart_detected
        return GateReport(
            passed=passed,
            reason=f"evolution={p_score:.4f} vs random={b_score:.4f}, goodhart_detected={goodhart_detected}",
            stats={"evolution_score": p_score, "random_score": b_score},
        )
