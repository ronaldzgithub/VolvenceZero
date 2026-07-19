"""P2 Latent Controller — GRPO token vs hidden-state RL probe (stage 1).

Hypothesis: RL in hidden-state space (latent controller) achieves better
sample efficiency and generalization than token-space RL (GRPO), because
the latent space is lower-dimensional and more structured.

Cells:
- baseline (token_grpo): standard GRPO in token space
- probe_on (hidden_grpo): GRPO in hidden-state space (latent controller)
- probe_off (random_policy): random actions (lower bound)
- counterfactual (oracle_policy): optimal policy (upper bound)

Eval: Synthetic bandit task with structured latent space.
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


def _generate_real_bandit_task(seed: int, model_id: str = "sshleifer/tiny-gpt2", n_arms: int = 8, n_episodes: int = 50) -> dict:
    """Generate bandit task with real model hidden states as arm embeddings.

    Each "arm" is a short text prompt; the arm embedding is the model's
    hidden state representation. This creates a structured latent space
    that hidden-state RL can exploit but token-space RL cannot.
    """
    try:
        from ...framework.runtime import get_model_runtime

        arm_texts = [
            "Choose the red door",
            "Pick the blue option",
            "Select the green path",
            "Take the yellow route",
            "Go through the purple gate",
            "Enter the orange portal",
            "Follow the silver trail",
            "Cross the golden bridge",
            "Open the crystal box",
            "Touch the diamond key",
            "Pull the iron lever",
            "Push the wooden button",
            "Turn the copper dial",
            "Flip the bronze switch",
            "Spin the platinum wheel",
            "Twist the titanium knob",
        ][:n_arms]

        rt = get_model_runtime(model_id, dtype="fp32")
        result = rt.encode_text(arm_texts, max_length=32)
        arm_embeddings = result["embeddings"].numpy().astype(np.float32)
        latent_dim = arm_embeddings.shape[1]

        # Reward function: linear in the real latent space
        rng = np.random.default_rng(seed)
        reward_weights = rng.standard_normal(latent_dim).astype(np.float32)
        true_rewards = arm_embeddings @ reward_weights
        true_rewards = (true_rewards - true_rewards.min()) / (true_rewards.max() - true_rewards.min() + 1e-8)

        # Context varies per episode
        contexts = rng.standard_normal((n_episodes, latent_dim)).astype(np.float32) * 0.3

        return {
            "n_arms": n_arms,
            "latent_dim": latent_dim,
            "n_episodes": n_episodes,
            "arm_embeddings": arm_embeddings.tolist(),
            "true_rewards": true_rewards.tolist(),
            "contexts": contexts.tolist(),
            "reward_weights": reward_weights.tolist(),
            "model_id": model_id,
            "model_sha": rt.model_sha,
            "source": "real",
        }
    except Exception as e:
        result = _generate_bandit_task(seed=seed, n_arms=n_arms, n_episodes=n_episodes)
        result["source"] = "synthetic_fallback"
        result["fallback_reason"] = str(e)
        return result


def _generate_bandit_task(seed: int, n_arms: int = 8, latent_dim: int = 4, n_episodes: int = 50) -> dict:
    """Generate a structured bandit task with latent space.

    Arms have rewards determined by a low-rank structure in latent space.
    Token-space RL must discover this structure from scratch;
    hidden-state RL can exploit it directly.
    """
    rng = np.random.default_rng(seed)

    # Latent structure: arms are embedded in low-dim space
    arm_embeddings = rng.standard_normal((n_arms, latent_dim)).astype(np.float32)

    # Reward function: linear in latent space + noise
    reward_weights = rng.standard_normal(latent_dim).astype(np.float32)
    true_rewards = arm_embeddings @ reward_weights
    # Normalize to [0, 1]
    true_rewards = (true_rewards - true_rewards.min()) / (true_rewards.max() - true_rewards.min() + 1e-8)

    # Context varies per episode (shifts optimal arm)
    contexts = rng.standard_normal((n_episodes, latent_dim)).astype(np.float32) * 0.3

    return {
        "n_arms": n_arms,
        "latent_dim": latent_dim,
        "n_episodes": n_episodes,
        "arm_embeddings": arm_embeddings.tolist(),
        "reward_weights": reward_weights.tolist(),
        "true_rewards": true_rewards.tolist(),
        "contexts": contexts.tolist(),
    }


def _run_token_grpo(
    arm_embeddings: np.ndarray,
    reward_weights: np.ndarray,
    contexts: np.ndarray,
    n_arms: int,
    seed: int,
    learning_rate: float = 0.1,
) -> dict[str, Any]:
    """Simulate token-space GRPO: learns arm preferences without latent structure."""
    rng = np.random.default_rng(seed)
    n_episodes = len(contexts)

    # Token-space policy: softmax over arm logits (no latent awareness)
    logits = np.zeros(n_arms, dtype=np.float32)
    rewards_collected = []
    regrets = []

    for ep in range(n_episodes):
        # Context-dependent true rewards
        ctx_rewards = arm_embeddings @ (reward_weights + contexts[ep])
        ctx_rewards = (ctx_rewards - ctx_rewards.min()) / (ctx_rewards.max() - ctx_rewards.min() + 1e-8)
        optimal_reward = ctx_rewards.max()

        # Sample action from policy
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        action = rng.choice(n_arms, p=probs)

        # Get reward
        reward = ctx_rewards[action] + rng.normal(0, 0.1)
        rewards_collected.append(float(reward))
        regrets.append(float(optimal_reward - ctx_rewards[action]))

        # GRPO update: reinforce the chosen action proportional to reward
        logits[action] += learning_rate * reward

    return {
        "mean_reward": float(np.mean(rewards_collected)),
        "mean_regret": float(np.mean(regrets)),
        "final_reward": float(np.mean(rewards_collected[-10:])),
        "cumulative_regret": float(np.sum(regrets)),
    }


def _run_hidden_grpo(
    arm_embeddings: np.ndarray,
    reward_weights: np.ndarray,
    contexts: np.ndarray,
    n_arms: int,
    latent_dim: int,
    seed: int,
    learning_rate: float = 0.1,
) -> dict[str, Any]:
    """Simulate hidden-state GRPO: learns in latent space (more efficient)."""
    rng = np.random.default_rng(seed)
    n_episodes = len(contexts)

    # Latent-space policy: learns reward_weights directly
    learned_weights = np.zeros(latent_dim, dtype=np.float32)
    rewards_collected = []
    regrets = []
    

    for ep in range(n_episodes):
        ctx_rewards = arm_embeddings @ (reward_weights + contexts[ep])
        ctx_rewards = (ctx_rewards - ctx_rewards.min()) / (ctx_rewards.max() - ctx_rewards.min() + 1e-8)
        optimal_reward = ctx_rewards.max()

        # Predict rewards using learned weights
        predicted_rewards = arm_embeddings @ (learned_weights + contexts[ep])
        # Softmax selection (guard against NaN from large values)
        predicted_rewards = np.nan_to_num(predicted_rewards, nan=0.0, posinf=100.0, neginf=-100.0)
        shifted = predicted_rewards - predicted_rewards.max()
        probs = np.exp(np.clip(shifted, -50, 0))
        probs /= probs.sum() + 1e-8
        probs = np.clip(probs, 1e-8, None)
        probs /= probs.sum()
        action = rng.choice(n_arms, p=probs)

        reward = ctx_rewards[action] + rng.normal(0, 0.1)
        rewards_collected.append(float(reward))
        regrets.append(float(optimal_reward - ctx_rewards[action]))

        # Update learned weights toward true reward direction
        error = reward - (arm_embeddings[action] @ learned_weights)
        learned_weights += learning_rate * error * arm_embeddings[action]

    return {
        "mean_reward": float(np.mean(rewards_collected)),
        "mean_regret": float(np.mean(regrets)),
        "final_reward": float(np.mean(rewards_collected[-10:])),
        "cumulative_regret": float(np.sum(regrets)),
    }


@register_probe
class LatentControllerProbe(BaseProbe):
    id = "latent-controller-v1"
    hypothesis = (
        "RL in hidden-state space (latent controller) achieves better sample efficiency "
        "than token-space GRPO on structured bandit tasks."
    )
    primitive = PrimitiveTag.P2_LATENT_CONTROLLER
    r_ids = ("R3", "R4")

    def knobs(self) -> dict[str, list]:
        return {
            "learning_rate": [0.05, 0.1, 0.2],
            "n_arms": [8, 16],
            "use_real_model": [False, True],
            "model_id": ["sshleifer/tiny-gpt2"],
        }

    def default_inputs(self, seed: int) -> Any:
        return _generate_bandit_task(seed=seed)

    def real_inputs(self, seed: int, knobs: Mapping[str, Any]) -> Any:
        """Generate bandit task with real model hidden states as arm embeddings."""
        model_id = knobs.get("model_id", "sshleifer/tiny-gpt2")
        return _generate_real_bandit_task(seed=seed, model_id=model_id)

    def run_cell(self, ctx: ProbeContext, knobs: Mapping[str, Any]) -> RunOutcome:
        inputs = ctx.inputs
        arm_embeddings = np.array(inputs["arm_embeddings"], dtype=np.float32)
        reward_weights = np.array(inputs["reward_weights"], dtype=np.float32)
        contexts = np.array(inputs["contexts"], dtype=np.float32)
        n_arms = inputs["n_arms"]
        latent_dim = inputs["latent_dim"]
        lr = knobs.get("learning_rate", 0.1)

        if ctx.cell == AblationCell.BASELINE:
            result = _run_token_grpo(arm_embeddings, reward_weights, contexts, n_arms, ctx.seed, lr)

        elif ctx.cell == AblationCell.PROBE_ON:
            result = _run_hidden_grpo(arm_embeddings, reward_weights, contexts, n_arms, latent_dim, ctx.seed, lr)

        elif ctx.cell == AblationCell.PROBE_OFF:
            # Random policy
            rng = np.random.default_rng(ctx.seed + 2222)
            n_episodes = len(contexts)
            rewards_collected = []
            regrets = []
            for ep in range(n_episodes):
                ctx_rewards = arm_embeddings @ (reward_weights + contexts[ep])
                ctx_rewards = (ctx_rewards - ctx_rewards.min()) / (ctx_rewards.max() - ctx_rewards.min() + 1e-8)
                action = rng.integers(0, n_arms)
                reward = ctx_rewards[action] + rng.normal(0, 0.1)
                rewards_collected.append(float(reward))
                regrets.append(float(ctx_rewards.max() - ctx_rewards[action]))
            result = {
                "mean_reward": float(np.mean(rewards_collected)),
                "mean_regret": float(np.mean(regrets)),
                "final_reward": float(np.mean(rewards_collected[-10:])),
                "cumulative_regret": float(np.sum(regrets)),
            }

        elif ctx.cell == AblationCell.COUNTERFACTUAL:
            # Oracle: always pick best arm
            n_episodes = len(contexts)
            rewards_collected = []
            rng = np.random.default_rng(ctx.seed + 4444)
            for ep in range(n_episodes):
                ctx_rewards = arm_embeddings @ (reward_weights + contexts[ep])
                ctx_rewards = (ctx_rewards - ctx_rewards.min()) / (ctx_rewards.max() - ctx_rewards.min() + 1e-8)
                reward = ctx_rewards.max() + rng.normal(0, 0.1)
                rewards_collected.append(float(reward))
            result = {
                "mean_reward": float(np.mean(rewards_collected)),
                "mean_regret": 0.0,
                "final_reward": float(np.mean(rewards_collected[-10:])),
                "cumulative_regret": 0.0,
            }
        else:
            raise ValueError(f"unknown cell: {ctx.cell!r}")

        readouts = ReadoutBundle(
            metrics={
                "mean_reward": result["mean_reward"],
                "mean_regret": result["mean_regret"],
                "final_reward": result["final_reward"],
                "cumulative_regret": result["cumulative_regret"],
            },
            artifacts={"result": result},
            tags={"cell": ctx.cell.value, "seed": ctx.seed},
        )

        return RunOutcome(
            readouts=readouts,
            output={"cell": ctx.cell.value, "mean_reward": result["mean_reward"]},
        )

    def gate(self, outcomes: list[RunOutcome]) -> GateReport:
        if not outcomes:
            return GateReport(passed=False, reason="no outcomes", stats={})

        baseline = [o for o in outcomes if o.readouts.tags.get("cell") == "baseline"]
        probe_on = [o for o in outcomes if o.readouts.tags.get("cell") == "probe_on"]

        if not baseline or not probe_on:
            return GateReport(passed=False, reason="missing baseline or probe_on", stats={})

        b_reward = sum(o.readouts.metrics["mean_reward"] for o in baseline) / len(baseline)
        p_reward = sum(o.readouts.metrics["mean_reward"] for o in probe_on) / len(probe_on)

        passed = p_reward > b_reward
        return GateReport(
            passed=passed,
            reason=f"hidden_grpo reward={p_reward:.4f} vs token_grpo={b_reward:.4f}",
            stats={"baseline_reward": b_reward, "probe_on_reward": p_reward},
        )
