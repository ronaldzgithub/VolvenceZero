"""Feasibility smoke for the differentiable steered forward (plan step 1).

Loads the frozen Qwen2.5-0.5B-Instruct locally, injects a requires-grad
control delta at a hooked block via forward hook, keeps the upper blocks in
the autograd graph, computes an action-NLL over a restricted action-token
vocabulary, and backprops to the delta. Reports per-step wall time and
gradient sanity. No repository state is modified.
"""

from __future__ import annotations

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
INJECTION_LAYER = 20
ACTION_OPTIONS = ("alpha", "beta", "gamma", "delta", "epsilon", "hub")


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, local_files_only=True)
    model.to(DEVICE)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    blocks = model.model.layers
    hidden_size = int(model.config.hidden_size)
    print(f"device={DEVICE} blocks={len(blocks)} hidden={hidden_size}")

    candidate_ids = []
    for option in ACTION_OPTIONS:
        ids = tokenizer(" " + option, add_special_tokens=False)["input_ids"]
        candidate_ids.append(ids[0])
        print(f"option={option!r} first_token_id={ids[0]} n_tokens={len(ids)}")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise RuntimeError("candidate first tokens collide")

    prompts = [
        (
            "Task context: steady guidance alignment planning support corridor "
            "branch anchor. Available transitions: alpha, beta, delta, hub. "
            "Current location: entry. Remaining route: alpha -> beta -> delta."
            "\nNext move:"
        )
        for _ in range(8)
    ]
    encoded = tokenizer(prompts, return_tensors="pt", padding=True)
    encoded = {key: value.to(DEVICE) for key, value in encoded.items()}
    batch = encoded["input_ids"].shape[0]

    delta = torch.zeros(
        (batch, hidden_size), dtype=torch.float32, requires_grad=True
    )

    def hook(module, args, output):
        del module, args
        hidden = output[0] if isinstance(output, tuple) else output
        adjusted = hidden + delta.to(hidden.device, hidden.dtype).view(
            batch, 1, hidden_size
        )
        if isinstance(output, tuple):
            return (adjusted, *output[1:])
        return adjusted

    target_indices = torch.tensor([0, 1, 2, 3, 4, 5, 0, 1], device=DEVICE)
    candidate_tensor = torch.tensor(candidate_ids, device=DEVICE)

    timings = []
    for step in range(3):
        start = time.perf_counter()
        handle = blocks[INJECTION_LAYER].register_forward_hook(hook)
        try:
            outputs = model(**encoded, use_cache=False)
        finally:
            handle.remove()
        lengths = encoded["attention_mask"].sum(dim=-1) - 1
        last_logits = outputs.logits[
            torch.arange(batch, device=DEVICE), lengths
        ]
        candidate_logits = last_logits[:, candidate_tensor]
        log_probs = torch.log_softmax(candidate_logits, dim=-1)
        nll = -log_probs[torch.arange(batch, device=DEVICE), target_indices]
        loss = nll.mean()
        if delta.grad is not None:
            delta.grad = None
        loss.backward()
        elapsed = time.perf_counter() - start
        timings.append(elapsed)
        grad_norm = float(delta.grad.norm())
        print(
            f"step={step} loss={float(loss):.4f} grad_norm={grad_norm:.6f} "
            f"elapsed={elapsed:.2f}s"
        )
        if grad_norm <= 0.0:
            raise RuntimeError("no gradient reached the injected delta")

    if DEVICE == "mps":
        print(
            "mps allocated MB:",
            round(torch.mps.current_allocated_memory() / 1e6, 1),
        )
    print(f"mean step time: {sum(timings) / len(timings):.2f}s")

    with torch.no_grad():
        handle = blocks[INJECTION_LAYER].register_forward_hook(hook)
        try:
            outputs = model(**encoded, use_cache=False)
        finally:
            handle.remove()
    print("no_grad steered forward ok")

    hidden_norm_probe = {}

    def norm_hook(module, args, output):
        del module, args
        hidden = output[0] if isinstance(output, tuple) else output
        hidden_norm_probe["norm"] = float(
            hidden.norm(dim=-1).mean()
        )
        return None

    with torch.no_grad():
        handle = blocks[INJECTION_LAYER].register_forward_hook(norm_hook)
        try:
            model(**encoded, use_cache=False)
        finally:
            handle.remove()
    print(f"mean per-token hidden norm at layer {INJECTION_LAYER}: "
          f"{hidden_norm_probe['norm']:.2f}")


if __name__ == "__main__":
    main()
