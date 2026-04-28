"""Pure-Python tensor operations for n-dimensional metacontroller computation.

All operations work on tuples of floats — no external dependencies.
Provides: matrix-vector multiply, GRU cell, 2-layer FFN, sigmoid, tanh,
element-wise ops, and random initialization.
"""

from __future__ import annotations

import math
import random


def sigmoid(x: float) -> float:
    if x > 20.0:
        return 1.0
    if x < -20.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def tanh(x: float) -> float:
    return math.tanh(x)


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


Vec = tuple[float, ...]
Mat = tuple[tuple[float, ...], ...]


def zeros(n: int) -> Vec:
    return tuple(0.0 for _ in range(n))


def ones(n: int) -> Vec:
    return tuple(1.0 for _ in range(n))


def rand_vec(n: int, *, scale: float = 0.1, seed: int | None = None) -> Vec:
    rng = random.Random(seed)
    return tuple(rng.gauss(0.0, scale) for _ in range(n))


def rand_mat(rows: int, cols: int, *, scale: float = 0.1, seed: int | None = None) -> Mat:
    rng = random.Random(seed)
    return tuple(
        tuple(rng.gauss(0.0, scale) for _ in range(cols))
        for _ in range(rows)
    )


def identity_mat(n: int) -> Mat:
    return tuple(
        tuple(1.0 if i == j else 0.0 for j in range(n))
        for i in range(n)
    )


def mat_vec(matrix: Mat, vector: Vec) -> Vec:
    return tuple(
        sum(w * v for w, v in zip(row, vector))
        for row in matrix
    )


def vec_add(a: Vec, b: Vec) -> Vec:
    return tuple(x + y for x, y in zip(a, b))


def vec_sub(a: Vec, b: Vec) -> Vec:
    return tuple(x - y for x, y in zip(a, b))


def vec_mul(a: Vec, b: Vec) -> Vec:
    """Element-wise multiplication (Hadamard product)."""
    return tuple(x * y for x, y in zip(a, b))


def vec_scale(a: Vec, s: float) -> Vec:
    return tuple(x * s for x in a)


def vec_clamp(a: Vec, lo: float = 0.0, hi: float = 1.0) -> Vec:
    return tuple(clamp(x, lo, hi) for x in a)


def vec_sigmoid(a: Vec) -> Vec:
    return tuple(sigmoid(x) for x in a)


def vec_tanh(a: Vec) -> Vec:
    return tuple(tanh(x) for x in a)


def vec_abs(a: Vec) -> Vec:
    return tuple(abs(x) for x in a)


def vec_mean(a: Vec) -> float:
    return sum(a) / max(len(a), 1)


def vec_max(a: Vec) -> float:
    return max(a) if a else 0.0


def vec_norm(a: Vec) -> float:
    return math.sqrt(sum(x * x for x in a))


def dot(a: Vec, b: Vec) -> float:
    return sum(x * y for x, y in zip(a, b))


# --- GRU cell ---

def gru_cell(
    *,
    x: Vec,
    h_prev: Vec,
    W_z: Mat, U_z: Mat, b_z: Vec,
    W_r: Mat, U_r: Mat, b_r: Vec,
    W_h: Mat, U_h: Mat, b_h: Vec,
) -> Vec:
    """One step of a GRU cell: x is input, h_prev is previous hidden state."""
    n = len(h_prev)
    z_gate = vec_sigmoid(vec_add(vec_add(mat_vec(W_z, x), mat_vec(U_z, h_prev)), b_z))
    r_gate = vec_sigmoid(vec_add(vec_add(mat_vec(W_r, x), mat_vec(U_r, h_prev)), b_r))
    h_candidate = vec_tanh(
        vec_add(
            vec_add(mat_vec(W_h, x), mat_vec(U_h, vec_mul(r_gate, h_prev))),
            b_h,
        )
    )
    one_minus_z = tuple(1.0 - z for z in z_gate)
    return vec_add(vec_mul(one_minus_z, h_prev), vec_mul(z_gate, h_candidate))


# --- 2-layer FFN ---

def ffn_2layer(
    *,
    x: Vec,
    W1: Mat, b1: Vec,
    W2: Mat, b2: Vec,
) -> Vec:
    """2-layer feed-forward network with tanh activation after layer 1."""
    hidden = vec_tanh(vec_add(mat_vec(W1, x), b1))
    return vec_add(mat_vec(W2, hidden), b2)


# --- Parameter initialization helpers ---

def init_gru_params(
    input_dim: int,
    hidden_dim: int,
    *,
    seed: int = 42,
) -> dict[str, Mat | Vec]:
    """Initialize GRU parameters with Xavier-like scaling."""
    scale = 1.0 / math.sqrt(max(input_dim + hidden_dim, 1))
    s = seed
    return {
        "W_z": rand_mat(hidden_dim, input_dim, scale=scale, seed=s),
        "U_z": rand_mat(hidden_dim, hidden_dim, scale=scale, seed=s + 1),
        "b_z": zeros(hidden_dim),
        "W_r": rand_mat(hidden_dim, input_dim, scale=scale, seed=s + 2),
        "U_r": rand_mat(hidden_dim, hidden_dim, scale=scale, seed=s + 3),
        "b_r": zeros(hidden_dim),
        "W_h": rand_mat(hidden_dim, input_dim, scale=scale, seed=s + 4),
        "U_h": rand_mat(hidden_dim, hidden_dim, scale=scale, seed=s + 5),
        "b_h": zeros(hidden_dim),
    }


def init_ffn_params(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    *,
    seed: int = 42,
) -> dict[str, Mat | Vec]:
    """Initialize 2-layer FFN parameters."""
    scale1 = 1.0 / math.sqrt(max(input_dim, 1))
    scale2 = 1.0 / math.sqrt(max(hidden_dim, 1))
    return {
        "W1": rand_mat(hidden_dim, input_dim, scale=scale1, seed=seed),
        "b1": zeros(hidden_dim),
        "W2": rand_mat(output_dim, hidden_dim, scale=scale2, seed=seed + 1),
        "b2": zeros(output_dim),
    }
