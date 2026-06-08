"""pass@k + cross-file completion on held-out hyperswitch code.

pass@k uses the unbiased estimator from the Codex paper. Generation is delegated to a
vLLM endpoint; verification reuses the cargo harness (compile/test the completion).
"""
from __future__ import annotations

import math


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k: n samples, c correct, estimate probability >=1 of k passes."""
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def aggregate_pass_at_k(per_problem: list[tuple[int, int]], ks: list[int]) -> dict[int, float]:
    """per_problem = [(n_samples, n_correct), ...] -> {k: mean pass@k}."""
    out: dict[int, float] = {}
    for k in ks:
        out[k] = sum(pass_at_k(n, c, k) for n, c in per_problem) / max(1, len(per_problem))
    return out
