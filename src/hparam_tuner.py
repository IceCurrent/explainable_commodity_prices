# src/evaluation/hparam_tuner.py

from __future__ import annotations

import itertools
import random
import numpy as np

from dataclasses import dataclass
from typing import Any

from .hparam_cv import walk_forward_cv


@dataclass
class TuningResult:
    best_params: dict[str, Any]
    best_score: float
    all_results: list[dict]


def grid_search(
    X: np.ndarray,
    arch_cls,
    param_grid: dict[str, list],
    fixed_params: dict[str, Any] | None = None,
    n_splits: int = 5,
    val_size: int = 21,
    gap: int = 5,
    train_window: int = 252,
    seed: int = 42,
    verbose: bool = True,
) -> TuningResult:
    """
    Exhaustive walk-forward grid search.
    """

    fixed = fixed_params or {}

    keys = list(param_grid.keys())
    values = list(param_grid.values())

    combinations = list(itertools.product(*values))

    results = []

    print(f"\nGrid Search: {len(combinations)} combinations\n")

    for idx, combo in enumerate(combinations, start=1):

        params = dict(zip(keys, combo))
        all_params = {**fixed, **params}

        def factory(p=all_params):
            return arch_cls(
                n_inputs=X.shape[1],
                seed=seed,
                **p
            )

        try:

            score = walk_forward_cv(
                X=X,
                factory=factory,
                n_splits=n_splits,
                val_size=val_size,
                gap=gap,
                train_window=train_window,
            )

        except Exception as e:

            score = np.inf

            if verbose:
                print(
                    f"[{idx}/{len(combinations)}] "
                    f"{params} -> FAILED: {e}"
                )

        results.append(
            {
                "params": params,
                "score": score,
            }
        )

        if verbose and np.isfinite(score):
            print(
                f"[{idx}/{len(combinations)}] "
                f"{params} -> "
                f"MSE={score:.8f}"
            )

    results.sort(key=lambda x: x["score"])

    return TuningResult(
        best_params=results[0]["params"],
        best_score=float(results[0]["score"]),
        all_results=results,
    )


def random_search(
    X: np.ndarray,
    arch_cls,
    param_distributions: dict[str, list],
    n_iter: int = 40,
    fixed_params: dict[str, Any] | None = None,
    n_splits: int = 5,
    val_size: int = 21,
    gap: int = 5,
    train_window: int = 252,
    seed: int = 42,
    verbose: bool = True,
) -> TuningResult:
    """
    Random-search walk-forward tuning.
    Preferred for large AE search spaces.
    """

    fixed = fixed_params or {}

    rng = random.Random(seed)

    results = []

    print(f"\nRandom Search: {n_iter} trials\n")

    for idx in range(1, n_iter + 1):

        params = {
            k: rng.choice(v)
            for k, v in param_distributions.items()
        }

        all_params = {**fixed, **params}

        def factory(p=all_params):
            return arch_cls(
                n_inputs=X.shape[1],
                seed=seed,
                **p
            )

        try:

            score = walk_forward_cv(
                X=X,
                factory=factory,
                n_splits=n_splits,
                val_size=val_size,
                gap=gap,
                train_window=train_window,
            )

        except Exception as e:

            score = np.inf

            if verbose:
                print(
                    f"[{idx}/{n_iter}] "
                    f"{params} -> FAILED: {e}"
                )

        results.append(
            {
                "params": params,
                "score": score,
            }
        )

        if verbose and np.isfinite(score):
            print(
                f"[{idx}/{n_iter}] "
                f"{params} -> "
                f"MSE={score:.8f}"
            )

    results.sort(key=lambda x: x["score"])

    return TuningResult(
        best_params=results[0]["params"],
        best_score=float(results[0]["score"]),
        all_results=results,
    )