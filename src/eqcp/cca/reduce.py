"""Bloc reduction for interpretable macro CCA."""

from __future__ import annotations

import numpy as np
import pandas as pd

from eqcp.cca.linear import _pearson


def bloc_reduce(M_df: pd.DataFrame, bloc_map: dict[str, list[str]]) -> pd.DataFrame:
    """One sign-fixed first PC per macro bloc."""
    cols_out: dict[str, np.ndarray] = {}
    for bloc, cols in bloc_map.items():
        present = [c for c in cols if c in M_df.columns]
        if not present:
            continue
        X = M_df[present].to_numpy(float)
        sd = X.std(0)
        Xs = (X - X.mean(0)) / np.where(sd == 0, 1.0, sd)
        U, S, _ = np.linalg.svd(Xs, full_matrices=False)
        pc = U[:, 0] * S[0]
        if _pearson(pc, Xs.mean(1)) < 0:
            pc = -pc
        cols_out[bloc] = pc
    return pd.DataFrame(cols_out, index=M_df.index)
