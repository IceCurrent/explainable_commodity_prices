"""Lead/lag canonical correlation scan."""

from __future__ import annotations

import numpy as np

from eqcp.cca.linear import _as_array, linear_cca_full


def leadlag_scan(F, M, lags, ridge: float = 0.0) -> dict[int, float]:
    Fa, Ma = _as_array(F), _as_array(M)
    T = Fa.shape[0]
    out: dict[int, float] = {}
    for lag in lags:
        if lag == 0:
            fi, mi = Fa, Ma
        elif lag > 0:
            fi, mi = Fa[lag:], Ma[: T - lag]
        else:
            fi, mi = Fa[: T + lag], Ma[-lag:]
        if fi.shape[0] <= Fa.shape[1] + 1:
            out[int(lag)] = np.nan
            continue
        out[int(lag)] = float(linear_cca_full(fi, mi, ridge=ridge)[0].mean())
    return out
