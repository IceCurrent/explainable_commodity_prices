"""E0: cross-window factor alignment and stability audit.

The AE is re-estimated on every rolling window, so its latent coordinates are
identified only up to permutation, sign, and reparametrization. "Factor 3" in
window w need not be "Factor 3" in window w+1. Before any macro regression can
pool across windows, the per-window factors must be matched to a common
identity. This module implements both anchors from the directive:

* **Anchor 1 (output correlation).** Match adjacent windows by maximizing total
  ``|corr|`` between their overlapping factor series via the Hungarian
  (linear-assignment) algorithm, fix signs by the matched correlation sign, and
  *chain* the adjacent transforms back to a reference window. Adjacent matching
  is used (not direct-to-reference) because distant windows do not overlap.

* **Anchor 2 (decoder geometry).** Match each window's decoder weight columns to
  the reference's by ``|cosine|`` similarity. This needs no temporal overlap, so
  it aligns every window directly and serves as the corroborating cross-check.

The agreement between the two anchors is itself a stability signal. The module
also reports the gate metrics (median adjacent ``|corr|``, sign-flip rate,
permutation frequency) and emits the aligned daily factor panel that E1-E4
consume.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from src.factor_extraction import FactorBundle


@dataclass
class Alignment:
    """Per-window permutation + sign mapping each window to a reference identity.

    For window ``w`` and reference factor ``r``:
        aligned_factor_r(w) = sign[w, r] * raw_factor_{perm[w, r]}(w)

    Attributes
    ----------
    perm : ndarray[int], shape (W, K)
        ``perm[w, r]`` = raw factor index in window w that maps to reference r.
    sign : ndarray[float], shape (W, K)
        Sign (+/-1) applied to that raw factor.
    match_quality : ndarray[float], shape (W,)
        For correlation anchor: mean matched ``|corr|`` of window w to window w-1
        (NaN for the reference). For cosine anchor: mean matched ``|cosine|`` to
        the reference.
    anchor : str
    reference : int
    """

    perm: np.ndarray
    sign: np.ndarray
    match_quality: np.ndarray
    anchor: str
    reference: int

    @property
    def n_windows(self) -> int:
        return self.perm.shape[0]

    @property
    def n_factors(self) -> int:
        return self.perm.shape[1]


def _corr_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """K x K Pearson correlation between columns of a and columns of b."""
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    a_norm = np.linalg.norm(a, axis=0)
    b_norm = np.linalg.norm(b, axis=0)
    a_norm = np.where(a_norm == 0, 1.0, a_norm)
    b_norm = np.where(b_norm == 0, 1.0, b_norm)
    return (a.T @ b) / np.outer(a_norm, b_norm)


def _cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """K x K cosine similarity between columns of a (N x K) and b (N x K)."""
    a_norm = np.linalg.norm(a, axis=0)
    b_norm = np.linalg.norm(b, axis=0)
    a_norm = np.where(a_norm == 0, 1.0, a_norm)
    b_norm = np.where(b_norm == 0, 1.0, b_norm)
    return (a.T @ b) / np.outer(a_norm, b_norm)


def _assign(score_abs: np.ndarray) -> np.ndarray:
    """Hungarian assignment maximizing total score. Returns col_for_row (len K)."""
    row, col = linear_sum_assignment(-score_abs)
    out = np.empty(score_abs.shape[0], dtype=np.int64)
    out[row] = col
    return out


def align_by_correlation(bundle: FactorBundle, reference: int = 0) -> Alignment:
    """Anchor 1: chained adjacent-window correlation matching."""
    W, M, K = bundle.factors.shape
    stride = int(bundle.config.get("stride", 1))
    overlap = M - stride
    if overlap <= 1:
        raise ValueError(
            f"stride ({stride}) leaves no usable overlap with window_size ({M}); "
            "correlation anchor needs overlapping windows."
        )

    perm = np.tile(np.arange(K), (W, 1))
    sign = np.ones((W, K))
    quality = np.full(W, np.nan)

    # Local adjacent transform: how window w+1's raw factors map to window w's
    # raw factor identities.
    local_match = np.tile(np.arange(K), (W, 1))   # local_match[w] maps w-index -> (w+1)-index
    local_sign = np.ones((W, K))
    for w in range(W - 1):
        a = bundle.factors[w, stride:, :]      # overlap rows in window w
        b = bundle.factors[w + 1, :overlap, :]  # same global days in window w+1
        c = _corr_matrix(a, b)                  # rows=w factors, cols=w+1 factors
        col_for_row = _assign(np.abs(c))
        local_match[w] = col_for_row
        local_sign[w] = np.sign(c[np.arange(K), col_for_row])
        local_sign[w][local_sign[w] == 0] = 1.0

    # Chain forward from reference to later windows.
    for w in range(reference + 1, W):
        prev_perm = perm[w - 1]
        prev_sign = sign[w - 1]
        m = local_match[w - 1]
        s = local_sign[w - 1]
        perm[w] = m[prev_perm]
        sign[w] = prev_sign * s[prev_perm]
        c = _corr_matrix(
            bundle.factors[w - 1, stride:, :], bundle.factors[w, : M - stride, :]
        )
        matched = np.abs(c[np.arange(K), local_match[w - 1]])
        quality[w] = float(np.mean(matched))

    # Chain backward from reference to earlier windows (if reference > 0).
    for w in range(reference - 1, -1, -1):
        # local transform maps window w's factors to window w+1's identity;
        # invert it to express window w in window (w+1)'s reference coordinates.
        next_perm = perm[w + 1]
        next_sign = sign[w + 1]
        m = local_match[w]          # w-index -> (w+1)-index
        s = local_sign[w]
        inv = np.empty(K, dtype=np.int64)
        inv[m] = np.arange(K)       # (w+1)-index -> w-index
        perm[w] = inv[next_perm]
        sign[w] = next_sign * s[inv[next_perm]]
        c = _corr_matrix(
            bundle.factors[w, stride:, :], bundle.factors[w + 1, : M - stride, :]
        )
        matched = np.abs(c[np.arange(K), local_match[w]])
        quality[w] = float(np.mean(matched))

    return Alignment(perm, sign, quality, anchor="correlation", reference=reference)


def align_by_decoder_cosine(bundle: FactorBundle, reference: int = 0) -> Alignment:
    """Anchor 2: match decoder weight columns to the reference by |cosine|."""
    W, N, K = bundle.decoder_weight.shape
    ref_W = bundle.decoder_weight[reference]
    perm = np.tile(np.arange(K), (W, 1))
    sign = np.ones((W, K))
    quality = np.full(W, np.nan)
    for w in range(W):
        c = _cosine_matrix(ref_W, bundle.decoder_weight[w])  # rows=ref, cols=w
        col_for_row = _assign(np.abs(c))
        perm[w] = col_for_row
        s = np.sign(c[np.arange(K), col_for_row])
        s[s == 0] = 1.0
        sign[w] = s
        quality[w] = float(np.mean(np.abs(c[np.arange(K), col_for_row])))
    return Alignment(perm, sign, quality, anchor="decoder_cosine", reference=reference)


def apply_alignment_live(bundle: FactorBundle, alignment: Alignment) -> np.ndarray:
    """Aligned live (last-row) factor per window, shape (W, K)."""
    W, K = alignment.perm.shape
    out = np.empty((W, K))
    live = bundle.live_factors()
    for w in range(W):
        out[w] = alignment.sign[w] * live[w, alignment.perm[w]]
    return out


def aligned_factor_panel(bundle: FactorBundle, alignment: Alignment) -> pd.DataFrame:
    """Daily aligned factor panel indexed by the factor's as-of date (E0 output)."""
    aligned = apply_alignment_live(bundle, alignment)
    df = pd.DataFrame(
        aligned,
        index=bundle.as_of_dates,
        columns=[f"f{r+1}" for r in range(alignment.n_factors)],
    )
    df.index.name = "date"
    return df


def anchor_agreement(corr_align: Alignment, cos_align: Alignment) -> dict:
    """Fraction of windows where both anchors imply the same factor identity.

    The permutations are compared after removing the global labelling: two
    alignments agree on window w if mapping the raw factors through one and then
    the inverse of the other yields the identity permutation.
    """
    W, K = corr_align.perm.shape
    per_window = np.zeros(W, dtype=bool)
    for w in range(W):
        inv = np.empty(K, dtype=np.int64)
        inv[cos_align.perm[w]] = np.arange(K)
        per_window[w] = np.array_equal(corr_align.perm[w], cos_align.perm[w])
    return {
        "agreement_rate": float(per_window.mean()),
        "per_window_agreement": per_window,
    }


def stability_report(
    bundle: FactorBundle,
    corr_align: Alignment,
    cos_align: Alignment | None = None,
    gate_stable: float = 0.8,
    gate_fail: float = 0.5,
) -> dict:
    """Compute the E0 gate metrics and a verdict.

    Returns a dict with the median adjacent ``|corr|``, per-factor sign-flip and
    permutation rates, anchor agreement, and a gate decision in
    {``stable``, ``within_window_only``, ``unstable_finding``}.
    """
    quality = corr_align.match_quality
    median_corr = float(np.nanmedian(quality))

    # Sign flips and permutation events between adjacent ALIGNED windows.
    W, K = corr_align.perm.shape
    sign_flips = np.zeros(K, dtype=int)
    perm_events = 0
    for w in range(1, W):
        if not np.array_equal(corr_align.perm[w], corr_align.perm[w - 1]):
            perm_events += 1
        flipped = corr_align.sign[w] != corr_align.sign[w - 1]
        sign_flips += flipped.astype(int)

    if median_corr >= gate_stable:
        verdict = "stable"
    elif median_corr < gate_fail:
        verdict = "unstable_finding"
    else:
        verdict = "within_window_only"

    report = {
        "n_windows": int(W),
        "n_factors": int(K),
        "median_adjacent_abs_corr": median_corr,
        "mean_adjacent_abs_corr": float(np.nanmean(quality)),
        "min_adjacent_abs_corr": float(np.nanmin(quality)),
        "sign_flip_rate_per_factor": (sign_flips / max(W - 1, 1)).tolist(),
        "permutation_event_rate": perm_events / max(W - 1, 1),
        "verdict": verdict,
        "gate_stable_threshold": gate_stable,
        "gate_fail_threshold": gate_fail,
    }
    if cos_align is not None:
        report["anchor_agreement_rate"] = anchor_agreement(corr_align, cos_align)["agreement_rate"]
        report["median_decoder_cosine"] = float(np.nanmedian(cos_align.match_quality))
    return report
