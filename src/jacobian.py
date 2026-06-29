"""E3: decoder-Jacobian commodity fingerprint -> macro hypothesis.

The column J_{.k} = dx/df_k of the decoder is a loading vector of factor k over
the 22 commodities. Because the decoder is a single linear layer, the Jacobian
is *exactly* the decoder weight matrix and is constant in f (no sampling needed).

This module averages the aligned decoder loadings across rolling windows,
inspects the top-loading commodities per factor, summarizes the loading mass by
sector, and maps the dominant sector to its canonical macro hypothesis. The
output is the narrative/triangulation layer that cross-checks E1/E2: does the
macro a factor regresses onto match the sector its Jacobian points to?
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.factor_alignment import Alignment
from src.factor_extraction import FactorBundle

# Refined sector taxonomy for the 22-commodity universe. Distinguishes base vs
# precious metals because the directive maps them to different macro factors.
COMMODITY_SECTORS: dict[str, str] = {
    # Energy
    "Brent": "Energy", "WTI": "Energy", "NaturalGas": "Energy", "Gasoline": "Energy",
    "Diesel": "Energy", "HeatingOil": "Energy", "ThermalCoal": "Energy", "Methanol": "Energy",
    # Precious metals
    "Gold": "PreciousMetals", "Silver": "PreciousMetals", "Platinum": "PreciousMetals",
    # Base / industrial metals
    "Copper": "BaseMetals", "Aluminium": "BaseMetals", "Nickel": "BaseMetals",
    "Zinc": "BaseMetals", "HRCSteel": "BaseMetals", "SGXIronOre": "BaseMetals",
    "Lithium": "BaseMetals",
    # Agriculture
    "Coffee": "Agriculture", "Corn": "Agriculture", "Cotton": "Agriculture",
    "LeanHogs": "Agriculture", "LiveCattle": "Agriculture", "Sugar": "Agriculture",
    "Soybeans": "Agriculture", "SoybeanMeal": "Agriculture", "SoybeanOil": "Agriculture",
    "HRWWheat": "Agriculture", "Wheat": "Agriculture",
}

# Canonical sector -> macro hypothesis (directive E3 step 3).
SECTOR_MACRO_HYPOTHESIS: dict[str, str] = {
    "Energy": "Kilian (2009) aggregate-demand & oil-specific supply shocks",
    "BaseMetals": "global industrial production, USD, real rates",
    "PreciousMetals": "real rates (TIPS), USD, risk-off (VIX)",
    "Agriculture": "USD and supply/weather (typically weak macro link)",
}


@dataclass
class JacobianFingerprint:
    commodities: list[str]
    factors: list[str]
    loadings: np.ndarray            # (N, K) mean aligned decoder loadings
    loadings_normalized: np.ndarray  # (N, K) unit-norm columns
    sector_mass: pd.DataFrame       # (sector x K) share of |loading| per sector
    sector_intensity: pd.DataFrame  # (sector x K) mean |loading| per commodity
    dominant_sector: dict[str, str]
    macro_hypothesis: dict[str, str]
    concentration: dict[str, float]  # top-2 |loading| share per factor

    def loadings_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(self.loadings, index=self.commodities, columns=self.factors)
        df.insert(0, "sector", [COMMODITY_SECTORS.get(c, "Unknown") for c in self.commodities])
        df.index.name = "commodity"
        return df

    def top_loadings(self, k: int, n: int = 5) -> pd.DataFrame:
        col = self.factors[k]
        s = pd.Series(self.loadings[:, k], index=self.commodities)
        top = s.reindex(s.abs().sort_values(ascending=False).index).head(n)
        return pd.DataFrame({
            "commodity": top.index,
            "loading": top.values,
            "sector": [COMMODITY_SECTORS.get(c, "Unknown") for c in top.index],
        })


def aligned_mean_jacobian(bundle: FactorBundle, alignment: Alignment) -> np.ndarray:
    """Mean decoder Jacobian across windows, aligned to the reference identity.

    Returns (N, K). Sign/permutation from E0 are applied so factor k is the same
    object in every window before averaging.
    """
    W, N, K = bundle.decoder_weight.shape
    acc = np.zeros((N, K))
    for w in range(W):
        aligned = bundle.decoder_weight[w][:, alignment.perm[w]] * alignment.sign[w]
        acc += aligned
    return acc / W


def compute_fingerprint(bundle: FactorBundle, alignment: Alignment) -> JacobianFingerprint:
    loadings = aligned_mean_jacobian(bundle, alignment)
    N, K = loadings.shape
    factors = [f"f{k+1}" for k in range(K)]

    norms = np.linalg.norm(loadings, axis=0)
    norms = np.where(norms == 0, 1.0, norms)
    loadings_norm = loadings / norms

    abs_load = np.abs(loadings)
    sectors = [COMMODITY_SECTORS.get(c, "Unknown") for c in bundle.commodities]
    sector_df = pd.DataFrame(abs_load, index=sectors, columns=factors)
    sector_mass = sector_df.groupby(level=0).sum()
    sector_share = sector_mass / sector_mass.sum(axis=0)
    # Mean |loading| per commodity in each sector: removes the size bias of
    # Agriculture (11 of 22 names) so dominance reflects intensity, not count.
    sector_intensity = sector_df.groupby(level=0).mean()

    dominant_sector: dict[str, str] = {}
    macro_hypothesis: dict[str, str] = {}
    concentration: dict[str, float] = {}
    for k, f in enumerate(factors):
        dom = sector_intensity[f].idxmax()
        dominant_sector[f] = dom
        macro_hypothesis[f] = SECTOR_MACRO_HYPOTHESIS.get(dom, "n/a")
        col_abs = np.sort(abs_load[:, k])[::-1]
        concentration[f] = float(col_abs[:2].sum() / max(col_abs.sum(), 1e-12))

    return JacobianFingerprint(
        commodities=bundle.commodities,
        factors=factors,
        loadings=loadings,
        loadings_normalized=loadings_norm,
        sector_mass=sector_share,
        sector_intensity=sector_intensity,
        dominant_sector=dominant_sector,
        macro_hypothesis=macro_hypothesis,
        concentration=concentration,
    )
