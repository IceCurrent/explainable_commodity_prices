"""Synthetic DGP recovery tests (lightweight; PCA or mocked AE)."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd

from eqcp.cca.kernel import kernel_canonical_correlations
from eqcp.cca.linear import canonical_correlations
from eqcp.factors.autoencoder import AETrainConfig
from eqcp.synthetic.recovery import SyntheticConfig, make_synthetic, run_recovery


def _driver_frame(data, prefix: str = "f") -> pd.DataFrame:
    cols = [f"{prefix}{i + 1}" for i in range(data.driver.shape[1])]
    return pd.DataFrame(data.driver, index=data.index, columns=cols)


def test_linear_rank_r_canon_min_near_one():
    """Linear rank-r DGP: driver space and true g share the same span."""
    cfg = SyntheticConfig(n_obs=300, true_rank=3, relation="linear", snr=12.0, seed=0)
    data = make_synthetic(cfg)
    cc = canonical_correlations(_driver_frame(data), data.g)
    assert float(cc.min()) > 0.99


def test_tanh_relation_kcca_exceeds_linear_canon():
    """Tanh DGP: kernel CCA sees more shared structure than linear CCA."""
    cfg = SyntheticConfig(n_obs=300, true_rank=2, relation="tanh", snr=8.0, seed=1)
    data = make_synthetic(cfg)
    f_df = _driver_frame(data)
    cc_lin = canonical_correlations(f_df, data.g)
    kcc = kernel_canonical_correlations(f_df, data.g, top_k=2)
    assert float(kcc.min()) > float(cc_lin.min()) + 0.02
    assert float(cc_lin.min()) < 0.98


def test_tanh_dgp_driver_is_nonlinear():
    cfg = SyntheticConfig(n_obs=100, true_rank=2, relation="tanh", seed=2)
    data = make_synthetic(cfg)
    assert not np.allclose(data.driver, data.g.to_numpy(), atol=0.05)


def test_linear_ae_mock_avoids_heavy_training():
    cfg = SyntheticConfig(n_obs=60, n_inputs=8, true_rank=2, relation="linear", snr=10.0, seed=3)
    data = make_synthetic(cfg)

    def fake_encode(xz, n_factors, activation, seed):
        assert activation == "relu"
        return (
            data.driver[:, :n_factors],
            np.zeros((cfg.n_obs, cfg.n_inputs)),
            np.eye(cfg.n_inputs, n_factors),
        )

    with patch("eqcp.synthetic.recovery._encode_ae", side_effect=fake_encode):
        res = run_recovery(cfg, n_factors=2, encoder="relu")
    assert res.encoder == "relu"
    assert res.canon_min > 0.99
    assert res.n_active == 2


def test_small_relu_ae_completes_quickly():
    cfg = SyntheticConfig(n_obs=80, n_inputs=8, true_rank=2, relation="linear", snr=10.0, seed=4)
    fast = AETrainConfig(n_factors=2, epochs=3, patience=2, batch_size=32, seed=4)
    with patch("eqcp.synthetic.recovery.AETrainConfig", return_value=fast):
        res = run_recovery(cfg, n_factors=2, encoder="relu")
    assert res.n_active >= 1
    assert np.isfinite(res.canon_min)
