"""Beta-VAE: VAE with an elevated KL weight for disentanglement."""

from .vae import VAE


class BetaVAE(VAE):
    """VAE with beta = 4 (the canonical Higgins et al. setting).

    Caveat documented for the research record: with only ~252 observations
    per window, a strong KL weight can drive partial posterior collapse
    (factors shrinking toward the prior mean). If BetaVAE matches the plain
    AR(1) benchmark in the results, collapse is the likely mechanism -- that
    is itself an informative finding for architecture selection, not a bug.
    """

    name = "BetaVAE"

    def __init__(self, *args, beta: float = 4.0, **kwargs):
        super().__init__(*args, beta=beta, **kwargs)
