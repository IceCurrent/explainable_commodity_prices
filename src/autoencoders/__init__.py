from .vanilla import VanillaAE
from .beta_vae import BetaVAE

ALL_ARCHITECTURES = [VanillaAE, BetaVAE]

__all__ = ["VanillaAE", "BetaVAE", "ALL_ARCHITECTURES"]
