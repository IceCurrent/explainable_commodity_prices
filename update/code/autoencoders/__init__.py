from .vanilla import VanillaAE
from .denoising import DenoisingAE
from .sparse import SparseAE
from .contractive import ContractiveAE
from .vae import VAE
from .beta_vae import BetaVAE
from .recurrent import RecurrentAE
from .masked import MaskedAE

ALL_ARCHITECTURES = [
    VanillaAE,
    DenoisingAE,
    SparseAE,
    ContractiveAE,
    VAE,
    BetaVAE,
    RecurrentAE,
    MaskedAE,
]

__all__ = [cls.__name__ for cls in ALL_ARCHITECTURES] + ["ALL_ARCHITECTURES"]
