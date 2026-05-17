from .base import *
from .cvae import *

# Aliases
CVAE = ConditionalVAE
GaussianVAE = ConditionalVAE

vae_models = {
    'ConditionalVAE': ConditionalVAE
}