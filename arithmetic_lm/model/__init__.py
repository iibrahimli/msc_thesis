from .generate import generate
from .lightning_module import LightningModel
from .nanogpt import NanoGPT
from .pos_encoding import AbsolutePositionalEncoding, CoordinateEncoding
from .transformer import Transformer
from .universal_nanogpt import UniversalNanoGPT
from .universal_transformer import UniversalTransformer

MODELS = {
    "NanoGPT": NanoGPT,
    "Transformer": Transformer,
    "UniversalNanoGPT": UniversalNanoGPT,
    "UniversalTransformer": UniversalTransformer,
}
