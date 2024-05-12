from .generate import generate
from .lightning_module import LightningModel
from .nanogpt import NanoGPT
from .pos_encoding import (
    AbsolutePositionalEncoding,
    CoordinateEncoding,
    RelativeMultiheadAttention,
)
from .transformer import Transformer
from .universal_transformer import UniversalTransformer
from .universal_transformer_decoder import UniversalTransformerDecoder

MODELS = {
    "NanoGPT": NanoGPT,
    "Transformer": Transformer,
    "UniversalTransformerDecoder": UniversalTransformerDecoder,
    "UniversalTransformer": UniversalTransformer,
}
