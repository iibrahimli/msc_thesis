from .generate import generate
from .lightning_module import LightningModel
from .pos_encoding import (
    AbsolutePositionalEncoding,
    CoordinateEncoding,
    RelativeMultiheadAttention,
)
from .transformer import Transformer
from .transformer_decoder import TransformerDecoder
from .universal_transformer import UniversalTransformer
from .universal_transformer_decoder import UniversalTransformerDecoder

MODELS = {
    "TransformerDecoder": TransformerDecoder,
    "Transformer": Transformer,
    "UniversalTransformerDecoder": UniversalTransformerDecoder,
    "UniversalTransformer": UniversalTransformer,
}
