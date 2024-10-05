from .generate import generate
from .lightning_module import LightningModel
from .looped_decoder import LoopedDecoder
from .transformer import Transformer
from .transformer_decoder import TransformerDecoder
from .universal_transformer import UniversalTransformer
from .universal_transformer_decoder import UniversalTransformerDecoder
from .utils import find_latest_ckpt, load_model

MODELS = {
    "TransformerDecoder": TransformerDecoder,
    "Transformer": Transformer,
    "UniversalTransformerDecoder": UniversalTransformerDecoder,
    "UniversalTransformer": UniversalTransformer,
    "LoopedDecoder": LoopedDecoder,
}
