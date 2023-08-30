import logging
from typing import Optional

import torch

from sparseml.pytorch.sparsification.modifier import PyTorchModifierYAML, Modifier

_LOGGER = logging.getLogger(__name__)

@PyTorchModifierYAML()
class SparseGPTModifier(Modifier):
    """
    """
    def __init__(
        self,
        sparsity: float = 0.5,
        block_size: int = 128,
        quantize: bool = True,
        num_bits: int = 16,
        dampening_frac: Optional[float] = 0.001,
        sequential_update: Optional[bool] = True,
    ):
        self._model_preprocessors = [] # filled in by child classes
        self._compressible_layers = None 
        self._model = None
        self._bottom_compressor = None
        self._head_compressor = None
        self._compressible_layers = None

        self._sparsity = sparsity
        self._block_size = block_size
        self._quantize = quantize
        self._num_bits = num_bits
        self._dampening_frac = dampening_frac
        self._sequential_update = sequential_update

        self.device="cuda:0"

    def compressible_layers(self):
        pass

    def bottom_compressor(self):
        pass

    def head_compressor(self):
        pass

    def one_shot(self, model, dataloader, **kwargs):
        self.initialize(model, **kwargs)
        self.compress(dataloader, **kwargs)
        self.finalize(**kwargs)
        
    def initialize(self, model, **kwargs):
        self.model = model
        self._compressible_layers = self.compressible_layers()
        self._bottom_compressor = self.bottom_compressor()
        self._head_compressor = self.head_compressor()

    def finalize(self, **kwargs):
        use_cache = kwargs["use_cache"]
        self.model.apply(torch.quantization.disable_observer)
        self.model.config.use_cache = use_cache