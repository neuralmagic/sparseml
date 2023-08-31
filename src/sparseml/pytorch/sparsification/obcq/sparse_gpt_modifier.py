import logging
from typing import Optional

import torch

from sparseml.pytorch.sparsification.modifier import PyTorchModifierYAML, ScheduledModifier
from sparseml.pytorch.sparsification.obcq.layer_compressor import LayerCompressor

_LOGGER = logging.getLogger(__name__)

__all__ = ["SparseGPTModifier"]

@PyTorchModifierYAML
class SparseGPTModifier(ScheduledModifier):
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

    def one_shot(self, model, dataloader, initializer_kwargs, finalize_kwargs):
        self.initialize(model, **initializer_kwargs)
        self.compress(dataloader)
        self.finalize(**finalize_kwargs)

    def compress(self, dataloader):
        kwargs = {}
        # Step 0: BottomCompressor accomplishes two things:
        # 1) Compress the embedding if needed
        # 2) Pass the calibration data through the (compressed) bottom part of the network, capturing the outputs
        # which will become the inputs to the first decoder layer
        # Also return attention_mask as part of kwargs
        self.model, extras = self._bottom_compressor.compress(dev=self.device, **kwargs)
        kwargs.update(extras)

        # Step 1: Sequentially prune/quantize decoder layers
        inputs = kwargs["outputs"]
        for idx, layer in enumerate(self.compressible_layers):
            layer_kwargs = kwargs.deepcopy()
            layer_compressor = LayerCompressor(
                self.model, layer, idx, inputs, self.manager, **layer_kwargs
            )

            # Set up SparseGPT object, compute Hessian
            self.model, layer_kwargs = layer_compressor.pre_compress(**layer_kwargs)

            # Joinly prune/quantize using SparseGPT
            self.model, layer_kwargs = layer_compressor.compress(**layer_kwargs)

            # Compute outputs given compressed layer, memory clean up etc
            (
                self.model,
                layer_kwargs,
            ) = layer_compressor.post_compress(**layer_kwargs)
            inputs = layer_kwargs["outputs"]

        # Step 2: Prune/quantize head
        # TODO: Need update here -- see MPT for head quantization example
        head_compressor = LayerCompressor(self.model.head, inputs)
        head_compressor.pre_compress()
        head_compressor.compress()
        self.model, extras = head_compressor.post_compress()
        
    def initialize(self, model, **kwargs):
        self.model = model
        self._compressible_layers = self.compressible_layers()
        self._bottom_compressor = self.bottom_compressor()
        self._head_compressor = self.head_compressor()

    def finalize(self, **kwargs):
        use_cache = kwargs["use_cache"]
        self.model.apply(torch.quantization.disable_observer)
        self.model.config.use_cache = use_cache