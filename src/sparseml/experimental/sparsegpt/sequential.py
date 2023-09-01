from typing import List, Optional

import torch

from sparseml.experimental.sparsegpt.layer_compressor import LayerCompressor
from sparseml.experimental.sparsegpt.model_preprocessor import ModelPreProcessor


class SequentialSparseGPT:
    def __init__(
        self,
        model,
        recipe: Optional[str] = None,
        model_preprocessors: Optional[List[ModelPreProcessor]] = None,
        bottom_compressor: Optional[LayerCompressor] = None,
        head_compressor: Optional[LayerCompressor] = None,
    ):
        self.model = model
        self.model_preprocessors = model_preprocessors
        self.bottom_compressor = bottom_compressor
        self.head_compressor = head_compressor
        self.recipe = recipe
        self.manager = None
        self.compressible_layers = self.compressible_layers()

    def compressible_layers(self):
        """
        Derived class could override
        """
        try:
            return self.model.model.decoders.layers
        except:
            raise RuntimeError(
                "Derived class should override to provide list of compressible layers"
            )

    def pre_compress(self, dev: str = "cuda:0", **kwargs):
        model = self.model
        for processor in self.model_preprocessors:
            model, extras = processor.pre_process(model, dev=dev, **kwargs)
            kwargs.update(extras)
        return model, kwargs

    def compress(self, dev: str = "cuda:0", **kwargs):

        self.model, kwargs = self.pre_compress(**kwargs)

        # Step 0: BottomCompressor accomplishes two things:
        # 1) Compress the embedding if needed
        # 2) Pass the calibration data through the (compressed) bottom part of the network, capturing the outputs
        # which will become the inputs to the first decoder layer
        # Also return attention_mask as part of kwargs
        self.model, extras = self.bottom_compressor.compress(dev=dev, **kwargs)
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
        head_compressor = LayerCompressor(model.head, inputs)
        head_compressor.pre_compress()
        head_compressor.compress()
        model, extras = head_compressor.post_compress()

        return model, extras

    def post_compress(self, **kwargs):
        use_cache = kwargs["use_cache"]
        self.model.apply(torch.quantization.disable_observer)
        self.model.config.use_cache = use_cache

        return (self.model,)
