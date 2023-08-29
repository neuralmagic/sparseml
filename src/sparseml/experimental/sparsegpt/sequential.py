from typing import List, Optional
from copy import deepcopy

import torch

from layer_compressor import LayerCompressor
from model_preprocessor import ModelPreProcessor


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

    def pre_compress(self, args, dev: str = "cuda:0", **kwargs):
        model = self.model
        all_extras = {}
        for processor in self.model_preprocessors:
            # We assume the processors are independent, and therefore
            # pass in the initial kwargs into each of them
            model, extras = processor(model, dev=dev, **kwargs)
            all_extras.update(extras)
        return model, all_extras

    def compress(self, args, dev: str = "cuda:0", **kwargs):
        accum_kwargs = deepcopy(kwargs)

        import pdb; pdb.set_trace()
        self.model, extras = self.pre_compress(**kwargs)

        # Step 0: BottomCompressor accomplishes two things:
        # 1) Compress the embedding if needed
        # 2) Pass the calibration data through the (compressed) bottom part of the network, capturing the outputs
        # which will become the inputs to the first decoder layer
        # Also return attention_mask as part of kwargs
        accum_kwargs.update(extras)
        self.model, extras = self.bottom_compressor.compress(dev=dev, **accum_kwargs)
        accum_kwargs.update(extras)

        # Step 1: Sequentially prune/quantize decoder layers

        inputs = None
        for idx, layer in enumerate(self.compressible_layers):
            if "outputs" not in accum_kwargs:
                raise RuntimeError("The 'outputs' key is expected but not found from the "
                                   "return of the bottom compressor")
            inputs = accum_kwargs["outputs"]

            layer_compressor = LayerCompressor(
                self.model, layer, idx, inputs, self.manager, **accum_kwargs
            )

            # Set up SparseGPT object, compute Hessian
            self.model, layer_kwargs = layer_compressor.pre_compress(**accum_kwargs)
            accum_kwargs.update(layer_kwargs)

            # Joinly prune/quantize using SparseGPT
            self.model, layer_kwargs = layer_compressor.compress(**accum_kwargs)
            accum_kwargs.update(layer_kwargs)

            # Compute outputs given compressed layer, memory clean up etc
            (
                self.model,
                layer_kwargs,
            ) = layer_compressor.post_compress(**accum_kwargs)
            accum_kwargs.update(layer_kwargs)

        # Step 2: Prune/quantize head
        # TODO: Need update here -- see MPT for head quantization example
        if self.head_compressor is not None:
            head_compressor = LayerCompressor(self.model.head, inputs)
            head_compressor.pre_compress()
            head_compressor.compress()
            model, extras = head_compressor.post_compress()

        return model, accum_kwargs

    def post_compress(self, **kwargs):
        use_cache = kwargs["use_cache"]
        self.model.apply(torch.quantization.disable_observer)
        self.model.config.use_cache = use_cache

        return (self.model,)
