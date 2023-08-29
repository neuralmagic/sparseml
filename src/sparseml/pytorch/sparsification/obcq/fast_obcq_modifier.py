import logging
from typing import Dict, List, Optional, Union, Any

import torch

from sparseml.pytorch.sparsification.modifier import ModifierProp, PyTorchModifierYAML

_LOGGER = logging.getLogger(__name__)


@PyTorchModifierYAML()
class FastOBCQModifier():
    """
    """
    def __init__(
        self,
        target_sparsity: float,
        block_size: int,
        mse: Dict,
        quantize: bool=True,
        num_bits: int=8,
        dampening_frac: Optional[float] = 0.001,
        sequential_update: Optional[bool] = True,
    ):
        self.model = None
        self._target_sparsity = target_sparsity
        self._block_size = block_size
        self._mse = mse
        self._quantize = quantize
        self._num_bits = num_bits
        self._dampening_frac = dampening_frac
        self._sequential_update = sequential_update

    @ModifierProp()
    def target_sparsity(self) -> Union[float, Dict[str, float], str]:
        """
        :return: Sparsity to compress model to. Can be a float, to
            apply uniform sparsity, or a profile mapping layer names to target sparsity.
            Can also be a file path to load a sparsity profile from
        """
        return self._target_sparsity

    @target_sparsity.setter
    def target_sparsity(self, value: Union[float, Dict[str, float], str]):
        """
        :params value: Sparsity to compress model to. Can be a float, to
            apply uniform sparsity, or a profile mapping layer names to target sparsity.
            Can also be a file path to load a sparsity profile from
        """
        self._target_sparsity = value

    @ModifierProp()
    def block_size(self) -> Union[int, float, Dict[str, Union[int, float]], str]:
        """
        :return: Used to determine number of columns to compress in one pass. If
            block_size is an int, then it directly corresponds to the number of columns.
            If a float, then it must be in the range (0, 1] and is used to specify
            fraction of columns compressed at one time (will be rounded down). A
            block_size profile can also be used, to specify different block sizes for
            each layer. The profile can be specified via a filepath to a yaml loadable
            file
        """
        return self._block_size

    @block_size.setter
    def block_size(self, value: Union[int, float, Dict[str, Union[int, float]], str]):
        """
        :params value: Used to determine number of columns to compress in one pass. If
            block_size is an int, then it directly corresponds to the number of columns.
            If a float, then it must be in the range (0, 1] and is used to specify
            fraction of columns compressed at one time (will be rounded down). A
            block_size profile can also be used, to specify different block sizes for
            each layer. The profile can be specified via a filepath to a yaml loadable
            file
        """
        self._block_size = value


    @ModifierProp()
    def mse(self) -> Optional[Dict]:
        """
        :return: settings for MSE-based optimization of quantization parameters. This
            optimization attempts to select the scale and zero point that minimizes the
            quantization MSE
        """
        return self._mse

    @mse.setter
    def mse(self, value: Optional[Dict]):
        """
        :params value: settings for MSE-based optimization of quantization parameters.
            This optimization attempts to select the scale and zero point that minimizes
            the quantization MSE
        """
        self._mse = value


    def initialize(self, model: Any, experiment_name: Optional[str] = None, **kwargs):
        """
        """
        super().initialize(model, experiment_name)
        model = self.model

        for processor in self.model_preprocessors:
            self.model, extras = processor.pre_process(model, dev="cuda:0", **kwargs)
        return model, kwargs

    def update(self, model: Any, dataloader: List, **kwargs):
        """
        """
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
    
    def finalize(self, model: Any, **kwargs):
        use_cache = kwargs["use_cache"]
        self.model.apply(torch.quantization.disable_observer)
        self.model.config.use_cache = use_cache

        return (self.model,)