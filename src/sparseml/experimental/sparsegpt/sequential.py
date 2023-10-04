# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from copy import deepcopy
from typing import List, Optional

import torch

from sparseml.experimental.sparsegpt.layer_compressor import LayerCompressor
from sparseml.experimental.sparsegpt.model_preprocessor import ModelPreprocessor


class SequentialSparseGPT:
    def __init__(
        self,
        model,
        recipe: Optional[str] = None,
        model_preprocessors: Optional[List[ModelPreprocessor]] = None,
        bottom_compressor: Optional[LayerCompressor] = None,
        args=None,
    ):
        self.model = model
        self.model_preprocessors = model_preprocessors
        self.bottom_compressor = bottom_compressor
        self.recipe = recipe
        self.manager = None
        self.compressible_layers = self.compressible_layers()
        self.args = args

    def compressible_layers(self):
        """
        Derived class could override
        """
        try:
            return self.model.model.decoders.layers
        except Exception:
            raise RuntimeError(
                "Derived class should override to provide list of compressible layers"
            )

    def pre_compress(self, dev: str = "cuda:0", **kwargs):
        model = self.model
        all_extras = {}
        for processor in self.model_preprocessors:
            # We assume the processors are independent, and therefore
            # pass in the initial kwargs into each of them
            model, extras = processor(dev=dev, **kwargs)
            all_extras.update(extras)
        return model, all_extras

    def compress(self, dev: str = "cuda:0", **kwargs):
        accum_kwargs = deepcopy(kwargs)

        if "args" not in kwargs:
            # Ensure that all CLI arguments are also passed down to all the steps
            kwargs["args"] = self.args

        self.model, extras = self.pre_compress(dev=dev, **kwargs)
        self.model = self.model.cpu()
        torch.cuda.empty_cache()

        self.manager = extras.pop("manager", self.manager)

        # Step 0: BottomCompressor accomplishes two things:
        # 1) Compress the embedding if needed
        # 2) Pass the calibration data through the (compressed) bottom part
        # of the network, capturing the output which will become the inputs
        # to the first decoder layer
        # Also return attention_mask as part of kwargs
        accum_kwargs.update(extras)
        self.model, extras = self.bottom_compressor.compress(dev=dev, **accum_kwargs)
        accum_kwargs.update(extras)

        # Step 1: Sequentially prune/quantize decoder layers
        inputs = None
        num_layers = len(self.compressible_layers)
        for idx, layer in enumerate(self.compressible_layers):
            if "outputs" not in accum_kwargs:
                raise RuntimeError(
                    "The 'outputs' key is expected but not found from the "
                    "return of the bottom compressor"
                )
            inputs = accum_kwargs["outputs"]
            print(f"\n===== Compressing layer {idx}/{num_layers-1} =====")
            layer_compressor = LayerCompressor(
                self.model, layer, idx, inputs, self.manager, self.args
            )

            # Prune/quantize using SparseGPT
            self.model, layer_kwargs = layer_compressor.compress(
                dev=dev, **accum_kwargs
            )
            accum_kwargs.update(layer_kwargs)

        return self.model, {}

    def post_compress(self, dev: str = "cuda:0", **kwargs):
        use_cache = kwargs["use_cache"]
        self.model.apply(torch.quantization.disable_observer)
        self.model.config.use_cache = use_cache

        return self.model, {}
