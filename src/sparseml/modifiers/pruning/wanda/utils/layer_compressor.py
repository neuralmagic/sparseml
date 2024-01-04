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


from typing import Dict

from sparseml.modifiers.pruning.wanda.utils.wanda_wrapper import WandaModuleCompressor
from sparseml.modifiers.utils.compression_wrapper import ModuleCompressor
from sparseml.modifiers.utils.layer_compressor import LayerCompressor


__all__ = ["WandaLayerCompressor"]


class WandaLayerCompressor(LayerCompressor):
    """
    Runs the Wanda algorithm on a single layer using calibration data inputs

    Lifecycle:
        - compress
            - pre_compress_parallel (optional)
            - add_batch
            - fasterprune
            - post_compress

    :param model: model containing the layer we are running compression on
    :param layer: layer to run compression on
    :param layer_index: index of layer in the model
    :param inputs: calibration data to pass through the layer
    :param args: additional keyword arguments
    """

    module_compressor_class: ModuleCompressor = WandaModuleCompressor

    def compress(self, dev: str = "cuda:0", **kwargs) -> Dict:
        """
        Run WANDA compression on all compressible modules in the layer

        :param dev: device to run computation on
        """
        self.layer.to(dev)
        self.sequentially_compress(**kwargs)
        extras = self.post_compress(**kwargs)
        return {"outputs": extras["outputs"]}

    def invoke_fasterprune(self, module_compressor: "WandaModuleCompressor"):
        # run WandaGPT algorithm on current module
        module_compressor.fasterprune(
            self.args["sparsity"],
            prunen=self.args["prunen"],
            prunem=self.args["prunem"],
        )
