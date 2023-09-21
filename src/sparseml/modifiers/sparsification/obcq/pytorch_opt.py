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

import torch
from torch.nn import ModuleList

from sparseml.modifiers.sparsification.obcq.utils.layer_compressor import BaseCompressor
from sparseml.transformers.sparsification.obcq.sparse_gpt_modifier import (
    SparseGPTModifier,
)
from sparseml.modifiers.sparsification.obcq.utils.utils import (
    catch,
    execute_offloaded_module,
)


__all__ = ["SparseOPTModifierPyTorch", "OPTBottomCompressor"]


class OPTBottomCompressor(BaseCompressor):
    """
    The OPT-specific BottomCompressor accomplishes three things:
        1) Compress the embedding if needed
        2) Pass the calibration data through the (compressed) bottom part of the
        network, capturing the outputs which will become the inputs to the first
        decoder layer
        3) Return attention_mask as part of kwargs
    """

    @staticmethod
    def _cache_attention_inputs(model, dataloader, device, nsamples):
        model.model.decoder.embed_tokens.to(device)
        model.model.decoder.embed_positions.to(device)
        if (
            hasattr(model.model.decoder, "project_out")
            and model.model.decoder.project_out
        ):
            model.model.decoder.project_out.to(device)
        if (
            hasattr(model.model.decoder, "project_in")
            and model.model.decoder.project_in
        ):
            model.model.decoder.project_in.to(device)

        model.model.decoder.layers[0].to(device)
        cached_inputs = catch(
            model,
            model.model.decoder.layers[0],
            ["attention_mask"],
            dataloader,
            nsamples,
        )

        model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions.cpu()
        if (
            hasattr(model.model.decoder, "project_out")
            and model.model.decoder.project_out
        ):
            model.model.decoder.project_out.cpu()
        if (
            hasattr(model.model.decoder, "project_in")
            and model.model.decoder.project_in
        ):
            model.model.decoder.project_in.cpu()

        model.model.decoder.layers[0].cpu()
        torch.cuda.empty_cache()
        return cached_inputs

    @staticmethod
    def forward(model, data_loader, device, nsamples=None):
        # Catch attention mask
        cached_inputs = OPTBottomCompressor._cache_attention_inputs(
            model, data_loader, device, nsamples
        )
        buffer = [b[0] for b in cached_inputs.pop("inputs")]
        for layer in model.model.layers:
            buffer = execute_offloaded_module(
                layer,
                buffer,
                device,
                cached_inputs=cached_inputs,
                use_cache=False,
            )
            buffer = [b[0] for b in buffer]

        del cached_inputs
        torch.cuda.empty_cache()

        if model.model.decoder.final_layer_norm is not None:
            buffer = execute_offloaded_module(
                model.model.decoder.final_layer_norm,
                buffer,
                device,
            )
        if model.model.decoder.project_out is not None:
            buffer = execute_offloaded_module(
                model.model.decoder.project_out,
                buffer,
                device,
            )
        logits = execute_offloaded_module(
            model.lm_head,
            buffer,
            device,
        )

        return logits


class SparseOPTModifier(SparseGPTModifier):
    """
    OPT-specific functions for applying the one-shot OBCQ algorithm to a model
    """
    def compressible_layers(self) -> ModuleList:
        """
        :return: list of OPT submodules that can be sparsified
        """
        return self.model.model.decoder.layers

    def bottom_compressor(self) -> OPTBottomCompressor:
        """
        :return: model used for calibration, outputs from bottom part of network,
        attention mask, and kv-cache state
        """
        return OPTBottomCompressor(self.model)

    def head_compressor(self) -> None:
        return None  # no head compressor for OPT
