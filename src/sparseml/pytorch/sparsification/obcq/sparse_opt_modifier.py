import logging
from typing import Optional
import torch
from sparseml.pytorch.sparsification.modifier import PyTorchModifierYAML
from sparseml.pytorch.sparsification.obcq.sparse_gpt_modifier import SparseGPTModifier
from sparseml.pytorch.sparsification.obcq.layer_compressor import BaseCompressor

_LOGGER = logging.getLogger(__name__)

__all__ = ["SparseOPTModifier"]

class OPTBottomCompressor(BaseCompressor):
    """
    OPT specific
    """

    def post_compress(
        self, dataloader=None, nsamples: int = None, dev: str = "cuda:0", **kwargs
    ):
        model = self.model
        layers = model.model.decoder.layers
        nsamples = len(dataloader) if nsamples is None else nsamples

        use_cache = model.config.use_cache
        model.config.use_cache = False

        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(
            dev
        )
        if (
            hasattr(model.model.decoder, "project_out")
            and model.model.decoder.project_out
        ):
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        if (
            hasattr(model.model.decoder, "project_in")
            and model.model.decoder.project_in
        ):
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
        layers[0] = layers[0].to(dev)

        dtype = next(iter(model.parameters())).dtype
        inps = torch.zeros(
            (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
        )
        cache = {"i": 0, "attention_mask": None}

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, inp, **kwargs):
                inps[cache["i"]] = inp
                cache["i"] += 1
                cache["attention_mask"] = kwargs["attention_mask"]
                raise ValueError

        layers[0] = Catcher(layers[0])
        for batch in dataloader:
            try:
                model(batch[0].to(dev))
            except ValueError:
                pass
        layers[0] = layers[0].module

        layers[0] = layers[0].cpu()
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if (
            hasattr(model.model.decoder, "project_out")
            and model.model.decoder.project_out
        ):
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if (
            hasattr(model.model.decoder, "project_in")
            and model.model.decoder.project_in
        ):
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
        torch.cuda.empty_cache()

        outs = torch.zeros_like(inps)
        attention_mask = cache["attention_mask"]

        extras = {
            "use_cache": use_cache,
            "outputs": outs,
            "attention_mask": attention_mask,
        }
        self.model = model
        return model, extras

@PyTorchModifierYAML
class SparseOPTModifier(SparseGPTModifier):
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
        super().__init__(
            sparsity=sparsity,
            block_size=block_size,
            quantize=quantize,
            num_bits=num_bits,
            dampening_frac=dampening_frac,
            sequential_update=sequential_update
        )

    def compressible_layers(self):
        return self.model.model.decoder.layers
    
    def bottom_compressor(self):
        return OPTBottomCompressor(self.model)
    
    def head_compressor(self):
        return None #no head compressor for OPT
    
