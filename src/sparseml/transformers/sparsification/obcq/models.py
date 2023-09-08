import torch

__all__ = ["load_opt_model"]

def load_opt_model(model_path):
    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import OPTForCausalLM

    model = OPTForCausalLM.from_pretrained(model_path, torch_dtype="auto")
    model.seqlen = model.config.max_position_embeddings
    return model
