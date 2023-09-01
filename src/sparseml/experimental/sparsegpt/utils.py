import torch


class Catcher(torch.nn.Module):
    def __init__(self, module, target_keys):
        super().__init__()
        self.module = module
        self.target_keys = target_keys
        self.cache = {}
        for key in target_keys:
            self.cache[key] = []

    def forward(self, inp, **kwargs):
        for key in self.target_keys:
            if key in kwargs:
                self.cache[key].append(kwargs[key])
        raise ValueError

    def get_cache(self):
        return self.cache


def catch(model, attention_layer, target_keys, data_loader, nsamples):
    catcher_module = Catcher(attention_layer, target_keys)
    for input_id, inp in enumerate(data_loader):
        if nsamples is not None and input_id == nsamples:
            break
        try:
            model(inp, use_cache=False)
        except ValueError:
            pass
    return catcher_module.get_cache()


def execute_offloaded_module(
    module,
    buffer,
    dev,
    nsamples=None,
    overwrite_buffer=True,
    cached_inputs=None,
    **kwargs,
):
    module.to(dev)
    if not overwrite_buffer:
        new_buffer = []
    for input_index, inp in enumerate(buffer):
        if nsamples is not None and input_index == nsamples:
            break
        if cached_inputs:
            module_kwargs = kwargs[input_index].update(kwargs)
        else:
            module_kwargs = kwargs

        output = module(inp, **module_kwargs)
        if overwrite_buffer:
            buffer[input_index] = output
        else:
            new_buffer.append(output)

    module.cpu()
    torch.cuda.empty_cache()
    if overwrite_buffer:
        return buffer
    else:
        return new_buffer


class OffLoadedModule(torch.nn.Module):
    def __init__(self, module, device, compression_strategy=None):
        self._module = module
        self.device = device
        self.compression_strategy = compression_strategy

        for name, child in module.named_modules():
            setattr(self._module, name, OffLoadedModule(child, device))

    def forward(self, inputs, *args, **kwargs):
        if isinstance(self._module, torch.nn.Linear): # Is this a leaf of the graph?
            self._module.to(self.device)
            if self.compression_strategy is not None:
                self._module = self.compression_strategy(self._module)

        output = self._module.forward(inputs, *args, **kwargs)

        if isinstance(self._module, torch.nn.Linear):
            self._module.cpu()
            torch.cuda.empty_cache()

        return output


