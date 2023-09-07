import torch


class Catcher(torch.nn.Module):
    def __init__(self, module, target_keys):
        super().__init__()
        self.module = module
        self.cache = {key: [] for key in target_keys}

    def forward(self, *args, **kwargs):
        for key in self.cache:
            self.cache[key].append(kwargs[key])
        raise ValueError

    def get_cache(self):
        return self.cache


def replace_module(model, old_module, new_module):
    for module_name, module in model.named_modules():
        if old_module == module:
            break

    current_module = model
    module_name = module_name.split(".")
    for child_module in module_name[:-1]:
        current_module = getattr(current_module, child_module)
    setattr(current_module, module_name[-1], new_module)


def catch(model, attention_layer, target_keys, data_loader, nsamples):
    catcher_module = Catcher(attention_layer, target_keys)
    replace_module(model, attention_layer, catcher_module)
    device = next(attention_layer.parameters()).device
    for input_id, inp in enumerate(data_loader):
        if nsamples is not None and input_id == nsamples:
            break
        try:
            model(inp.to(device), use_cache=False)
        except ValueError:
            pass
    replace_module(model, catcher_module, attention_layer)
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
        if cached_inputs is None:
            module_kwargs = kwargs
        else:
            module_kwargs = {key: cached_inputs[key][input_index] for key in cached_inputs}
            module_kwargs.update(kwargs)
        output = module(inp.to(dev), **module_kwargs)
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
    def __init__(self, module, device):
        self._module = module
        self.device = device

        for name, child in module.named_modules():
            setattr(self._module, name, OffLoadedModule(child, device))

    def load_parameters(self):
        [p.to(self.device) for p in self._module.parameters(recurse=False)]

    def offload_parameters(self):
        [p.cpu() for p in self._module.parameters(recurse=False)]
        torch.cuda.empty_cache()

    def offloaded_forward(self, *args, **kwargs):
        self.load_parameters()
        output = self._module.forward(*args, **kwargs)
        self.offload_parameters()

        return output

    def forward(self, *args, **kwargs):
        return self.offloaded_forward(*args, **kwargs)


class SequentialCompressor(OffLoadedModule):
    def __init__(self, module, device, compression_algorithm=None, parent_module=None):
        self._module = module
        self.device = device
        self.compression_algorithm = compression_algorithm
        self.parent_module = parent_module
        self.cache = None
        self.cache_inputs = False

        for name, child in module.named_modules():
            setattr(self._module, name, SequentialCompressor(child, device, compression_algorithm, self._module))

    def is_compressible(self):
        if self.compression_algorithm is not None:
            return self.compression_algorithm.is_compressible(self._module)
        else:
            return False

    def evaluate_cached_inputs(self, inputs, *args, **kwargs):
        if self.cache is None:
            if self.parent_module is None:
                for inp in inputs:
                    self._module(inp, *args, **kwargs)
            else:
                self.cache_inputs = True
                self.parent_module.evaluate_cached_inputs(inputs, *args, **kwargs)

    def clear_cached(self):
        del self.cache
        self.cache = None
        self.cache_inputs = False
        torch.cuda.empty_cache()

    def forward(self, *args, **kwargs):
        if self.cache_inputs:
            if self.cache is None:
                self.cache["args"] = [args]
                self.cache["kwargs"] = [kwargs]
            else:
                self.cache["args"].append(args)
                self.cache["kwargs"].append(kwargs)
        return self.offloaded_forward(*args, **kwargs)

    def compress(self, *args, **kwargs):
        if self.is_comporessible():
            self.cache_inputs = True
            self.parent_module.evaluate_cached_inputs(*args, **kwargs)
            self._module = self.compression_strategy(self._module, self.cached_inputs)
            self.clear_cache()
        else:
            for child in self._module.modules():
                child.compress(*args, **kwargs)