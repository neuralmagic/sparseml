import torch
import torch.nn as nn
from collections.abc import Mapping

from sparseml.pytorch.optim.manager import ScheduledModifierManager
from sparseml.pytorch.sparsification.quantization.helpers import freeze_bn_stats


DEV = torch.device("cuda:0")


def find_quant_layers(module, layers=[torch.nn.qat.Linear], name=""):
    if type(module) in layers:
        pieces = name.split(".")
        if pieces[-1] == "module":
            name = ".".join(pieces[:-1])
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )
    return res


def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=""):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )
    return res


def apply_recipe(model, recipe):
    manager = ScheduledModifierManager.from_yaml(recipe)
    model.train()
    manager.apply_structure(model, epoch=0.1)
    model.eval()
    return model, manager


def initialize_scales_from_batches(model, data_loader, num_batches):
    print("Collecting data statistics for quantization scales...")
    model.train()
    model.to(DEV)
    with torch.no_grad():
        batches = 0
        while batches < num_batches:
            for batch in data_loader:
                if batches == num_batches:
                    break
                print(f"Batch {batches + 1}/{num_batches}")
                if isinstance(batch, tuple):
                    inp, _ = batch  # Ignore target
                    inp = inp.to(DEV)
                elif isinstance(batch, Mapping):
                    if 'labels' in batch:
                        batch.pop('labels')
                    inp = {k: v.to(DEV) for k, v in batch.items()}
                else:
                    raise ValueError(f"Dont know how to process given batch type: {type(batch)}")

                model(inp)
                batches += 1
    model.apply(torch.quantization.disable_observer)
    model.apply(freeze_bn_stats)