import torch
from sparseml.pytorch.utils import ModuleExporter
from sparseml.pytorch.models import ModelRegistry
from sparseml.pytorch.optim import ScheduledModifierManager

model = ModelRegistry.create(
    key='resnet50',
    pretrained=False,
    pretrained_dataset="imagenet",
    num_classes=1000
)


ScheduledModifierManager.from_yaml("quantization_recipe.yaml").apply(model, epoch=float("inf"))

print(model)

exporter = ModuleExporter(model, ".")
exporter.export_onnx(
    torch.randn(1, 3, 224, 224),
    "quantized_test.onnx",
    convert_qat=False,
)