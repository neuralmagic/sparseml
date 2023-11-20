# SparseML ultralytics yolov8 integration

## Installation

We recommend using a [virtualenv] to install dependencies.
```pip install sparseml[ultralytics]```

Note that we require a specific version of ultralytics!

## Dense training

You can run normal ultralytics training with all the normal flags of ultralytics!

```bash
sparseml.ultralytics.train
```

Use `--help` for more information on flags.

## Sparsifying segmentation models

First, you'll need to create a recipe to sparsify yolov8.

Here is a sample one, where only the first two conv layers are
sparsified.  You may save this to a file named `yolov8-pq.yaml`:

```yaml
version: 1.1.0

training_modifiers:
  - !EpochRangeModifier
    start_epoch: 0
    end_epoch: 15

pruning_modifiers:
  - !GMPruningModifier
    init_sparsity: 0.05
    final_sparsity: 0.10
    start_epoch: 5.0
    end_epoch: 10.0
    update_frequency: 1.0
    params: ["model.0.conv.weight", "model.1.conv.weight"]

quantization_modifiers:
  - !QuantizationModifier
    start_epoch: 11.0
    freeze_bn_stats_epoch: 12.0
    disable_quantization_observer_epoch: 13.0
    ignore: ["Upsample", "Concat", "SiLU"]
```

Next we add the `--recipe` argument to our cli command we used
for dense training:

```bash
sparseml.ultralytics.train --recipe yolov8-pq.yaml
```

## ONNX export

To export the trained model to ONNX format, execute the appropriate
CLI command and specify the path to the `.pt` model (`--model`):

```bash
sparseml.ultralytics.export_onnx --model path/to/model.pt
```

To look up optional arguments for the `export` command, run:

```bash
sparseml.ultralytics.export_onnx --help
```

The resulting exported model will be saved in the appropriate directory structure:
```bash
.
├── deployment
│    ├── config.json
│    ├── model.onnx
│    └── recipe.yaml # optionally, if model contains a sparse checkpoint
└── model.onnx
```
