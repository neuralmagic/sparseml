# Scripts for SparseML

[TODO: ENGINEERING: EDIT THE FOLLOWING SO IT REFLECTS ANY UPDATES; THEN REMOVE THIS COMMENT.]

Ease-of-use SparseML scripts, which are implemented as Python scripts for easy consumption and editing, are provided under the `scripts` directory.

To run one of the scripts, invoke it with a Python command from the command line along with the relevant arguments.

```bash
python scripts/onnx/model_download.py \
    --dom cv --sub-dom classification --arch resnet-v1 --sub-arch 50 \
    --dataset imagenet --framework pytorch --desc recal
```

Each script file is fully documented with descriptions, command help printouts, and example commands.

## ONNX

The `onnx` subdirectory is provided for working with models converted to or trained in the [ONNX](https://onnx.ai/) framework.
The following scripts are currently maintained for use:

- `classification_validation.py`: Run an image classification model over a selected dataset to measure validation metrics.
- `model_analysis.py`: Analyze a model to parse it into relevant info for each node/op in the graph such as param counts, flops, is prunable, etc.
- `model_benchmark.py`: Benchmark the inference speed for a model in either the [DeepSparse Engine](https://docs.neuralmagic.com/deepsparse) or [ONNX Runtime](https://github.com/microsoft/onnxruntime).
- `model_download.py`: Download a model from the [SparseZoo](https://docs.neuralmagic.com/sparsezoo/).
- `model_kernel_sparsity.py`: Measure the sparsity of the weight parameters across a model (the result of pruning).
- `model_pruning_config.py`: Create a config.yaml file or a pruning information table to guide the creation of a config.yaml file for pruning a given model in the `sparseml` Python package.
- `model_pruning_loss_sensitivity.py`: Calculate the sensitivity for each prunable layer in a model towards the loss,
  where, for example, a higher score means the layer affects the loss more and therefore should be pruned less.
- `model_pruning_perf_sensitivity.py`: Calculate the sensitivity for each prunable layer in a model towards the performance, where, for example, a higher score means the layer did not give as much net speedup for pruning and therefore should be pruned less.
- `model_quantize_post_training.py`: Perform post-training quantization on a trained model from an ONNX file. Supports
  static quantization which requires a sample dataset to calibrate quantization parameters with.
  
### PyTorch

The `pytorch` subdirectory is provided for working with models trained in the [PyTorch](https://pytorch.org/) framework.
The following scripts are currently maintained for use:

- `classification_export.py`: Export an image classification model to a standard structure including
  an ONNX format, sample inputs, sample outputs, and sample labels.
- `classification_lr_sensitivity.py`: Calculate the learning rate sensitivity for an image classification model
  as compared with the loss. A higher sensitivity means a higher loss impact.
- `classification_pruning_loss_sensitivity.py`: Calculate the sensitivity for each prunable layer in an
  image classification model towards the loss, where, for example, a higher score means the layer affects the loss more and therefore should be pruned less.
- `classification_train.py`: Train an image classification model using a config.yaml file to modify the training process such as for pruning or sparse transfer learning.
- `detection_lr_sensitivity.py`: Calculate the learning rate sensitivity for an object detection model
  as compared with the loss. A higher sensitivity means a higher loss impact.
- `detection_pruning_loss_sensitivity.py`: Calculate the sensitivity for each prunable layer in an
  object detection model towards the loss, where, for example, a higher score means the layer affects the loss more and therefore should be pruned less.
- `detection_train.py`: Train an object detection model using a config.yaml file
  to modify the training process, such as for pruning or sparse transfer learning.
- `detection_validation.py`: Calculate the mean average precision (mAP) for a given object detection model at
  a single IoU value or given range and output per class, per IoU averages.
- `model_download.py`: Download a model from the [SparseZoo](https://docs.neuralmagic.com/sparsezoo/).
- `model_export.py`: Convert a PyTorch model to an ONNX file, saving it along with the model checkpoint,
  sample input, and sample output data.
- `model_quantize_qat_export.py`: Convert an ONNX graph exported from PyTorch using QAT
  to a fully quantized ONNX model.
- `torchvision_export.py`: Download a model from the [torchvision](https://pytorch.org/docs/stable/torchvision/models.html)
  model zoo, saving it as an ONNX file along with the model checkpoint, sample input, and sample output data.

### TensorFlow

The `tensorflow` subdirectory is provided for working with models trained in the [TensorFlow](https://www.tensorflow.org/) framework.
The following scripts are currently maintained for use:

- `classification_export.py`: Export an image classification model to a standard structure including an ONNX format, sample inputs, sample outputs, and sample labels.
- `classification_train.py`: Train and evaluate an image classification model, optionally with a configuration yaml file for model pruning and sparse transfer learning.
- `tf_object_detection_api_train.py`: Train and evaluate an object detection model, optionally with a configuration yaml file for model pruning.
- `model_download.py`: Download a model from the [SparseZoo](https://docs.neuralmagic.com/sparsezoo/).
