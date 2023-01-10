# Training

| Base           | NM IC | torchvision IC | yolov5 | transformers | yolact | openpifpaf |
| -------------- | ----- | -------------- | ------ | ------------ | ------ | ---------- |
| CLI            | ✅     | ✅              | ❓      | ❓            | ❓      | ❓          |
| API            | ✅     | ✅              | ❓      | ❓            | ❓      | ❓          |
| Dense training | ✅     | ✅              | ❓      | ❓            | ❓      | ❓          |

| Sparsification             | NM IC | torchvision IC | yolov5 | transformers | yolact | openpifpaf |
| -------------------------- | ----- | -------------- | ------ | ------------ | ------ | ---------- |
| Recipe                     | ✅     | ✅              | ❓      | ❓            | ❓      | ❓          |
| Recipe args                | ✅     | ✅              | ❓      | ❓            | ❓      | ❓          |
| Recipe is optional         | ✅     | ✅              | ❓      | ❓            | ❓      | ❓          |
| Pruning from a recipe      | ✅     | ✅              | ❓      | ❓            | ❓      | ❓          |
| Pruning with EMA           | ❌     | ✅              | ❓      | ❓            | ❓      | ❓          |
| Quantization from a recipe | ✅     | ✅              | ❓      | ❓            | ❓      | ❓          |
| Qauntization with AMP      | ✅     | ✅              | ❓      | ❓            | ❓      | ❓          |
| Quantization with EMA      | ❌     | ✅              | ❓      | ❓            | ❓      | ❓          |
| Distillation support       | ❌     | ❌              | ❓      | ❓            | ❓      | ❓          |

| Datasets                | NM IC | torchvision IC | yolov5 | transformers | yolact | openpifpaf |
| ----------------------- | ----- | -------------- | ------ | ------------ | ------ | ---------- |
| Use standard datasets   | ✅     | ✅              | ❓      | ❓            | ❓      | ❓          |
| train/val/test datasets | ✅     | ✅              | ❓      | ❓            | ❓      | ❓          |
| Auto download datasets  | ✅     | ❌              | ❓      | ❓            | ❓      | ❓          |

| Checkpoints                                   | NM IC | torchvision IC | yolov5 | transformers | yolact | openpifpaf |
| --------------------------------------------- | ----- | -------------- | ------ | ------------ | ------ | ---------- |
| Model checkpoints from original integration   | ✅     | ✅              | ❓      | ❓            | ❓      | ❓          |
| Model checkpoints from sparsezoo              | ✅     | ✅              | ❓      | ❓            | ❓      | ❓          |
| Checkpoints store all necessary configuration | ✅     | ✅              | ❓      | ❓            | ❓      | ❓          |
| Best checkpoint saved                         | ✅     | ✅              | ❓      | ❓            | ❓      | ❓          |
| Best Pruned checkpoint saved                  | ❌     | ❌              | ❓      | ❓            | ❓      | ❓          |
| Best Quantized checkpoint saved               | ❌     | ❌              | ❓      | ❓            | ❓      | ❓          |
| Best Pruned/Quantized checkpoint saved        | ❌     | ❌              | ❓      | ❓            | ❓      | ❓          |
| Architecture changes from saved recipe        | ✅     | ✅              | ❓      | ❓            | ❓      | ❓          |
| Staged recipes                                | ✅     | ✅              | ❓      | ❓            | ❓      | ❓          |

| Logging            | NM IC | torchvision IC | yolov5 | transformers | yolact | openpifpaf |
| ------------------ | ----- | -------------- | ------ | ------------ | ------ | ---------- |
| stdout             | ✅     | ✅              | ❓      | ❓            | ❓      | ❓          |
| Weights and Biases | ❌     | ✅              | ❓      | ❓            | ❓      | ❓          |
| TensorBoard        | ✅     | ✅              | ❓      | ❓            | ❓      | ❓          |

# Export

| Exporting                        | NM IC | torchvision IC | yolov5 | transformers | yolact | openpifpaf |
| -------------------------------- | ----- | -------------- | ------ | ------------ | ------ | ---------- |
| Optional one shot                | ✅     | ❌              | ❓      | ❓            | ❓      | ❓          |
| Convert to ONNX                  | ✅     | ✅              | ❓      | ❓            | ❓      | ❓          |
| ONNX graph validation            | ✅     | ✅              | ❓      | ❓            | ❓      | ❓          |
| Graph folding/optimization       | ✅     | ✅              | ❓      | ❓            | ❓      | ❓          |
| Convert to TorchScript           | ❌     | ❌              | ❓      | ❓            | ❓      | ❓          |
| Batch size fixed or dynamic      | ❓     | ❓              | ❓      | ❓            | ❓      | ❓          |
| Input shape fixed or dynamic     | ❓     | ❓              | ❓      | ❓            | ❓      | ❓          |
| Save to simple deployment folder | ✅     | ✅              | ❓      | ❓            | ❓      | ❓          |
| Save to SparseZoo folder         | ❌     | ❌              | ❓      | ❓            | ❓      | ❓          |

