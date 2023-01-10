Key: 
- ✅ it is supported explicitly by NM code
- ✔️ it is supported by the underlying integration
- ❌ it is not supported
- ❓ unknown, need to check

# Training

| Base           | NM IC | torchvision IC | yolov5 | transformers | yolact | openpifpaf |
| -------------- | ----- | -------------- | ------ | ------------ | ------ | ---------- |
| CLI            | ✅     | ✅              | ✅      | ❓            | ❓      | ✔️          |
| API            | ✅     | ✅              | ✅      | ❓            | ❓      | ✔️          |
| Dense training | ✅     | ✅              | ✅      | ❓            | ❓      | ✔️          |

| Sparsification    | NM IC | torchvision IC | yolov5 | transformers | yolact | openpifpaf |
| ----------------- | ----- | -------------- | ------ | ------------ | ------ | ---------- |
| Recipe (optional) | ✅     | ✅              | ✅      | ❓            | ❓      | ✅          |
| Recipe args       | ✅     | ✅              | ✅      | ❓            | ❓      | ❌          |
| Pruning           | ✅     | ✅              | ✅      | ❓            | ❓      | ✅          |
| Quantization[0]   | ✅     | ✅              | ✅      | ❓            | ❓      | ✅          |
| EMA               | ❌     | ✅              | ✅      | ❓            | ❓      | ❌          |
| AMP               | ✅     | ✅              | ✅      | ❓            | ❓      | ❌          |
| Distillation      | ❌     | ❌              | ❌      | ❓            | ❓      | ❌          |

[0] Quantization needs to work with EMA/AMP (disable them both when quantization is activated)

| Datasets                | NM IC | torchvision IC | yolov5 | transformers | yolact | openpifpaf |
| ----------------------- | ----- | -------------- | ------ | ------------ | ------ | ---------- |
| Use standard datasets   | ✅     | ✅              | ✅      | ❓            | ❓      | ✔️          |
| train/val/test datasets | ✅     | ✅              | ✅      | ❓            | ❓      | ✔️          |
| Auto download datasets  | ✅     | ❌              | ✅      | ❓            | ❓      | ✔️          |

| Checkpoints                           | NM IC | torchvision IC | yolov5 | transformers | yolact | openpifpaf |
| ------------------------------------- | ----- | -------------- | ------ | ------------ | ------ | ---------- |
| Checkpoints from original integration | ✅     | ✅              | ✅      | ❓            | ❓      | ✅          |
| Checkpoints from sparsezoo            | ✅     | ✅              | ✅      | ❓            | ❓      | ✅          |
| Best checkpoint                       | ✅     | ✅              | ✅      | ❓            | ❓      | ✅          |
| Best Pruned checkpoint                | ❌     | ❌              | ❌      | ❓            | ❓      | ❌          |
| Best Quantized checkpoint             | ❌     | ❌              | ❌      | ❓            | ❓      | ❌          |
| Best Pruned/Quantized checkpoint      | ❌     | ❌              | ❌      | ❓            | ❓      | ❌          |
| Changes architecture from recipe      | ✅     | ✅              | ✅      | ❓            | ❓      | ✅          |
| Staged recipes                        | ✅     | ✅              | ✅      | ❓            | ❓      | ✅          |

| Logging            | NM IC | torchvision IC | yolov5 | transformers | yolact | openpifpaf |
| ------------------ | ----- | -------------- | ------ | ------------ | ------ | ---------- |
| stdout             | ✅     | ✅              | ✅      | ❓            | ❓      | ✔️          |
| Weights and Biases | ❌     | ✅              | ✅      | ❓            | ❓      | ❌          |
| TensorBoard        | ✅     | ✅              | ❌      | ❓            | ❓      | ❌          |

# Export

| Exporting                        | NM IC | torchvision IC | yolov5 | transformers | yolact | openpifpaf |
| -------------------------------- | ----- | -------------- | ------ | ------------ | ------ | ---------- |
| Optional one shot                | ✅     | ❌              | ✅      | ❓            | ❓      | ✅          |
| Convert to ONNX                  | ✅     | ✅              | ✅      | ❓            | ❓      | ✅          |
| ONNX graph validation            | ✅     | ✅              | ✅      | ❓            | ❓      | ✅          |
| Graph folding/optimization       | ✅     | ✅              | ✅      | ❓            | ❓      | ✅          |
| Convert to TorchScript           | ❌     | ❌              | ✅      | ❓            | ❓      | ❌          |
| Static batch size                | ❌     | ❌              | ❌      | ❓            | ❓      | ❌          |
| Dynamic batch size               | ✅     | ✅              | ✅      | ❓            | ❓      | ✅          |
| Static input shape               | ✅     | ✅              | ✅      | ❓            | ❓      | ✅          |
| Dynamic input shape              | ❌     | ❌              | ❌      | ❓            | ❓      | ❌          |
| Save to simple deployment folder | ✅     | ✅              | ✅      | ❓            | ❓      | ✅          |
| Save to SparseZoo folder         | ❌     | ❌              | ❌      | ❓            | ❓      | ❌          |

