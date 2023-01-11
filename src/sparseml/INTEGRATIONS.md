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
| Gradient accumulation[0] | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ |

[0] steps_per_epoch should be `len(data_loader) / gradient_accum_steps`

| Sparsification    | NM IC | torchvision IC | yolov5 | transformers | yolact | openpifpaf |
| ----------------- | ----- | -------------- | ------ | ------------ | ------ | ---------- |
| Recipe (optional) | ✅     | ✅              | ✅      | ❓            | ❓      | ✅          |
| Recipe args       | ✅     | ✅              | ✅      | ❓            | ❓      | ❌          |
| Pruning           | ✅     | ✅              | ✅      | ❓            | ❓      | ✅          |
| Quantization[0]   | ✅     | ✅              | ✅      | ❓            | ❓      | ✅          |
| EMA               | ❌     | ✅              | ✅      | ❓            | ❓      | ❌          |
| AMP[1]            | ✅     | ✅              | ✅      | ❓            | ❓      | ❌          |
| Distillation[2]   | ❌     | ❌              | ❌      | ❓            | ❓      | ❌          |

[0] Quantization needs to work with EMA/AMP (disable them both when quantization is activated)

[1] AMP implementation details:
1. Use wrap_optim argument of manager.modify `manager.modify(..., wrap_optim=scaler)`
2. Call `scaler._enabled = False` when using quantization is enabled

[2] Distillation implementation details
1. Pass `distillation_teacher=...` to `manager.initializer`
2. Call `loss = manager.loss_update(...)` after loss is computed

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

| Logging[0]          | NM IC | torchvision IC | yolov5 | transformers | yolact | openpifpaf |
| ------------------ | ----- | -------------- | ------ | ------------ | ------ | ---------- |
| stdout             | ✅     | ✅              | ✅      | ❓            | ❓      | ✔️          |
| Weights and Biases | ❌     | ✅              | ✅      | ❓            | ❓      | ❌          |
| TensorBoard        | ✅     | ✅              | ❌      | ❓            | ❓      | ❌          |

[0] The units for x axis in logging should be number of optimizer steps. Notably: `num_optimizer_steps = num_batches / gradient_accum_steps`. So when gradient_accumuluation is not used, the x axis will be number of batches trained on. 

# Export

| Exporting                        | NM IC | torchvision IC | yolov5 | transformers | yolact | openpifpaf |
| -------------------------------- | ----- | -------------- | ------ | ------------ | ------ | ---------- |
| Optional one shot                | ✅     | ❌              | ✅      | ❓            | ❓      | ✅          |
| Convert to ONNX                  | ✅     | ✅              | ✅      | ❓            | ❓      | ✅          |
| ONNX graph validation            | ✅     | ✅              | ✅      | ❓            | ❓      | ✅          |
| Graph folding/optimization       | ✅     | ✅              | ✅      | ❓            | ❓      | ✅          |
| Convert to TorchScript           | ❌     | ❌              | ✔️       | ❓            | ❓      | ❌          |
| Static batch size                | ❌     | ❌              | ✔️       | ❓            | ❓      | ❌          |
| Dynamic batch size               | ✅     | ✅              | ✅      | ❓            | ❓      | ✅          |
| Static input shape               | ✅     | ✅              | ✅      | ❓            | ❓      | ✅          |
| Dynamic input shape              | ❌     | ❌              | ❌      | ❓            | ❓      | ❌          |
| Save to simple deployment folder | ✅     | ✅              | ✅      | ❓            | ❓      | ✅          |
| Save to SparseZoo folder         | ❌     | ❌              | ❌      | ❓            | ❓      | ❌          |

