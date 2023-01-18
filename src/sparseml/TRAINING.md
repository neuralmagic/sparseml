Key: 
- ✅ it is supported explicitly by NM code
- ✔️ it is supported by the underlying integration
- ❌ it is not supported
- ❓ unknown, need to check

# Training

| Base                     | NM IC | torchvision IC | yolov5 | transformers | yolact | openpifpaf |
| ------------------------ | ----- | -------------- | ------ | ------------ | ------ | ---------- |
| CLI                      | ✅     | ✅              | ✅      | ✅            | ✅      | ✔️          |
| API                      | ✅     | ✅              | ✅      | ✅            | ✅      | ✔️          |
| Dense training           | ✅     | ✅              | ✅      | ✅            | ✅      | ✔️          |
| Gradient accumulation[0] | ❌     | ✅              | ❓      | ✔️            | ❌      | ❌          |
| DP                       | ❓     | ❌              | ❓      | ❓            | ❓      | ✔️          |
| DDP                      | ✅     | ✅              | ❓      | ✔️            | ❌      | ✔️          |

[0] steps_per_epoch should be `len(data_loader) / gradient_accum_steps`

| Sparsification    | NM IC | torchvision IC | yolov5 | transformers | yolact | openpifpaf |
| ----------------- | ----- | -------------- | ------ | ------------ | ------ | ---------- |
| Recipe (optional) | ✅     | ✅              | ✅      | ✅            | ✅      | ✅          |
| Recipe args       | ✅     | ✅              | ✅      | ✅            | ❌      | ❌          |
| EMA               | ❌     | ✅              | ✅      | ❌            | ❌      | ❌          |
| AMP[1]            | ✅     | ✅              | ✅      | ✅            | ✅      | ❌          |
| Distillation[2]   | ❌     | ❌              | ❌      | ✅            | ❌      | ❌          |

[0] Quantization needs to work with EMA/AMP (disable them both when quantization is activated)

[1] AMP implementation details:
1. Use wrap_optim argument of manager.modify `manager.modify(..., wrap_optim=scaler)`
2. Call `scaler._enabled = False` when using quantization is enabled

[2] Distillation implementation details
1. Pass `distillation_teacher=...` to `manager.initializer`
2. Call `loss = manager.loss_update(...)` after loss is computed

| Datasets                | NM IC | torchvision IC | yolov5 | transformers | yolact | openpifpaf |
| ----------------------- | ----- | -------------- | ------ | ------------ | ------ | ---------- |
| Use standard datasets   | ✅     | ✅              | ✅      | ✅            | ✔️      | ✔️          |
| train/val/test datasets | ✅     | ✅              | ✅      | ✅            | ✔️      | ✔️          |
| Auto download datasets  | ✅     | ❌              | ✅      | ✅            | ❌      | ✔️          |

| Checkpoints                           | NM IC | torchvision IC | yolov5 | transformers | yolact | openpifpaf |
| ------------------------------------- | ----- | -------------- | ------ | ------------ | ------ | ---------- |
| Checkpoints from original integration | ✅     | ✅              | ✅      | ✅            | ✅      | ✅          |
| Checkpoints from sparsezoo            | ✅     | ✅              | ✅      | ✅            | ✅      | ✅          |
| Best checkpoint                       | ✅     | ✅              | ✅      | ✅            | ✅      | ✅          |
| Best Pruned checkpoint[0]             | ❌     | ❌              | ❌      | ❌            | ❌      | ❌          |
| Best Quantized checkpoint             | ❌     | ❌              | ❌      | ❌            | ❌      | ❌          |
| Best Pruned/Quantized checkpoint      | ❌     | ❌              | ❌      | ❌            | ❌      | ❌          |
| Changes architecture from recipe[1]   | ✅     | ✅              | ✅      | ✅            | ✅      | ✅          |
| Staged recipes [2]                    | ✅     | ✅              | ✅      | ✅            | ✅      | ✅          |

[0] Can only be saved after pruning fully completes

[1] If the checkpoint has a completed recipe, then need to call `manager.apply_structure(model, checkpoint_epoch)`

[2] Needs to use `manager.compose_staged(...)` on checkpoint recipe & current recipe

| Logging[0]         | NM IC | torchvision IC | yolov5 | transformers | yolact | openpifpaf |
| ------------------ | ----- | -------------- | ------ | ------------ | ------ | ---------- |
| stdout             | ✅     | ✅              | ✅      | ✅            | ✅      | ✔️          |
| Weights and Biases | ❌     | ✅              | ✅      | ✅            | ✅      | ❌          |
| TensorBoard        | ✅     | ✅              | ✅      | ✅            | ✅      | ❌          |

[0] The units for x axis in logging should be number of optimizer steps. Notably: `num_optimizer_steps = num_batches / gradient_accum_steps`. So when gradient_accumuluation is not used, the x axis will be number of batches trained on. 

# Export

| Exporting                           | NM IC | torchvision IC | yolov5 | transformers | yolact | openpifpaf |
| ----------------------------------- | ----- | -------------- | ------ | ------------ | ------ | ---------- |
| Optional one shot                   | ✅     | ❌              | ✅      | ✅            | ✅      | ✅          |
| Convert to ONNX[0]                  | ✅     | ✅              | ✅      | ✅            | ✅      | ✅          |
| Convert to TorchScript              | ❌     | ❌              | ✔️      | ❌            | ❌      | ❌          |
| Static batch size                   | ❌     | ❌              | ✔️      | ❌            | ❌      | ❌          |
| Dynamic batch size                  | ✅     | ✅              | ✅      | ✅            | ✅      | ✅          |
| Static input shape                  | ✅     | ✅              | ✅      | ✅            | ✅      | ✅          |
| Dynamic input shape                 | ❌     | ❌              | ❌      | ❌            | ❌      | ❌          |
| Save to simple deployment folder[1] | ✅     | ❌              | ✅      | ✅            | ✅      | ✅          |
| Save to SparseZoo folder            | ❌     | ❌              | ❌      | ❌            | ❌      | ❌          |

[0] Should use our `ModuleExporter`

[1] Should only require specifying checkpoint path and necessary configuration files
