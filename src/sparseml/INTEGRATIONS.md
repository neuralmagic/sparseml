# Training

| Base | SparseML IC | torchvision IC |
| --- | --- | --- |
| CLI | ✅ | ✅ |
| API | ✅ | ✅ |
| Dense training | ✅ | ✅ |

| Sparsification | SparseML IC | torchvision IC |
| --- | --- | --- |
| Recipe | ✅ | ✅ |
| Recipe args | ✅ | ❌ |
| Recipe is optional | ✅ | ❌ |
| Pruning from a recipe | ✅ | ✅ |
| Pruning with EMA | ❌ | ✅ |
| Quantization from a recipe | ✅ | ✅ |
| Qauntization with AMP | ✅ | ✅ |
| Quantization with EMA | ❌ | ✅ |
| Distillation support | ❌ | ❌ |

| Datasets | SparseML IC | torchvision IC |
| --- | --- | --- |
| Use standard datasets | ✅ | ✅ |
| Support train/validation/test datasets | ✅ | ✅ |
| Auto download datasets | ✅ | ❌ |

| Checkpoints | SparseML IC | torchvision IC |
| --- | --- | --- |
| Model checkpoints from original integration | ✅ | ✅ |
| Model checkpoints from sparsezoo | ✅ | ✅ |
| Checkpoints store all necessary configuration | ✅ | ✅ |
| Best checkpoint saved | ✅ | ✅ |
| Best Pruned checkpoint saved | ❌ | ❌ |
| Best Quantized checkpoint saved | ❌ | ❌ |
| Best Pruned/Quantized checkpoint saved | ❌ | ❌ |
| Architecture changes from saved recipe | ✅ | ✅ |
| Staged recipes | ✅ | ✅ |

| Logging | SparseML IC | torchvision IC |
| --- | --- | --- |
| Log to stdout | ✅ | ❌ |
| Log to Weights and Biases | ✅ | ❌ |
| Log to TensorBoard | ✅ | ❌ |

# Export

| Exporting | SparseML IC | torchvision IC |
| --- | --- | --- |
| Optional one shot application of a recipe | ✅ | ❌ |
| Convert to ONNX | ✅ | ✅ |
| ONNX graph validation | ✅ | ✅ |
| Graph folding/optimization for DeepSparse targets | ✅ | ✅ |
| Convert to TorchScript | ❌ | ❌ |
| Batch size fixed or dynamic | | |
| Input shape fixed or dynamic | | |
| Save to simple deployment folder | ✅ | ✅ |
| Save to SparseZoo folder | ❌ | ❌ |

