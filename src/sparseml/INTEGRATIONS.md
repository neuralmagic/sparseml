# Training

| Base | Image Classification |
| --- | --- |
| CLI | ✅ |
| API | ✅ |
| Dense training | ✅ |

| Sparsification | Image Classification |
| --- | --- |
| Optional recipe/recipe_args | ✅ |
| Pruning from a recipe | |
| Pruning with EMA | |
| Quantization from a recipe | |
| Qauntization with AMP | |
| Quantization with EMA | |
| Distillation support | |

| Datasets | Image Classification |
| --- | --- |
| Use standard datasets | |
| Support train/validation/test datasets | |
| Auto download datasets | | 

| Checkpoints | Image Classification |
| --- | --- |
| Model checkpoints from original integration | |
| Model checkpoints from sparsezoo | |
| Checkpoints store all necessary configuration | |
| Best checkpoint saved | |
| Best Pruned checkpoint saved | |
| Best Quantized checkpoint saved | |
| Best Pruned/Quantized checkpoint saved | |
| Architecture changes are preserved and restored from saved recipe | |
| Staged recipes - apply new recipes to pre-applied recipe | |

| Logging | Image Classification |
| --- | --- |
| Log to stdout | |
| Log to Weights and Biases | |
| Log to TensorBoard | |

# Export

| Exporting | Image Classification |
| --- | --- |
| Optional one shot application of a recipe | |
| Convert to ONNX | |
| ONNX graph validation | |
| Graph folding/optimization for DeepSparse targets | |
| Convert to TorchScript | |
| Batch size fixed or dynamic | |
| Input shape fixed or dynamic | |
| Save to simple deployment folder | |
| Save to SparseZoo folder | |

