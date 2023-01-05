# Training

| Base | SparseML IC | torchvision IC |
| --- | --- | --- |
| CLI | :white_check_mark: | :white_check_mark: |
| API | :white_check_mark: | :white_check_mark: |
| Dense training | :white_check_mark: | :white_check_mark: |

| Sparsification | SparseML IC | torchvision IC |
| --- | --- | --- |
| Optional recipe/recipe_args | :white_check_mark: | |
| Pruning from a recipe | :white_check_mark: | :white_check_mark: |
| Pruning with EMA | :x: | :white_check_mark: |
| Quantization from a recipe | :white_check_mark: | :white_check_mark: |
| Qauntization with AMP | :white_check_mark: | :white_check_mark: |
| Quantization with EMA | :x: | :white_check_mark: |
| Distillation support | :x: | :x: |

| Datasets | SparseML IC | torchvision IC |
| --- | --- | --- |
| Use standard datasets | :white_check_mark: | :white_check_mark: |
| Support train/validation/test datasets | :white_check_mark: | :white_check_mark: |
| Auto download datasets | :white_check_mark: | :x: |

| Checkpoints | SparseML IC | torchvision IC |
| --- | --- | --- |
| Model checkpoints from original integration | :white_check_mark: | :white_check_mark: |
| Model checkpoints from sparsezoo | :white_check_mark: | :white_check_mark: |
| Checkpoints store all necessary configuration | :white_check_mark: | :white_check_mark: |
| Best checkpoint saved | :white_check_mark: | :white_check_mark: |
| Best Pruned checkpoint saved | :x: | :x: |
| Best Quantized checkpoint saved | :x: | :x: |
| Best Pruned/Quantized checkpoint saved | :x: | :x: |
| Architecture changes from saved recipe | :white_check_mark: | :white_check_mark: |
| Staged recipes | :white_check_mark: | :white_check_mark: |

| Logging | SparseML IC | torchvision IC |
| --- | --- | --- |
| Log to stdout | :white_check_mark: | :x: |
| Log to Weights and Biases | :white_check_mark: | :x: |
| Log to TensorBoard | :white_check_mark: | :x: |

# Export

| Exporting | SparseML IC | torchvision IC |
| --- | --- | --- |
| Optional one shot application of a recipe | :white_check_mark: | :x: |
| Convert to ONNX | :white_check_mark: | :white_check_mark: |
| ONNX graph validation | :white_check_mark: | :white_check_mark: |
| Graph folding/optimization for DeepSparse targets | :white_check_mark: | :white_check_mark: |
| Convert to TorchScript | :x: | :x: |
| Batch size fixed or dynamic | | |
| Input shape fixed or dynamic | | |
| Save to simple deployment folder | :white_check_mark: | :white_check_mark: |
| Save to SparseZoo folder | :x: | :x: |

