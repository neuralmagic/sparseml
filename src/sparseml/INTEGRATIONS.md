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
| Pruning with EMA | | :white_check_mark: |
| Quantization from a recipe | :white_check_mark: | :white_check_mark: |
| Qauntization with AMP | | :white_check_mark: |
| Quantization with EMA | | :white_check_mark: |
| Distillation support | :x: | :x: |

| Datasets | SparseML IC | torchvision IC |
| --- | --- | --- |
| Use standard datasets | :white_check_mark: | :white_check_mark: |
| Support train/validation/test datasets | :white_check_mark: | :white_check_mark: |
| Auto download datasets | :white_check_mark: | :x: |

| Checkpoints | SparseML IC | torchvision IC |
| --- | --- | --- |
| Model checkpoints from original integration | | |
| Model checkpoints from sparsezoo | | |
| Checkpoints store all necessary configuration | | |
| Best checkpoint saved | | |
| Best Pruned checkpoint saved | | |
| Best Quantized checkpoint saved | | |
| Best Pruned/Quantized checkpoint saved | | |
| Architecture changes from saved recipe | | |
| Staged recipes | | |

| Logging | SparseML IC | torchvision IC |
| --- | --- | --- |
| Log to stdout | :white_check_mark: | :x: |
| Log to Weights and Biases | :white_check_mark: | :x: |
| Log to TensorBoard | :white_check_mark: | :x: |

# Export

| Exporting | SparseML IC | torchvision IC |
| --- | --- | --- |
| Optional one shot application of a recipe | | |
| Convert to ONNX | | |
| ONNX graph validation | | |
| Graph folding/optimization for DeepSparse targets | | |
| Convert to TorchScript | | |
| Batch size fixed or dynamic | | |
| Input shape fixed or dynamic | | |
| Save to simple deployment folder | | |
| Save to SparseZoo folder | | 

