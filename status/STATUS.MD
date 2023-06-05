#  SparseML Integration Project Status Page
Feature status tables related to required and target features for SparseML sparsification aware training integrations

## Base Training


|                           | NM IC              | OpenPifPaf         | YOLOv5             | Transformers       | Torchvision IC     |
| ------------------------- | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ |
| **cli**                   | :white_check_mark: | :heavy_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| **api**                   | :white_check_mark: | :heavy_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| **dense_training**        | :white_check_mark: | :heavy_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| **gradient_accumulation** | :x:                | :x:                | :white_check_mark: | :heavy_check_mark: | :white_check_mark: |
| **DP**                    | :white_check_mark: | :heavy_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| **DDP**                   | :white_check_mark: | :heavy_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |

## Sparsification
Features related to sparsification integration. Notes: 
* Recipe support should be optional
* AMP must be disabled during QAT. (`scaler._enabled = False`)
Distillation:
* distillation_teacher kwarg must be passed to manager initialzation
* Call loss = manager.loss_update(...) after loss is computed

|                  | NM IC              | OpenPifPaf         | YOLOv5             | Transformers       | Torchvision IC     |
| ---------------- | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ |
| **recipe**       | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| **recipe_args**  | :white_check_mark: | :x:                | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| **EMA**          | :x:                | :x:                | :white_check_mark: | :x:                | :white_check_mark: |
| **AMP**          | :white_check_mark: | :x:                | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| **distillation** | :x:                | :x:                | :white_check_mark: | :white_check_mark: | :white_check_mark: |

## Datasets


|                             | NM IC              | OpenPifPaf         | YOLOv5             | Transformers       | Torchvision IC     |
| --------------------------- | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ |
| **use_standard_datasets**   | :white_check_mark: | :heavy_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| **train_val_test_datasets** | :white_check_mark: | :heavy_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| **auto_download_datasets**  | :white_check_mark: | :heavy_check_mark: | :white_check_mark: | :heavy_check_mark: | :x:                |

## Checkpoints
Features related to checkpoints. Notes: 
* best_* checkpoints can only be saved after the entire sparsification step completes
* update_architecture_from_recipe requires a call to apply_structure() on a torch model before loading sparsified checkpoint
* staged_recipes requires manager.compose_staged(...) before checkpoint save

|                                      | NM IC              | OpenPifPaf         | YOLOv5             | Transformers       | Torchvision IC     |
| ------------------------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ |
| **original_integration_checkpoints** | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| **sparsezoo_checkpoints**            | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| **best_checkpoint**                  | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| **best_pruned_checkpoint**           | :x:                | :x:                | :question:         | :x:                | :x:                |
| **best_pruned_quantized_checkpoint** | :x:                | :x:                | :question:         | :x:                | :x:                |
| **recipe_saved_to_checkpoint**       | :x:                | :x:                | :question:         | :x:                | :x:                |
| **update_architecture_from_recipe**  | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| **staged_recipes**                   | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |

## Logging
Logging units for x axis in logging should be number of optimizer steps. Notably: `num_optimizer_steps = num_batches / gradient_accum_steps`. So when gradient_accumuluation is not used, the x axis will be number of batches trained on.

|                        | NM IC              | OpenPifPaf         | YOLOv5             | Transformers       | Torchvision IC     |
| ---------------------- | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ |
| **stdout**             | :white_check_mark: | :heavy_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| **weights_and_biases** | :x:                | :x:                | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| **tensorboard**        | :white_check_mark: | :x:                | :white_check_mark: | :white_check_mark: | :white_check_mark: |

## Export
PyTorch export features should use `ModuleExporter` and only require specifying checkpoint path and necessary configuration files

|                                         | NM IC              | OpenPifPaf         | YOLOv5             | Transformers       | Torchvision IC     |
| --------------------------------------- | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ |
| **cli**                                 | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| **api**                                 | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| **one_shot**                            | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| **onnx**                                | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| **torch_script**                        | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                |
| **static_batch_size**                   | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                |
| **dynamic_batch_size**                  | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| **static_input_shape**                  | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| **dynamic_input_shape**                 | :x:                | :x:                | :x:                | :x:                | :x:                |
| **save_to_simple_deployment_directory** | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| **save_to_sparsezoo_directory**         | :x:                | :x:                | :x:                | :x:                | :x:                |

### Key
 * :white_check_mark: - implemented by neuralmagic integration
 * :heavy_check_mark: - implemented by underlying integration
 * :x: - not implemented yet
 * :question: - not sure, not tested, or to be investigated