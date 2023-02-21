# NM IC SparseML Integration Project Status Page
Deprecated (in favor of sparseml/pytorch/torchvision) IC training integration

## Base Training


| cli                | api                | dense_training     | gradient_accumulation | DP                 | DDP                |
| ------------------ | ------------------ | ------------------ | --------------------- | ------------------ | ------------------ |
| :white_check_mark: | :white_check_mark: | :white_check_mark: | :x:                   | :white_check_mark: | :white_check_mark: |

## Sparsification
Features related to sparsification integration. Notes: 
* Recipe support should be optional
* AMP must be disabled during QAT. (`scaler._enabled = False`)
Distillation:
* distillation_teacher kwarg must be passed to manager initialzation
* Call loss = manager.loss_update(...) after loss is computed

| recipe             | recipe_args        | EMA | AMP                | distillation |
| ------------------ | ------------------ | --- | ------------------ | ------------ |
| :white_check_mark: | :white_check_mark: | :x: | :white_check_mark: | :x:          |

## Datasets


| use_standard_datasets | train_val_test_datasets | auto_download_datasets |
| --------------------- | ----------------------- | ---------------------- |
| :white_check_mark:    | :white_check_mark:      | :white_check_mark:     |

## Checkpoints
Features related to checkpoints. Notes: 
* best_* checkpoints can only be saved after the entire sparsification step completes
* update_architecture_from_recipe requires a call to apply_structure() on a torch model before loading sparsified checkpoint
* staged_recipes requires manager.compose_staged(...) before checkpoint save

| original_integration_checkpoints | sparsezoo_checkpoints | best_checkpoint    | best_pruned_checkpoint | best_pruned_quantized_checkpoint | recipe_saved_to_checkpoint | update_architecture_from_recipe | staged_recipes     |
| -------------------------------- | --------------------- | ------------------ | ---------------------- | -------------------------------- | -------------------------- | ------------------------------- | ------------------ |
| :white_check_mark:               | :white_check_mark:    | :white_check_mark: | :x:                    | :x:                              | :x:                        | :white_check_mark:              | :white_check_mark: |

## Logging
Logging units for x axis in logging should be number of optimizer steps. Notably: `num_optimizer_steps = num_batches / gradient_accum_steps`. So when gradient_accumuluation is not used, the x axis will be number of batches trained on.

| stdout             | weights_and_biases | tensorboard        |
| ------------------ | ------------------ | ------------------ |
| :white_check_mark: | :x:                | :white_check_mark: |

## Export
PyTorch export features should use `ModuleExporter` and only require specifying checkpoint path and necessary configuration files

| cli                | api                | one_shot           | onnx               | torch_script | static_batch_size | dynamic_batch_size | static_input_shape | dynamic_input_shape | save_to_simple_deployment_directory | save_to_sparsezoo_directory |
| ------------------ | ------------------ | ------------------ | ------------------ | ------------ | ----------------- | ------------------ | ------------------ | ------------------- | ----------------------------------- | --------------------------- |
| :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :x:          | :x:               | :white_check_mark: | :white_check_mark: | :x:                 | :white_check_mark:                  | :x:                         |

### Key
 * :white_check_mark: - implemented by neuralmagic integration
 * :heavy_check_mark: - implemented by underlying integration
 * :x: - not implemented yet
 * :question: - not sure, not tested, or to be investigated