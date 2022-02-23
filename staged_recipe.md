---
high_level_variable: 1

sparsification_stage:
  num_epochs: 13
  init_lr: 1.5e-4 
  final_lr: 0
  qat_epochs: 3 
  qat_no_observer_epochs: 1

  # Modifiers:
  training_modifiers:
    - !EpochRangeModifier
        end_epoch: eval(num_epochs + high_level_variable)
        start_epoch: 0.0

    - !LearningRateFunctionModifier
      start_epoch: 0
      end_epoch: eval(num_epochs)
      lr_func: linear
      init_lr: eval(init_lr)
      final_lr: eval(final_lr)

  pruning_modifiers:
    - !ConstantPruningModifier
        start_epoch: 0.0
        params: __ALL_PRUNABLE__

  quantization_modifiers:
    - !QuantizationModifier
        start_epoch: eval(num_epochs - qat_epochs)
        disable_quantization_observer_epoch: eval(num_epochs - qat_no_observer_epochs)
        freeze_bn_stats_epoch: eval(num_epochs - qat_no_observer_epochs)
        quantize_linear_activations: 0
        exclude_module_types: ['LayerNorm', 'Tanh']

next_stage:
  new_variable: 1
  new_num_epochs: 2

  next_stage_modifiers:
    - !EpochRangeModifier
        end_epoch: eval(new_num_epochs + new_variable + high_level_variable)
        start_epoch: 0.0
---