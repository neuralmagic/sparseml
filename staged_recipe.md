---
high_level_variable1: 1
high_level_variable2: 2

sparsification_stage:
  num_epochs: 13
  init_lr: 1.5e-4 
  final_lr: 0

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
        quantize_embeddings: eval(quantize_embeddings)
        quantize_linear_activations: 0
        exclude_module_types: ['LayerNorm', 'Tanh']
        submodules:
          - bert.embeddings
          - bert.encoder
          - qa_outputs

next_stage:
  new_variable: 1
  new_num_epochs: 2

  next_stage_modifiers:
    - !EpochRangeModifier
        end_epoch: eval(new_num_epochs + new_variable + high_level_variable)
        start_epoch: 0.0
---