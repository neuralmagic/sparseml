---

version: 1.1.0

# General Variables
num_epochs: 3
lr_warmup_epochs: 0
init_lr: 0.0512
warmup_lr: 0.256
weight_decay: 0.00001

# Quantization variables
quantization_epochs: 2
quantization_start_epoch: eval(num_epochs - quantization_epochs)
quantization_end_epoch: eval(num_epochs)
quantization_constant_lr_epochs: 1
quantization_init_lr: 0.00001
quantization_final_lr: 0.000001
quantization_cycle_epochs: 1
quantization_observer_epochs: 1
quantization_keep_bn_epochs: 1

# Pruning Variables
pruning_epochs_fraction: 0.925
pruning_start_epoch: 0
pruning_end_epoch: 1
pruning_update_frequency: 0.5
pruning_sparsity: 0.85
pruning_init_sparsity: 0.6
pruning_final_lr: 0.0


training_modifiers:
  - !EpochRangeModifier
    start_epoch: 0
    end_epoch: eval(num_epochs)

  - !SetWeightDecayModifier
    start_epoch: 0
    weight_decay: eval(weight_decay)

pruning_modifiers:
  - !MagnitudePruningModifier
    start_epoch: eval(pruning_start_epoch)
    end_epoch: eval(pruning_end_epoch)
    init_sparsity: eval(pruning_init_sparsity)
    final_sparsity: eval(pruning_sparsity)
    update_frequency: eval(pruning_update_frequency)
    params: __ALL_PRUNABLE__

quantization_modifiers:
  - !SetLearningRateModifier
    start_epoch: eval(quantization_start_epoch)
    learning_rate: eval(quantization_init_lr)

  - !LearningRateFunctionModifier
    final_lr: 0.0
    init_lr: eval(quantization_init_lr)
    lr_func: cosine
    start_epoch: eval(quantization_start_epoch + quantization_keep_bn_epochs)
    end_epoch: eval(num_epochs)

  - !QuantizationModifier
    start_epoch: eval(quantization_start_epoch)
    submodules:
      - input
      - sections
    disable_quantization_observer_epoch: eval(quantization_start_epoch + quantization_observer_epochs)
    freeze_bn_stats_epoch: eval(quantization_start_epoch + quantization_keep_bn_epochs)
    quantize_conv_activations: False
---