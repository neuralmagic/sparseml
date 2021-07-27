---
# General Epoch/LR Hyperparams
num_epochs: &num_epochs 52
init_lr: &init_lr 0.0032
final_lr: &final_lr 0.000384
warmup_epochs: &warmup_epochs 2
weights_warmup_lr: &weights_warmup_lr 0
biases_warmup_lr: &biases_warmup_lr 0.05
quantization_lr: &quantization_lr 0.000002

# Quantization Params
quantization_start_epoch: &quantization_start_epoch 50

# modifiers
training_modifiers:
  - !EpochRangeModifier
    start_epoch: 0.0
    end_epoch: *num_epochs
    
  - !LearningRateFunctionModifier
    start_epoch: *warmup_epochs
    end_epoch: *num_epochs
    lr_func: cosine
    init_lr: *init_lr
    final_lr: *final_lr
    
  - !LearningRateFunctionModifier
    start_epoch: 0
    end_epoch: *warmup_epochs
    lr_func: linear
    init_lr: *weights_warmup_lr
    final_lr: *init_lr
    param_groups: [0, 1]
    
  - !LearningRateFunctionModifier
    start_epoch: 0
    end_epoch: *warmup_epochs
    lr_func: linear
    init_lr: *biases_warmup_lr
    final_lr: *init_lr
    param_groups: [2]
    
  - !SetLearningRateModifier
    start_epoch: *quantization_start_epoch
    learning_rate: *quantization_lr

pruning_modifiers:
  - !ConstantPruningModifier
    start_epoch: 0.0
    params: __ALL_PRUNABLE__
    
quantization_modifiers:
  - !QuantizationModifier
    start_epoch: *quantization_start_epoch
    submodules: [ 'model.0', 'model.1', 'model.2', 'model.3', 'model.4', 'model.5', 'model.6', 'model.7', 'model.8', 'model.9', 'model.10', 'model.11', 'model.12', 'model.13', 'model.14', 'model.15', 'model.16', 'model.17', 'model.18', 'model.19', 'model.20', 'model.21', 'model.22', 'model.23' ]
---

# YOLOv5 Pruned Quantized Transfer Learning