---
# General Epoch/LR Hyperparams
num_epochs: &num_epochs 50
init_lr: &init_lr 0.0032
final_lr: &final_lr 0.000384
warmup_epochs: &warmup_epochs 2
weights_warmup_lr: &weights_warmup_lr 0
biases_warmup_lr: &biases_warmup_lr 0.05

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

pruning_modifiers:
  - !ConstantPruningModifier
    start_epoch: 0.0
    params: __ALL_PRUNABLE__
---

# YOLOv5 Pruned Transfer Learning