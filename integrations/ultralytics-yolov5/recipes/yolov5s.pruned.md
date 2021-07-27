---
# General Hyperparams
num_epochs: &num_epochs 240
init_lr: &init_lr 0.01
final_lr: &final_lr 0.002
weights_warmup_lr: &weights_warmup_lr 0
biases_warmup_lr: &biases_warmup_lr 0.1

# Pruning Hyperparams
init_sparsity: &init_sparsity 0.05
pruning_start_epoch: &pruning_start_epoch 4
pruning_end_epoch: &pruning_end_epoch 100
update_frequency: &pruning_update_frequency 1.0
prune_none_target_sparsity: &prune_none_target_sparsity 0.5
prune_low_target_sparsity: &prune_low_target_sparsity 0.6
prune_mid_target_sparsity: &prune_mid_target_sparsity 0.725
prune_high_target_sparsity: &prune_high_target_sparsity 0.8

training_modifiers:
  - !EpochRangeModifier
    start_epoch: 0
    end_epoch: *num_epochs
    
  - !LearningRateFunctionModifier
    start_epoch: 3
    end_epoch: *num_epochs
    lr_func: cosine
    init_lr: *init_lr
    final_lr: *final_lr
    
  - !LearningRateFunctionModifier
    start_epoch: 0
    end_epoch: 3
    lr_func: linear
    init_lr: *weights_warmup_lr
    final_lr: *init_lr
    param_groups: [0, 1]
    
  - !LearningRateFunctionModifier
    start_epoch: 0
    end_epoch: 3
    lr_func: linear
    init_lr: *biases_warmup_lr
    final_lr: *init_lr
    param_groups: [2]
    
pruning_modifiers:
  - !GMPruningModifier
    params:
      - model.2.cv2.conv.weight
      - model.2.cv1.conv.weight
      - model.2.cv3.conv.weight
      - model.2.m.0.cv1.conv.weight
      - model.24.m.0.weight
      - model.24.m.1.weight
      - model.24.m.2.weight
    init_sparsity: *init_sparsity
    final_sparsity: *prune_none_target_sparsity
    start_epoch: *pruning_start_epoch
    end_epoch: *pruning_end_epoch
    update_frequency: *pruning_update_frequency
        
  - !GMPruningModifier
    params:
      - model.6.cv1.conv.weight
      - model.6.cv2.conv.weight
      - model.6.cv3.conv.weight
      - model.13.m.0.cv1.conv.weight
      - model.6.m.0.cv1.conv.weight
      - model.6.m.2.cv1.conv.weight
      - model.6.m.1.cv1.conv.weight
      - model.1.conv.weight
      - model.17.m.0.cv1.conv.weight
      - model.4.cv2.conv.weight
      - model.2.m.0.cv2.conv.weight
      - model.4.cv1.conv.weight
      - model.4.cv3.conv.weight
      - model.4.m.0.cv1.conv.weight
      - model.4.m.2.cv1.conv.weight
      - model.4.m.1.cv1.conv.weight
      - model.8.cv1.conv.weight
    init_sparsity: *init_sparsity
    final_sparsity: *prune_low_target_sparsity
    start_epoch: *pruning_start_epoch
    end_epoch: *pruning_end_epoch
    update_frequency: *pruning_update_frequency
        
  - !GMPruningModifier
    params:
      - model.9.cv3.conv.weight
      - model.6.m.2.cv2.conv.weight
      - model.5.conv.weight
      - model.9.cv1.conv.weight
      - model.6.m.1.cv2.conv.weight
      - model.6.m.0.cv2.conv.weight
      - model.17.m.0.cv2.conv.weight
      - model.9.cv2.conv.weight
      - model.10.conv.weight
      - model.13.cv2.conv.weight
      - model.9.m.0.cv1.conv.weight
      - model.20.m.0.cv1.conv.weight
      - model.13.cv3.conv.weight
      - model.13.cv1.conv.weight
      - model.17.cv3.conv.weight
      - model.14.conv.weight
      - model.4.m.2.cv2.conv.weight
      - model.3.conv.weight
      - model.4.m.1.cv2.conv.weight
      - model.4.m.0.cv2.conv.weight
      - model.17.cv1.conv.weight
      - model.23.m.0.cv1.conv.weight
      - model.20.cv1.conv.weight
      - model.23.cv1.conv.weight
    init_sparsity: *init_sparsity
    final_sparsity: *prune_mid_target_sparsity
    start_epoch: *pruning_start_epoch
    end_epoch: *pruning_end_epoch
    update_frequency: *pruning_update_frequency
        
  - !GMPruningModifier
    params:
      - model.23.m.0.cv2.conv.weight
      - model.21.conv.weight
      - model.23.cv3.conv.weight
      - model.23.cv2.conv.weight
      - model.20.m.0.cv2.conv.weight
      - model.18.conv.weight
      - model.9.m.0.cv2.conv.weight
      - model.7.conv.weight
      - model.20.cv3.conv.weight
      - model.20.cv2.conv.weight
      - model.8.cv2.conv.weight
      - model.13.m.0.cv2.conv.weight
      - model.17.cv2.conv.weight
    init_sparsity: *init_sparsity
    final_sparsity: *prune_high_target_sparsity
    start_epoch: *pruning_start_epoch
    end_epoch: *pruning_end_epoch
    update_frequency: *pruning_update_frequency
---

# YOLOv5s Pruned
