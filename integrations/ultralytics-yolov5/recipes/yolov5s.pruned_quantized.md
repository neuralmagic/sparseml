---
# General Hyperparams
num_epochs: &num_epochs 302
init_lr: &init_lr 0.01
final_lr: &final_lr 0.002
weights_warmup_lr: &weights_warmup_lr 0
biases_warmup_lr: &biases_warmup_lr 0.1
quantization_lr: &quantization_lr 0.000002

# Pruning Hyperparams
init_sparsity: &init_sparsity 0.05
pruning_start_epoch: &pruning_start_epoch 4
pruning_end_epoch: &pruning_end_epoch 100
update_frequency: &pruning_update_frequency 0.2
mask_type: &mask_type [1, 4]
prune_none_target_sparsity: &prune_none_target_sparsity 0.4
prune_low_target_sparsity: &prune_low_target_sparsity 0.5
prune_mid_target_sparsity: &prune_mid_target_sparsity 0.65
prune_high_target_sparsity: &prune_high_target_sparsity 0.75

# Quantization Params
quantization_start_epoch: &quantization_start_epoch 300

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
    
  - !SetLearningRateModifier
    start_epoch: *quantization_start_epoch
    learning_rate: *quantization_lr
    
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
    mask_type: *mask_type
        
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
    mask_type: *mask_type
        
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
    mask_type: *mask_type
        
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
    mask_type: *mask_type
             
quantization_modifiers:
  - !QuantizationModifier
    start_epoch: *quantization_start_epoch
    submodules: [ 'model.0', 'model.1', 'model.2', 'model.3', 'model.4', 'model.5', 'model.6', 'model.7', 'model.8', 'model.9', 'model.10', 'model.11', 'model.12', 'model.13', 'model.14', 'model.15', 'model.16', 'model.17', 'model.18', 'model.19', 'model.20', 'model.21', 'model.22', 'model.23' ]
---

# YOLOv5s Pruned Quantized
