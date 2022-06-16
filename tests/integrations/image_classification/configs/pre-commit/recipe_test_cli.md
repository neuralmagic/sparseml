---
# General Epoch/LR variables
num_epochs: &num_epochs 10.0
init_lr: &init_lr 0.0028
lr_step_epochs: &lr_step_epochs [43, 60]

# Pruning variables
pruning_start_epoch: &pruning_start_epoch 0.0
pruning_end_epoch: &pruning_end_epoch 8.0
pruning_update_frequency: &pruning_update_frequency 0.4
init_sparsity: &init_sparsity 0.05
final_sparsity: &final_sparsity 0.8
pruning_mask_type: &pruning_mask_type unstructured

# Modifiers
training_modifiers:
  - !EpochRangeModifier
    end_epoch: *num_epochs
    start_epoch: 0.0

  - !LearningRateModifier
    constant_logging: False
    end_epoch: -1.0
    init_lr: *init_lr
    log_types: __ALL__
    lr_class: MultiStepLR
    lr_kwargs: {'milestones': *lr_step_epochs, 'gamma': 0.1}
    start_epoch: 0.0
    update_frequency: -1.0
        
pruning_modifiers:
  - !GMPruningModifier
    end_epoch: *pruning_end_epoch
    final_sparsity: *final_sparsity
    init_sparsity: *init_sparsity
    inter_func: cubic
    leave_enabled: True
    log_types: __ALL__
    mask_type: *pruning_mask_type
    params: ['sections.0.0.conv1.weight', 'sections.0.0.conv2.weight', 'sections.0.0.conv3.weight', 'sections.0.0.identity.conv.weight', 'sections.0.1.conv1.weight', 'sections.0.1.conv2.weight', 'sections.0.1.conv3.weight', 'sections.0.2.conv1.weight', 'sections.0.2.conv2.weight', 'sections.0.2.conv3.weight', 'sections.1.0.conv1.weight', 'sections.1.0.conv2.weight', 'sections.1.0.conv3.weight', 'sections.1.0.identity.conv.weight', 'sections.1.1.conv1.weight', 'sections.1.1.conv2.weight', 'sections.1.1.conv3.weight', 'sections.1.2.conv1.weight', 'sections.1.2.conv2.weight', 'sections.1.2.conv3.weight', 'sections.1.3.conv1.weight', 'sections.1.3.conv2.weight', 'sections.1.3.conv3.weight', 'sections.2.0.conv1.weight', 'sections.2.0.conv2.weight', 'sections.2.0.conv3.weight', 'sections.2.0.identity.conv.weight', 'sections.2.1.conv1.weight', 'sections.2.1.conv2.weight', 'sections.2.1.conv3.weight', 'sections.2.2.conv1.weight', 'sections.2.2.conv2.weight', 'sections.2.2.conv3.weight', 'sections.2.3.conv1.weight', 'sections.2.3.conv2.weight', 'sections.2.3.conv3.weight', 'sections.2.4.conv1.weight', 'sections.2.4.conv2.weight', 'sections.2.4.conv3.weight', 'sections.2.5.conv1.weight', 'sections.2.5.conv2.weight', 'sections.2.5.conv3.weight', 'sections.3.0.conv1.weight', 'sections.3.0.conv2.weight', 'sections.3.0.conv3.weight', 'sections.3.0.identity.conv.weight', 'sections.3.1.conv1.weight', 'sections.3.1.conv2.weight', 'sections.3.1.conv3.weight', 'sections.3.2.conv1.weight', 'sections.3.2.conv2.weight', 'sections.3.2.conv3.weight']
    start_epoch: *pruning_start_epoch
    update_frequency: *pruning_update_frequency
---