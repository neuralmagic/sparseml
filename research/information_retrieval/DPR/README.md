# Compressing DPR
Author: @spacemanidol

Methods
1. Varying models
2. Sturctured Pruning
3. Unstructured Pruning
4. Dimensionality Reduction
## Usage
batch_size: 4
dev_batch_size: 16
adam_eps: 1e-8
adam_betas: (0.9, 0.999)
max_grad_norm: 2.0
log_batch_step: 1
train_rolling_loss_step: 100
weight_decay: 0.0
learning_rate: 2e-5
# Linear warmup over warmup_steps.
warmup_steps: 1237

# Number of updates steps to accumulate before performing a backward/update pass.
gradient_accumulation_steps: 1

# Total number of training epochs to perform.
num_train_epochs: 40
eval_per_epoch: 1
hard_negatives: 1
other_negatives: 0
val_av_rank_hard_neg: 30
val_av_rank_other_neg: 30
val_av_rank_bsz: 128
val_av_rank_max_qs: 10000

## Results

| Top-k passages        | Original DPR NQ model           | New DPR model  |
| ------------- |:-------------:| -----:|
| 1      | 45.87 | 52.47 |
| 5      | 68.14      |   72.24 |
| 20  | 79.97      |    81.33 |
| 100  | 85.87      |    87.29 |
### requirements.txt
