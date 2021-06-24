<!--
Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

---
# General Variables
num_epochs: &num_epochs 30

# Pruning Hyperparameters
init_sparsity: &init_sparsity 0.00
final_sparsity: &final_sparsity 0.80
pruning_start_epoch: &pruning_start_epoch 2
pruning_end_epoch: &pruning_end_epoch 20
update_frequency: &pruning_update_frequency 0.01

# Modifiers
training_modifiers:
  - !EpochRangeModifier
    end_epoch: 30
    start_epoch: 0.0

pruning_modifiers:
  - !GMPruningModifier
    params:
      - re:bert.encoder.layer.([0,2,4,6,8]|11).attention.self.query.weight
      - re:bert.encoder.layer.([0,2,4,6,8]|11).attention.self.key.weight
      - re:bert.encoder.layer.([0,2,4,6,8]|11).attention.self.value.weight
      - re:bert.encoder.layer.([0,2,4,6,8]|11).attention.output.dense.weight
      - re:bert.encoder.layer.([0,2,4,6,8]|11).intermediate.dense.weight
      - re:bert.encoder.layer.([0,2,4,6,8]|11).output.dense.weight
    start_epoch: *pruning_start_epoch
    end_epoch: *pruning_end_epoch
    init_sparsity: *init_sparsity
    final_sparsity: *final_sparsity
    inter_func: cubic
    update_frequency: *pruning_update_frequency
    leave_enabled: True
    mask_type: unstructured
    log_types: __ALL__

  - !LayerPruningModifier
    layers:
      - bert.encoder.layer.1
      - bert.encoder.layer.3
      - bert.encoder.layer.5
      - bert.encoder.layer.7
      - bert.encoder.layer.9
      - bert.encoder.layer.10
---

# BERT Model with Dropped and Pruned Encoder Layers

This recipe defines a dropping and pruning strategy to sparsify six encoder layers of a BERT model at 80% sparsity. It was used together with knowledge distillation to create sparse model that exceeds the F1 metric (83.632) of the baseline model by 0.02% on the SQuAD dataset. (We use the checkpoint at the end of the first 2 epochs as the baseline model for comparison, right before the pruning takes effect.)
Training was done using one V100 GPU at half precision using a training batch size of 16 with the
[SparseML integration with huggingface/transformers](https://github.com/neuralmagic/sparseml/tree/main/integrations/huggingface-transformers).

## Weights and Biases

- [Sparse BERT on SQuAD](https://wandb.ai/neuralmagic/sparse-bert-squad/runs/ebab4np4?workspace=user-neuralmagic)

## Training

To set up the training environment, follow the instructions on the [integration README](https://github.com/neuralmagic/sparseml/blob/main/integrations/huggingface-transformers/README.md).
Using the `run_qa.py` script from the question-answering examples, the following command can be used to launch this recipe with distillation.
Adjust the training command below with your setup for GPU device, checkpoint saving frequency, and logging options.

*training command*
```
python transformers/examples/pytorch/question-answering/run_qa.py \
  --distill_teacher MODELS_DIR/teacher \
  --model_name_or_path bert-base-uncased \
  --dataset_name squad \
  --do_train \
  --fp16 \
  --do_eval \
  --evaluation_strategy epoch \
  --per_device_train_batch_size 16 \
  --learning_rate 5e-5 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir MODELS_DIR/sparse80_6layers \
  --cache_dir cache \
  --preprocessing_num_workers 6 \
  --seed 42 \
  --num_train_epochs 30 \
  --distill_hardness 1.0 \
  --distill_temperature 2.0 \
  --save_steps 1000 \
  --save_total_limit 2 \
  --recipe ../recipes/bert-base-6layers_prune80.md \
  --onnx_export_path MODELS_DIR/sparse80_6layers/onnx \
  --report_to wandb
```
