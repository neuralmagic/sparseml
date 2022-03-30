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
num_epochs: &num_epochs 40
init_lr: &init_lr 0.00005

# Pruning Hyperparameters
init_sparsity: &init_sparsity 0.00
final_sparsity: &final_sparsity 0.9
pruning_start_epoch: &pruning_start_epoch 2
pruning_end_epoch: &pruning_end_epoch 20
update_frequency: &pruning_update_frequency 0.01

# Quantization Hyperparameters
quantization_start_epoch: &quantization_start_epoch 30.0
qat_freeze_epoch: &qat_freeze_epoch 35.0

# Distillation Hyperparams
distill_hardness: &distill_hardness 1.0
distill_temperature: &distill_temperature 2.0

# Modifiers
training_modifiers:
  - !EpochRangeModifier
    end_epoch: *num_epochs
    start_epoch: 0.0
    
  - !LearningRateFunctionModifier
    start_epoch: 0.0
    end_epoch: *quantization_start_epoch
    lr_func: linear
    init_lr: *init_lr
    final_lr: 0.0
    
  # reset LR schedule for QAT
  - !LearningRateFunctionModifier
    start_epoch: *quantization_start_epoch
    end_epoch: *num_epochs
    lr_func: linear
    init_lr: *init_lr
    final_lr: 0.0

pruning_modifiers:
  - !LayerPruningModifier
        end_epoch: -1.0
        layers:
              - bert.encoder.layer.1
              - bert.encoder.layer.3
              - bert.encoder.layer.5
              - bert.encoder.layer.7
              - bert.encoder.layer.9
              - bert.encoder.layer.10
        start_epoch: -1.0
        update_frequency: -1.0

  - !GMPruningModifier
    params:
      - re:bert.encoder.layer.*.attention.self.query.weight
      - re:bert.encoder.layer.*.attention.self.key.weight
      - re:bert.encoder.layer.*.attention.self.value.weight
      - re:bert.encoder.layer.*.attention.output.dense.weight
      - re:bert.encoder.layer.*.intermediate.dense.weight
      - re:bert.encoder.layer.*.output.dense.weight
    start_epoch: *pruning_start_epoch
    end_epoch: *pruning_end_epoch
    init_sparsity: *init_sparsity
    final_sparsity: *final_sparsity
    inter_func: cubic
    update_frequency: *pruning_update_frequency
    leave_enabled: True
    mask_type: [1,4]

quantization_modifiers:
  - !QuantizationModifier
      start_epoch: *quantization_start_epoch
      disable_quantization_observer_epoch: *qat_freeze_epoch
      freeze_bn_stats_epoch: *qat_freeze_epoch
      submodules:
        - bert.embeddings
        - bert.encoder
        - qa_outputs

distillation_modifiers:
  - !DistillationModifier
     hardness: *distill_hardness
     temperature: *distill_temperature
     distill_output_keys: [start_logits, end_logits]
---

# INT8 Quantized BERT Model with Dropped and Pruned Encoder Layers

This recipe defines a dropping, pruning, and quantization strategy to sparsify six encoder
layers of a BERT model at 90% sparsity with INT8 precision. It was used together with knowledge
distillation to create a sparse-quantized model that achieves 91% recovery from the F1 metric of the
baseline model on the SQuAD dataset.
(We use the teacher model fine-tuned for 2 epochs as the baseline for comparison.)
Training was done using one V100 GPU at half precision for pruning and full precision for QAT using a
training batch size of 16 with the
[SparseML integration with huggingface/transformers](https://github.com/neuralmagic/sparseml/tree/main/integrations/huggingface-transformers).

## Weights and Biases

- [Sparse BERT on SQuAD](https://wandb.ai/neuralmagic/huggingface/runs/tkpvq7iz?workspace=user-neuralmagic)

## Training

To set up the training environment, follow the instructions on the [integration README](https://github.com/neuralmagic/sparseml/blob/main/integrations/huggingface-transformers/README.md).
Using the `run_qa.py` script from the question-answering examples, the following command can be used to launch this recipe with distillation.
Adjust the training command below with your setup for GPU device, checkpoint saving frequency, and logging options.

*training command*

python transformers/examples/pytorch/question-answering/run_qa.py \
  --model_name_or_path bert-base-uncased \
  --distill_teacher $MODEL_DIR/bert-base-12layers \
  --dataset_name squad \
  --do_train \
  --do_eval \
  --evaluation_strategy epoch \
  --per_device_train_batch_size 16 \
  --learning_rate 5e-5 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir $MODEL_DIR/sparse90_quant_6layers \
  --cache_dir cache \
  --preprocessing_num_workers 6 \
  --seed 42 \
  --num_train_epochs 40 \
  --recipe ../recipes/bert-base-6layers_prune90_quant.md \
  --onnx_export_path $MODEL_DIR/sparse90_quant_6layers/onnx \
  --save_strategy epoch \
  --save_total_limit 2 \
  --fp16
