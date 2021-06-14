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

# Sparsifying Bert Models Using Recipes

This tutorial presents essential extension from SparseML to HuggingFace Transformers training workflow to support model sparsification that includes knowledge distillation, parameter pruning, quantization and layer dropping. The examples used in this tutorial are specifically for Bert base uncased model trained and pruned on the Squad dataset, and further support and results will be available for other dataset in the near future.

All the results listed in this tutorials are available publically through a [Weights and Biases project](https://wandb.ai/neuralmagic/sparse-bert-squad?workspace=user-neuralmagic).


## Creating a Pretrained Teacher Models

Before applying one of the pruning recipes with distillation approach, we need a "teacher" model pretrained on the dataset. In our experiments, we trained the Bert model adapted to Squad in two epochs, resulting in a teacher model with EM/F1 metrics of 80.9/88.4. The `run_qa.py` script could be used for this purpose as follows.

```bash
python transformers/examples/pytorch/question-answering/run_qa.py  \
  --model_name_or_path bert-base-uncased \
  --dataset_name squad \
  --do_train \
  --do_eval \
  --evaluation_strategy epoch \
  --per_device_train_batch_size 16 \
  --learning_rate 3e-5 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir MODELS_DIR/bert-base-12layers \
  --cache_dir cache \
  --preprocessing_num_workers 8 \
  --fp16 \
  --num_train_epochs 2 \
  --warmup_steps 5400 \
  --report_to wandb
```

If the command runs successfully, you should have a model folder called `bert-base-12layers` in the provided model directory `MODELS_DIR`.

## Applying Pruning Recipes

Using the teacher model `bert-base-12layers` above, you can now train and prune a "student" Bert model on the same dataset using knowledge distillation. `SparseML` extends the training script `run_qa.py` with the following arguments to support recipes and knowledge distillation:

- `--recipe`: path to a YAML recipe file that defines, among other information, the parameters and the desired sparsity levels to prune;
- `--distill_teacher`: path to the teacher model for distillation; the student model is trained to learn from both its correct targets and those "instructed" by the teacher model;
- `--distill_hardness`: ratio (in `[0.0, 1.0]`) of the loss defined on the teacher model targets;
- `--distill_temperature`: the temperature used to soften the distribution of the targets.

Additionally, you will use the argument `--onnx_export_path` to specify the destination folder for the exported ONNX model. The resulting exported model could then be used for inference with the `DeepSparse` engine.

The following command prunes the model in 30 epochs to 80% sparsity of the encoder layers:

```bash
python transformers/examples/pytorch/question-answering/run_qa.py \
  --distill_teacher MODELS_DIR/bert-base-12layers \
  --model_name_or_path bert-base-uncased \
  --dataset_name squad \
  --do_train \
  --do_eval \
  --evaluation_strategy epoch \
  --per_device_train_batch_size 16 \
  --learning_rate 5e-5 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir MODELS_DIR/bert-base-12layers_prune80 \
  --cache_dir cache \
  --preprocessing_num_workers 6 \
  --fp16 \
  --num_train_epochs 30 \
  --distill_hardness 1.0 \
  --distill_temperature 2.0 \
  --recipe ../recipes/bert-base-12layers_prune80.md \
  --onnx_export_path MODELS_DIR/bert-base-12layers_prune80/onnx \
  --report_to wandb
```

The directory `recipes` contains information about recipes and training commands used to produce our Bert pruned models on the Squad dataset. 

### Dropping Layers

In some situations, you might drop certain layers from a Bert model and retrain on your own dataset. `SparseML` supports these use cases with a modifier called `LayerPruningModifier` that can be used as part of a pruning recipe. As an example, below is an example modifier that prunes layers 5th and 7th from a Bert model:
```
!LayerPruningModifier
 layers: ['bert.encoder.layer.5', bert.encoder.layer.7']
```

The directory `recipes` contains recipes, for example `bert-base-6layers_prune80`, that drops six layers from a model before applying pruning.

The following table presents the recipes in the directory, the corresponding results and `wandb` logging for our pruned Bert models.

| Recipe name | Description | EM / F1 | DeepSparse performance | Weight and Biases |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| bert-base-12layers | Bert fine-tuned on Squad | 80.927 / 88.435 |  | [wandb](https://wandb.ai/neuralmagic/sparse-bert-squad/runs/w3b1ggyq?workspace=user-neuralmagic) |
| bert-base-12layers_prune80 | Prune baseline model fine-tuned on Squad at 80% sparsity of encoder units | 81.372 / 88.62 |  | [wandb](https://wandb.ai/neuralmagic/sparse-bert-squad/runs/18qdx7b3?workspace=user-neuralmagic) |
| bert-base-12layers_prune90 | Prune baseline model fine-tuned on Squad at 90% sparsity of encoder units | 79.376 / 87.229 |  | [wandb](https://wandb.ai/neuralmagic/sparse-bert-squad/runs/2ht2eqsn?workspace=user-neuralmagic) |
| bert-base-12layers_prune95 | Prune baseline model fine-tuned on Squad at 95% sparsity of encoder units | 74.939 / 83.929 |  | [wandb](https://wandb.ai/neuralmagic/sparse-bert-squad/runs/3gv0arxd?workspace=user-neuralmagic) |
| bert-base-6layers_prune80 | Prune 6-layer model fine-tuned on Squad at 80% sparsity of encoder units | 78.042 / 85.915 |  | [wandb](https://wandb.ai/neuralmagic/sparse-bert-squad/runs/ebab4np4?workspace=user-neuralmagic) |
| bert-base-6layers_prune90 | Prune 6-layer model fine-tuned on Squad at 90% sparsity of encoder units | 75.08 / 83.602 |  | [wandb](https://wandb.ai/neuralmagic/sparse-bert-squad/runs/3qvxoroz?workspace=user-neuralmagic) |
| bert-base-6layers_prune95 | Prune 6-layer model fine-tuned on Squad at 95% sparsity of encoder units | 70.946 / 80.483 |  | [wandb](https://wandb.ai/neuralmagic/sparse-bert-squad/runs/3plynclw?workspace=user-neuralmagic) |


## Exporting for Inference

The sparsification run with the argument `--export_onnx_path` will creates an ONNX model that can be used for benchmarking with `DeepSparse` engine. You can export a model as part of the training and pruning process (as in the commands above), or after the model is pruned. 

The following command evaluate a pruned model and convert it into ONNX format:

```bash
python transformers/examples/pytorch/question-answering/run_qa.py \
  --model_name_or_path MODELS_DIR/bert-base-12layers_prune80 \
  --dataset_name squad \
  --do_eval \
  --per_device_eval_batch_size 64 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --cache_dir cache \
  --preprocessing_num_workers 6 \
  --onnx_export_path MODELS_DIR/bert-base-12layers_prune80/onnx \
```

If it runs successfully, you will have the converted `model.onnx` in `MODELS_DIR/bert-base-12layers_prune80/onnx`. You can now run it in ONNX-compatible inference engines such as [DeepSparse](https://github.com/neuralmagic/deepsparse). The DeepSparse Engine is explicitly coded to support running sparsified models for significant improvements in inference performance.

