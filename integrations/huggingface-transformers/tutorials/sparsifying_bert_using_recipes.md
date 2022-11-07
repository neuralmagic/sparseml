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

# Sparsifying BERT Models Using Recipes

This tutorial presents an essential extension from SparseML to Hugging Face Transformers training workflow to support model sparsification that includes knowledge distillation, parameter pruning, quantization, and layer dropping. The examples used in this tutorial are specifically for BERT base uncased models, trained, pruned, and quantized on the SQuAD dataset; further support and results will be available for other datasets in the near future.

## Overview
Neural Magic’s ML team creates recipes that allow anyone to plug in their data and leverage SparseML’s recipe-driven approach on top of Hugging Face’s robust training pipelines. Sparsifying involves removing redundant information from neural networks using algorithms such as pruning and quantization, among others. This sparsification process results in many benefits for deployment environments, including faster inference and smaller file sizes. Unfortunately, many have not realized the benefits due to the complicated process and number of hyperparameters involved.

Working through this tutorial, you will experience how Neural Magic recipes simplify the sparsification process by:

- Creating a pre-trained teacher model for knowledge distillation.

- Applying a recipe to select the trade off between the amount of recovery to the baseline training performance with the amount of sparsification for inference performance.

- Exporting a pruned model to the ONNX format to run with an inference engine such as DeepSparse.

All the results listed in this tutorials are available publically through a [Weights and Biases project](https://wandb.ai/neuralmagic/sparse-bert-squad?workspace=user-neuralmagic).

<p float="left">
  <img src="https://github.com/neuralmagic/sparseml/raw/main/integrations/huggingface-transformers/tutorials/images/bert_12_6_layers_F1.png" width="450" height="300">
  <img src="https://github.com/neuralmagic/sparseml/raw/main/integrations/huggingface-transformers/tutorials/images/bert_12_6_layers_EM.png" width="450" height="300">
</p>

## Need Help?
For Neural Magic Support, sign up or log in to our [**Deep Sparse Community Slack**](https://join.slack.com/t/discuss-neuralmagic/shared_invite/zt-q1a1cnvo-YBoICSIw3L1dmQpjBeDurQ). Bugs, feature requests, or additional questions can also be posted to our [GitHub Issue Queue.](https://github.com/neuralmagic/sparseml/issues)

## Creating a Pretrained Teacher Model

Before applying one of the pruning recipes with the distillation approach, we need a "teacher" model pretrained on the dataset. In our experiments, we trained the BERT model adapted to SQuAD in two epochs, resulting in a teacher model with EM/F1 metrics of 80.9/88.4. The `run_qa.py` script could be used for this purpose as follows.

```bash
sparseml.transformers.question_answering  \
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
  --save_strategy epoch
```

If the command runs successfully, you should have a model folder called `bert-base-12layers` in the provided model directory `MODELS_DIR`.

## Applying Pruning Recipes

Using the teacher model `bert-base-12layers` above, you can now train and prune a "student" BERT model on the same dataset using knowledge distillation. `SparseML` extends the training script `run_qa.py` with the following arguments to support recipes and knowledge distillation:

- `--recipe`: path to a YAML recipe file that defines, among other information, the parameters and the desired sparsity levels to prune;
- `--distill_teacher`: path to the teacher model for distillation; the student model is trained to learn from both its correct targets and those "instructed" by the teacher model. The distillation hardness defining the ratio (in `[0.0, 1.0]`) of the loss defined on the teacher model targets, and the temperature used to soften the distribution of the targets are specified as parts of the distillation modifier in the recipe.

The following command prunes the model in 30 epochs to 80% sparsity of the encoder layers, saving two checkpoints during training:

```bash
sparseml.transformers.question_answering \
  --model_name_or_path bert-base-uncased \
  --distill_teacher MODELS_DIR/bert-base-12layers \
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
  --recipe recipes/bert-base-12layers_prune80.md \
  --save_strategy epoch \
  --save_total_limit 2
```

The directory `recipes` contains information about recipes and training commands used to produce our BERT pruned models on the SQuAD dataset.

### Dropping Layers

In some situations, you might drop certain layers from a BERT model and retrain on your own dataset. `SparseML` supports these use cases with a modifier called `LayerPruningModifier` that can be used as part of a pruning recipe. As an example, below is a modifier that prunes layers 5th and 7th from a BERT model:
```
!LayerPruningModifier
 layers: ['bert.encoder.layer.5', bert.encoder.layer.7']
```

The directory `recipes` contains recipes, for example `bert-base-6layers_prune80`, that drops six layers from a model before applying pruning.

The following table presents the recipes in the directory, the corresponding results, and `wandb` logging for our pruned BERT models.

| Recipe name | Description | EM / F1 | Weights and Biases |
|-------------|-------------|---------|-----------------------------------------------------------------------------------------------|
| bert-base-12layers               | BERT fine-tuned on SQuAD | 80.927 / 88.435 | [wandb](https://wandb.ai/neuralmagic/sparse-bert-squad/runs/w3b1ggyq?workspace=user-neuralmagic) |
| bert-base-12layers_prune80       | Prune baseline model fine-tuned on SQuAD at 80% sparsity of encoder units | 81.372 / 88.62 | [wandb](https://wandb.ai/neuralmagic/sparse-bert-squad/runs/18qdx7b3?workspace=user-neuralmagic) |
| bert-base-12layers_prune90       | Prune baseline model fine-tuned on SQuAD at 90% sparsity of encoder units | 79.376 / 87.229 | [wandb](https://wandb.ai/neuralmagic/sparse-bert-squad/runs/2ht2eqsn?workspace=user-neuralmagic) |
| bert-base-12layers_prune95       | Prune baseline model fine-tuned on SQuAD at 95% sparsity of encoder units | 74.939 / 83.929 | [wandb](https://wandb.ai/neuralmagic/sparse-bert-squad/runs/3gv0arxd?workspace=user-neuralmagic) |
| bert-base-6layers_prune80        | Prune 6-layer model fine-tuned on SQuAD at 80% sparsity of encoder units | 78.042 / 85.915 | [wandb](https://wandb.ai/neuralmagic/sparse-bert-squad/runs/ebab4np4?workspace=user-neuralmagic) |
| bert-base-6layers_prune90        | Prune 6-layer model fine-tuned on SQuAD at 90% sparsity of encoder units | 75.08 / 83.602 | [wandb](https://wandb.ai/neuralmagic/sparse-bert-squad/runs/3qvxoroz?workspace=user-neuralmagic) |
| bert-base-6layers_prune95        | Prune 6-layer model fine-tuned on SQuAD at 95% sparsity of encoder units | 70.946 / 80.483 | [wandb](https://wandb.ai/neuralmagic/sparse-bert-squad/runs/3plynclw?workspace=user-neuralmagic) |
| bert-base-12layers_prune_quant70 | Prune and quantize baseline model fine-tuned on SQuAD at 70% sparsity of encoder units | 80.331 / 87.537 | [wandb](https://wandb.ai/neuralmagic/huggingface/runs/33mglpyq?workspace=user-neuralmagic) |
| bert-base-12layers_prune_quant90 | Prune and quantize baseline model fine-tuned on SQuAD at 90% sparsity of encoder units | 75.383 / 83.924 | [wandb](https://wandb.ai/neuralmagic/huggingface/runs/hr48lbh1?workspace=user-neuralmagic) |
| bert-base-6layers_prune_quant70  | Prune and quantize 6-layer model fine-tuned on SQuAD at 70% sparsity of encoder units | 77.001 / 85.252 | [wandb](https://wandb.ai/neuralmagic/huggingface/runs/1ei4qarw?workspace=user-neuralmagic) |
| bert-base-6layers_prune_quant90  | Prune and quantize 6-layer model fine-tuned on SQuAD at 90% sparsity of encoder units | 71.948 / 81.060 | [wandb](https://wandb.ai/neuralmagic/huggingface/runs/tkpvq7iz?workspace=user-neuralmagic) |


## Exporting for Inference

Additionally, you may use the `sparseml.transformers.export_onnx` script to generate an ONNX model that can be used
for inference deployment and benchmarking with the `DeepSparse Engine`.

The following command evaluates a pruned model and converts it to the ONNX format:

```bash
sparseml.transformers.export_onnx \
  --task question-answering \
  --model_path MODELS_DIR/bert-base-12layers_prune80
```

If it runs successfully, you will have the converted `model.onnx` in `MODELS_DIR/bert-base-12layers_prune80/onnx`. You can now run it in ONNX-compatible inference engines such as [DeepSparse](https://github.com/neuralmagic/deepsparse). The `DeepSparse Engine` is explicitly coded to support running sparsified models for significant improvements in inference performance.
When running in `DeepSparse`, reference the entire model directory so that the ONNX file as well as tokenizer and data configs
are read.

## Wrap-Up

Neural Magic recipes simplify the sparsification process by encoding the hyperparameters and instructions needed to create highly accurate pruned BERT models. In this tutorial, you created a pre-trained model to establish a baseline, applied a Neural Magic recipe for sparsification, and exported to ONNX to run through an inference engine.

For Neural Magic Support, sign up or log in to our [**Deep Sparse Community Slack**](https://join.slack.com/t/discuss-neuralmagic/shared_invite/zt-q1a1cnvo-YBoICSIw3L1dmQpjBeDurQ). Bugs, feature requests, or additional questions can also be posted to our [GitHub Issue Queue.](https://github.com/neuralmagic/sparseml/issues)
