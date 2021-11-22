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

# Sparse Transfer Learning With BERT

This tutorial shows how Neural Magic sparse models simplify the sparsification process by offering pre-sparsified BERT models for transfer learning onto other datasets.

## Overview

Neural Magic’s ML team creates sparsified models that allow anyone to plug in their data and leverage pre-sparsified models from the SparseZoo on top of Hugging Faces’ robust training pipelines.
Sparsifying involves removing redundant information from neural networks using algorithms such as pruning and quantization, among others.
This sparsification process results in many benefits for deployment environments, including faster inference and smaller file sizes.
Unfortunately, many have not realized the benefits due to the complicated process and number of hyperparameters involved.
Working through this tutorial, you will experience how Neural Magic recipes simplify the sparsification process by:

- Selecting a pre-sparsified model.
- Selecting a dataset use case.
- Creating a dense teacher model for distillation.
- Applying a sparse transfer learning recipe.
- Exporting an inference graph to reduce its deployment size and run it in an inference engine such as DeepSparse.

Before diving in, be sure to go through setup as listed out in the [README](https://github.com/neuralmagic/sparseml/blob/main/integrations/huggingface-transformers/README.md) for this integration.
Additionally, all commands are intended to be run from the root of the `transformers` repository folder (`cd integrations/huggingface-transformers/transformers`).

## Need Help?

For Neural Magic Support, sign up or log in to get help with your questions in our Tutorials channel: [Discourse Forum](https://discuss.neuralmagic.com/) and/or [Slack](https://join.slack.com/t/discuss-neuralmagic/shared_invite/zt-q1a1cnvo-YBoICSIw3L1dmQpjBeDurQ).

## Selecting a Pre-sparsified Model

For this tutorial, we will use a 12 layer BERT model sparsified to 80% on the Wikitext and BookCorpus dataset. 
This model gives a good tradeoff between inference performance and accuracy.
This 12 layer model gives 3.9x better throughput while recovering close to the dense baseline for most transfer tasks.

The SparseZoo stub for this model is `zoo:nlp/masked_language_modeling/bert-base/pytorch/huggingface/bookcorpus_wikitext/12layer_pruned80-none` and will be used to select the model in the training commands to be used later. 
Additional BERT models, including ones with higher sparsity and fewer layers, are found on the [SparseZoo](https://sparsezoo.neuralmagic.com/?page=1&domain=nlp&sub_domain=masked_language_modeling) and can be subbed in place of the 12 layer 80% sparse model for better performance or recovery.

## Selecting a Dataset Use Case

Results and examples are given for 5 common use cases with popular datasets in this tutorial. 
If you would like to apply these approaches to your own dataset, Hugging Face has additional information for setup of custom datasets [here](https://huggingface.co/transformers/custom_datasets.html).
Once you have successfully converted your dataset into Hugging Face's format, it can be safely plugged into these flows and used for sparse transfer learning from the pre-sparsified models.

The use cases and datasets covered in this tutorial are listed below along with results for sparse transfer learning with the 80% pruned BERT model.

| Use Case                   | Dataset                                                                       | Description                                                                                                                                                                                                          | Sparse Transfer Results  |
|----------------------------|-------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------|
| Question and Answering     | [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)                          | A reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text from the corresponding reading passage.       | 87.1 F1 (87.5 baseline)  |
| Binary Classification      | [QQP](https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs) | A dataset made up of potential question pairs from Quora with a boolean label representing whether the questions are duplicates or not.                                                                              | 90 acc (91.3 baseline)   |
| Multi-Class Classification | [MNLI](https://cims.nyu.edu/~sbowman/multinli/)                               | A crowd-sourced collection of sentence pairs annotated with textual entailment information. It covers a range of genres of spoken and written text and supports a distinctive cross-genre generalization evaluation. | 81.9 acc (82.3 baseline) |
| Named Entity Recognition   | [CoNNL2003](https://www.clips.uantwerpen.be/conll2003/ner/)                   | A dataset concentrated on four types of named entities: persons, locations, organizations and names of miscellaneous entities that do not belong to the previous three groups.                                       | 88.5 acc (94.7 baseline) |
| Sentiment Analysis         | [SST2](https://nlp.stanford.edu/sentiment/)                                   | A corpus that includes fine-grained sentiment labels for phrases within sentences and presented new challenges for sentiment compositionality.                                                                       | 91.3 acc (91.2 baseline) |

Once you have selected a use case and dataset, you are ready to create a teacher model.

## Creating a Dense Teacher

You are ready to transfer learn the model.

## Transfer Learning the Model

You are ready to export for inference.

## Exporting for Inference

This step loads a checkpoint file of the best weights measured on the validation set, and converts it into the more common inference formats.
Then, you can run the file through a compression algorithm to reduce its deployment size and run it in an inference engine such as DeepSparse.

The `best.pt` file, located in the previous step, contains a checkpoint of the best weights measured on the validation set.
These weights can be loaded into the `train.py` and `test.py` scripts now.
However, other formats are generally more friendly for other inference deployment platforms, such as [ONNX](https://onnx.ai/).

The [export.py script](https://github.com/neuralmagic/yolov5/blob/master/models/export.py) handles the logic behind loading the checkpoint and converting it into the more common inference formats, as described here.

1. Enter the following command to load the PyTorch graph, convert to ONNX, and correct any misformatted pieces of the graph for the pruned and quantized models.
   ```bash
   python models/export.py --weights PATH_TO_SPARSIFIED_WEIGHTS  --dynamic
   ```
   The result is a new file added next to the sparsified checkpoint with a `.onnx` extension:
   ```
   |-- exp
   |   |-- weights
   |   |   |-- best.pt
   |   |   |-- best.onnx
   |   |   `-- last.pt
   `-- ...
   ```
2. Now you can run the `.onnx` file through a compression algorithm to reduce its deployment size and run it in ONNX-compatible inference engines such as [DeepSparse](https://github.com/neuralmagic/deepsparse).
   The DeepSparse Engine is explicitly coded to support running sparsified models for significant improvements in inference performance.
   An example for benchmarking and deploying YOLOv5 models with DeepSparse can be found [here](https://github.com/neuralmagic/deepsparse/tree/main/examples/ultralytics-yolo).

## Wrap-Up

Neural Magic sparse models and recipes simplify the sparsification process by enabling sparse transfer learning to create highly accurate pruned BERT models.
In this tutorial, you selected a pre-sparsified model, applied a Neural Magic recipe for sparse transfer learning, and exported to ONNX to run through an inference engine.

Now, refer [here](https://github.com/neuralmagic/deepsparse/tree/main/examples/huggingface-transformers) for an example for benchmarking and deploying BERT models with DeepSparse.

For Neural Magic Support, sign up or log in to get help with your questions in our Tutorials channel: [Discourse Forum](https://discuss.neuralmagic.com/) and/or [Slack](https://join.slack.com/t/discuss-neuralmagic/shared_invite/zt-q1a1cnvo-YBoICSIw3L1dmQpjBeDurQ).
