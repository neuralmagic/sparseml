# The Optimal BERT Surgeon: Scalable and Accurate Second-Order Pruning for Large Language Models (oBERT)

Author: @eldarkurtic

Paper: [https://arxiv.org/abs/2203.07259](https://arxiv.org/abs/2203.07259)

Demo: [https://neuralmagic.com/blog/obert/](https://neuralmagic.com/blog/obert/)

Abstract:
```
We introduce the Optimal BERT Surgeon (oBERT), an efficient and accurate pruning method based on approximate second-order information, which we show to yield state-of-the-art results for compression in both stages of language tasks: pre-training and fine-tuning. Specifically, oBERT extends existing work on second-order pruning by allowing for pruning weight blocks, and is the first such method that is applicable at BERT scale.
Second, we investigate compounding compression approaches to obtain highly compressed but accurate models for deployment on edge devices. These models significantly push boundaries of the current state-of-the-art sparse BERT models with respect to all metrics: model size, inference speed and task accuracy. For example, relative to the dense BERT-base, we obtain 10x model size compression with < 1% accuracy drop, 10x CPU-inference speedup with < 2% accuracy drop, and 29x CPU-inference speedup with < 7.5% accuracy drop.
```


## FAQs

### 1. Where is the code for oBERT pruner?
The Optimal BERT Surgeon (oBERT) is implemented and integrated in the SparseML library in the form of `OBSPruningModifier`, located at [https://github.com/neuralmagic/sparseml/blob/main/src/sparseml/pytorch/sparsification/pruning/modifier_pruning_obs.py](https://github.com/neuralmagic/sparseml/blob/main/src/sparseml/pytorch/sparsification/pruning/modifier_pruning_obs.py). Scroll below the FAQ section for scripts and recipes to reproduce results from the paper, but also to see some examples of how YAML recipes with this pruner are constructed.

### 2. Which sparsity patterns are supported?
Unstructured and 4-block (VNNI) patterns.

### 3. Can I use it to prune other models, for example RoBERTa, Yolo, ResNet, ViT, etc.?
Yes, the pruner can be used to prune any model.

### 4. How to configure it for other models?
`OBSPruningModifier` has three main hyper-parameters (`num_grads`, `fisher_block_size`, and `damp`) that need to be configured properly for new models for best results. The defaults provided in the scripts and recipes below are tuned for the BERT-base model, and are probably suboptimal for your custom model.

#### 4.1 `fisher_block_size`
This hyper-parameter determines size of blocks along the main diagonal of the block-diagonal Hessian approximation in the form of empirical Fisher information matrix. It should be set at the maximum possible value that GPU allows for the given model. Memory complexity is `O(number_of_params x fisher_block_size)`, so it is easy to estimate memory needed to do the pruning step. For example, for a pruning modifier that looks like this:

```yaml
- !OBSPruningModifier
  params:
   - model.2.cv1.conv.weight      <— e.g. 1M params
   - model.20.cv2.conv.weight     <— e.g. 1M params
   - model.20.m.0.cv1.conv.weight <— e.g. 1M params
   - model.4.m.0.cv2.conv.weight  <— e.g. 1M params
  fisher_block_size: 50
```

the total memory cost would be: `total_num_params x fisher_block_size x 4 / 1024^3 GBs = (1M + 1M + 1M + 1M) x 50 x 4 / 1024^3 GBs = 0.745 GBs`. This amount of memory has to be available on a GPU when the pruning step happens, and if more GPUs are available (e.g. via PyTorch DDP), it will be parallelized and reduced to `0.745GBs / num_GPUs`.
In the worst case scenario, when you are really memory bounded and can’t afford `fisher_block_size > 1`, then `fisher_block_size = 1` should also work well. In this case, the cost is the same as the cost of standard magnitude pruning, but the pruning step should be much better.
The main benefit of `fisher_block_size > 1` is that after pruning, the model gets a weight update based on the blockwise Fisher which compensates for the pruned weights. In `fisher_block_size=1` case, this update doesn’t happen because the Fisher is approximated with a diagonal matrix (but still better approximation than isotropic approximation that stands behind magnitude pruning).

#### 4.2 `num_grads`
The larger the better. However, at some point the returns start being diminishing. A good default value that should work across many models is `1024`. Feel free to experiment with smaller values in your first test runs to determine if it's safe to reduce this number to something like `512` or `256`.
This parameter is responsible for the major time-overhead in the pruning step, as it will run the forward pass for `num_grads` times to collect `num_grads` gradients.

#### 4.3 `damp`
This is the most problematic hyper-parameter, which if not configured properly, can make your pruning step really bad (sometimes even worse than simple magnitude pruning). We suggest tuning this param for each model independently as it can make a huge difference in results (check Ablation studies in the Appendix of the oBERT paper).
For example, if it's set to a large value (relative to the scale of gradients), the pruning step wouldn’t be any different than the simple magnitude pruning (your Fisher matrix would be isotropic). If it's set to a smaller value, matrix inversion and pruning become unstable.
Luckily, there is a very quick and cheap procedure to determine a good `damp` value (as well as `num_grads` value). It works as follows:

    1. fine-tune your model on the target task to get a dense baseline accuracy
    2. run one-shot pruning to for example 50% and 70% sparsities over a grid of values for `damp` and `num_grads` params
    3. the grid should look something like this: `num_grads = {256, 512, 1024, 2048}` and `damp = {1e-4, 1e-5, 1e-6, 1e-7, 1e-8}`
    4. evaluate all of the one-shot pruned models and pick the best `num_grads` and `damp` combination

#### 4.4 `global_sparsity`
If you don't have some desired sparsity distribution to impose over layers in your model, it is preferable to use `global_sparsity: True` with `OBSPruningModifier`. This pruner is able to come up with sparsity distributions with the best accuracy-sparsity trade-offs, and should be preferred over uniform sparsity distributions.

#### 4.5 `num_recomputations`
This feature improves `OBSPruningModifier` by allowing to recompute inverse Hessian `num_recomputations` times when doing a pruning step.
For example, in the standard pipeline we would ask `OBSPruningModifier` to prune a model to 50% and it would do so by estimating inverse Hessian once and then use it to prune weights. This recomputation feature enables doing the same pruning step but with multiple estimations of the inverse Hessian. For example, `num_recomputations=2` would estimate the inverse Hessian, prune `50%/num_recomputations = 25%` of weights, and then re-estimate it again (with newly pruned weights taken into consideration) and prune the remaining 25% to reach the pruning target of 50%.

This feature helps a lot in situations where a lot of weights are being removed in one pruning step (e.g. 50%, 70%). OBS method relies on the fact that the model is in the vicinity of the local minimum, and when removing many weights at once this assumption breaks.

To demonstrate benefits of recomputations, here are some one-shot pruning results of the fine-tuned BERT-base model on the SQuADv1.1 task:

| num\_recomputations | 70% sparsity | 80% sparsity | 90% sparsity |
| ------------------- | ------ | ------ | ------ |
| 1, default          | 83.38  | 57.48  | 14.57  |
| 4                   | 85.17  | 72.76  | 29.23  |
| 8                   | 85.34  | 77.85  | 39.58  |
| 16                  | 85.49  | 78.27  | 45.03  |
| 32                  | 85.49  | 78.65  | 47.67  |

`1, default` is the standard approach without any additional re-estimations.


## The Optimal BERT Surgeon - reproducibility
To ease reproducibility, in the following Tables (which correspond to the Tables reported in the paper), we provide links for open-sourced checkpoints, recipes and scripts used to produce them.
### Table 1: Dev-set performance of downstream-pruned BERT-base models
In the following Table (Table 1. from the paper) we provide links for the best performing unstructured-pruned oBERT models, along with recipes and scripts to reproduce them from scratch.

| Task | BERT<br>base | Sparsity | oBERT<br>10 epochs | oBERT<br>30 epochs |
| :---: | :-------: | :----: | :----------------: | :----------------: |
| SQuAD<br>F1 | 88.54<br>[model](https://huggingface.co/neuralmagic/oBERT-teacher-squadv1) | 80%<br>90%<br>97% | -<br>87.98<br>84.65 | 89.04 [model](https://huggingface.co/neuralmagic/oBERT-12-downstream-pruned-unstructured-80-squadv1), [recipe](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT/recipes/30epochs_unstructured80_squad.yaml), [script](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT/scripts/30epochs_gradual_pruning_squad.sh)<br>88.31 [model](https://huggingface.co/neuralmagic/oBERT-12-downstream-pruned-unstructured-90-squadv1), [recipe](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT/recipes/30epochs_unstructured90_squad.yaml), [script](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT/scripts/30epochs_gradual_pruning_squad.sh)<br>85.98 [model](https://huggingface.co/neuralmagic/oBERT-12-downstream-pruned-unstructured-97-squadv1), [recipe](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT/recipes/30epochs_unstructured97_squad.yaml), [script](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT/scripts/30epochs_gradual_pruning_squad.sh) |
| MNLI<br>m-acc | 84.54<br>[model](https://huggingface.co/neuralmagic/oBERT-teacher-mnli) | 80%<br>90%<br>97% | -<br>83.20<br>81.00 | 84.32 [model](https://huggingface.co/neuralmagic/oBERT-12-downstream-pruned-unstructured-80-mnli), [recipe](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT/recipes/30epochs_unstructured80_mnli.yaml), [script](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT/scripts/30epochs_gradual_pruning_mnli_qqp.sh)<br>83.79 [model](https://huggingface.co/neuralmagic/oBERT-12-downstream-pruned-unstructured-90-mnli), [recipe](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT/recipes/30epochs_unstructured90_mnli.yaml), [script](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT/scripts/30epochs_gradual_pruning_mnli_qqp.sh)<br>81.77 [model](https://huggingface.co/neuralmagic/oBERT-12-downstream-pruned-unstructured-97-mnli), [recipe](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT/recipes/30epochs_unstructured97_mnli.yaml), [script](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT/scripts/30epochs_gradual_pruning_mnli_qqp.sh) |
| QQP<br>acc | 91.06<br>[model](https://huggingface.co/neuralmagic/oBERT-teacher-qqp) | 80%<br>90%<br>97% | -<br>90.89<br>90.23 | 91.57 [model](https://huggingface.co/neuralmagic/oBERT-12-downstream-pruned-unstructured-80-qqp), [recipe](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT/recipes/30epochs_unstructured80_qqp.yaml), [script](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT/scripts/30epochs_gradual_pruning_mnli_qqp.sh)<br>91.35 [model](https://huggingface.co/neuralmagic/oBERT-12-downstream-pruned-unstructured-90-qqp), [recipe](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT/recipes/30epochs_unstructured90_qqp.yaml), [script](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT/scripts/30epochs_gradual_pruning_mnli_qqp.sh)<br>90.87 [model](https://huggingface.co/neuralmagic/oBERT-12-downstream-pruned-unstructured-97-qqp), [recipe](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT/recipes/30epochs_unstructured97_qqp.yaml), [script](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT/scripts/30epochs_gradual_pruning_mnli_qqp.sh) |


### Table 2: Sparse-transfer dev-set performance of upstream-pruned BERT-base models
In the following Table (Table 2. from the paper) we provide links for the best performing unstructured-pruned oBERT models, along with recipes and scripts to reproduce them from scratch.

**Note (models v2)**: these results will be presented in the upcoming updated version of the paper.

Upstream pruned oBERT models:
- 90% unstructured [model](https://huggingface.co/neuralmagic/oBERT-12-upstream-pruned-unstructured-90-v2), [recipe](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT/recipes/3epochs_unstructured90_mlm.yaml), [script](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT/scripts/3epochs_gradual_pruning_mlm.sh)
- 97% unstructured [model](https://huggingface.co/neuralmagic/oBERT-12-upstream-pruned-unstructured-97-v2), [recipe](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT/recipes/3epochs_unstructured97_mlm.yaml), [script](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT/scripts/3epochs_gradual_pruning_mlm.sh)

when fine-tuned on a downstream task with fixed masks (i.e. sparse-transfer):

| Task | BERT<br>base | Sparsity | oBERT |
| :---: | :-------: | :----: | :------: |
| SQuAD<br>F1 | 88.54<br>[model](https://huggingface.co/neuralmagic/oBERT-teacher-squadv1) | 90%<br>97% | 88.49 [model](https://huggingface.co/neuralmagic/oBERT-12-upstream-pruned-unstructured-90-finetuned-squadv1-v2), [recipe](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT/recipes/8epochs_sparse_transfer_squad.yaml), [script](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT/scripts/8epochs_sparse_transfer_squad.sh)<br>84.92 [model](https://huggingface.co/neuralmagic/oBERT-12-upstream-pruned-unstructured-97-finetuned-squadv1-v2), [recipe](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT/recipes/8epochs_sparse_transfer_squad.yaml), [script](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT/scripts/8epochs_sparse_transfer_squad.sh)|
| MNLI<br>m-acc | 84.54<br>[model](https://huggingface.co/neuralmagic/oBERT-teacher-mnli) | 90%<br>97% | 83.40 [model](https://huggingface.co/neuralmagic/oBERT-12-upstream-pruned-unstructured-90-finetuned-mnli-v2), [recipe](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT/recipes/8epochs_sparse_transfer_mnli.yaml), [script](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT/scripts/8epochs_sparse_transfer_mnli_qqp.sh)<br>80.91 [model](https://huggingface.co/neuralmagic/oBERT-12-upstream-pruned-unstructured-97-finetuned-mnli-v2), [recipe](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT/recipes/8epochs_sparse_transfer_mnli.yaml), [script](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT/scripts/8epochs_sparse_transfer_mnli_qqp.sh)|
| QQP<br>acc | 91.06<br>[model](https://huggingface.co/neuralmagic/oBERT-teacher-qqp) | 90%<br>97% | 90.99 [model](https://huggingface.co/neuralmagic/oBERT-12-upstream-pruned-unstructured-90-finetuned-qqp-v2), [recipe](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT/recipes/8epochs_sparse_transfer_qqp.yaml), [script](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT/scripts/8epochs_sparse_transfer_mnli_qqp.sh)<br>90.33 [model](https://huggingface.co/neuralmagic/oBERT-12-upstream-pruned-unstructured-97-finetuned-qqp-v2), [recipe](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT/recipes/8epochs_sparse_transfer_qqp.yaml), [script](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT/scripts/8epochs_sparse_transfer_mnli_qqp.sh) |

**Note**: the results below are currently presented in the paper and will be removed when the updated version of the paper is released with previously presented **v2** models.

Upstream pruned oBERT models:
- 90% unstructured [model](https://huggingface.co/neuralmagic/oBERT-12-upstream-pruned-unstructured-90)
- 97% unstructured [model](https://huggingface.co/neuralmagic/oBERT-12-upstream-pruned-unstructured-97)

when fine-tuned on a downstream task with fixed masks (i.e. sparse-transfer):

| Task | BERT<br>base | Sparsity | oBERT |
| :---: | :-------: | :----: | :------: |
| SQuAD<br>F1 | 88.54<br>[model](https://huggingface.co/neuralmagic/oBERT-teacher-squadv1) | 90%<br>97% | 88.42 [model](https://huggingface.co/neuralmagic/oBERT-12-upstream-pruned-unstructured-90-finetuned-squadv1)<br>84.39 [model](https://huggingface.co/neuralmagic/oBERT-12-upstream-pruned-unstructured-97-finetuned-squadv1) |
| MNLI<br>m-acc | 84.54<br>[model](https://huggingface.co/neuralmagic/oBERT-teacher-mnli) | 90%<br>97% | 82.29 [model](https://huggingface.co/neuralmagic/oBERT-12-upstream-pruned-unstructured-90-finetuned-mnli)<br>78.85 [model](https://huggingface.co/neuralmagic/oBERT-12-upstream-pruned-unstructured-97-finetuned-mnli) |
| QQP<br>acc | 91.06<br>[model](https://huggingface.co/neuralmagic/oBERT-teacher-qqp) | 90%<br>97% | 90.83 [model](https://huggingface.co/neuralmagic/oBERT-12-upstream-pruned-unstructured-90-finetuned-qqp)<br>89.76 [model](https://huggingface.co/neuralmagic/oBERT-12-upstream-pruned-unstructured-97-finetuned-qqp) |



### Table 3: F1 score of the 3, 6, and 12-layer models compound-compressed with oBERT on SQuAD
In the following Table (Table 3. from the paper) we provide links for the best performing oBERT models, along with recipes and scripts to reproduce them from scratch.

For the 12-layer model, we use the standard HuggingFace's `bert-base-uncased` [model](https://huggingface.co/bert-base-uncased) for a fair comparison with other compression approaches. For the 3 and 6 layer models, we drop layers from our upstream-pretrained 12-layer [model](https://huggingface.co/neuralmagic/oBERT-12-upstream-pretrained-dense), and pretrain them to obtain the following 3 and 6 layer dense models:
- 6-layer dense pretrained [model](https://huggingface.co/neuralmagic/oBERT-6-upstream-pretrained-dense)
- 3-layer dense pretrained [model](https://huggingface.co/neuralmagic/oBERT-3-upstream-pretrained-dense)

| Layers | Sparsity | Unstructured | 4-block | 4-block+QAT |
| :---:  | :---:    | :---:        | :---:   | :---:       |
| 12 | 0%<br>80%<br>90% | 89.48 [model](https://huggingface.co/neuralmagic/oBERT-12-downstream-dense-squadv1), [recipe](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT/recipes/30epochs_dense_squad.yaml), [script](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT/scripts/30epochs_gradual_pruning_squad.sh)<br>89.04 [model](https://huggingface.co/neuralmagic/oBERT-12-downstream-pruned-unstructured-80-squadv1), [recipe](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT/recipes/30epochs_unstructured80_squad.yaml), [script](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT/scripts/30epochs_gradual_pruning_squad.sh)<br>88.31 [model](https://huggingface.co/neuralmagic/oBERT-12-downstream-pruned-unstructured-90-squadv1), [recipe](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT/recipes/30epochs_unstructured90_squad.yaml), [script](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT/scripts/30epochs_gradual_pruning_squad.sh) | 89.48 [model](https://huggingface.co/neuralmagic/oBERT-12-downstream-dense-squadv1), [recipe](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT/recipes/30epochs_dense_squad.yaml), [script](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT/scripts/30epochs_gradual_pruning_squad.sh)<br>88.57 [model](https://huggingface.co/neuralmagic/oBERT-12-downstream-pruned-block4-80-squadv1), [recipe](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT/recipes/30epochs_4block80_squad.yaml), [script](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT/scripts/30epochs_gradual_pruning_squad.sh)<br>87.57 [model](https://huggingface.co/neuralmagic/oBERT-12-downstream-pruned-block4-90-squadv1), [recipe](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT/recipes/30epochs_4block90_squad.yaml), [script](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT/scripts/30epochs_gradual_pruning_squad.sh) | 89.06 [model](https://huggingface.co/neuralmagic/oBERT-12-downstream-dense-QAT-squadv1)<br>87.89 [model](https://huggingface.co/neuralmagic/oBERT-12-downstream-pruned-block4-80-QAT-squadv1)<br>86.68 [model](https://huggingface.co/neuralmagic/oBERT-12-downstream-pruned-block4-90-QAT-squadv1) |
| 6 | 0%<br>80%<br>90% | 88.32 [model](https://huggingface.co/neuralmagic/oBERT-6-downstream-dense-squadv1), [recipe](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT/recipes/30epochs_dense_squad.yaml), [script](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT/scripts/30epochs_gradual_pruning_squad.sh)<br>88.20 [model](https://huggingface.co/neuralmagic/oBERT-6-downstream-pruned-unstructured-80-squadv1), [recipe](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT/recipes/30epochs_init30_unstructured80_squad.yaml), [script](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT/scripts/30epochs_gradual_pruning_squad.sh)<br>86.78 [model](https://huggingface.co/neuralmagic/oBERT-6-downstream-pruned-unstructured-90-squadv1), [recipe](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT/recipes/30epochs_init30_unstructured90_squad.yaml), [script](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT/scripts/30epochs_gradual_pruning_squad.sh) | 88.32 [model](https://huggingface.co/neuralmagic/oBERT-6-downstream-dense-squadv1), [recipe](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT/recipes/30epochs_dense_squad.yaml), [script](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT/scripts/30epochs_gradual_pruning_squad.sh)<br>87.00 [model](https://huggingface.co/neuralmagic/oBERT-6-downstream-pruned-block4-80-squadv1), [recipe](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT/recipes/30epochs_init30_4block80_squad.yaml), [script](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT/scripts/30epochs_gradual_pruning_squad.sh)<br>85.34 [model](https://huggingface.co/neuralmagic/oBERT-6-downstream-pruned-block4-90-squadv1), [recipe](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT/recipes/30epochs_init30_4block90_squad.yaml), [script](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT/scripts/30epochs_gradual_pruning_squad.sh) | 87.94 [model](https://huggingface.co/neuralmagic/oBERT-6-downstream-dense-QAT-squadv1)<br>86.10 [model](https://huggingface.co/neuralmagic/oBERT-6-downstream-pruned-block4-80-QAT-squadv1)<br>84.59 [model](https://huggingface.co/neuralmagic/oBERT-6-downstream-pruned-block4-90-QAT-squadv1) |
| 3 | 0%<br>80%<br>90% | 84.66 [model](https://huggingface.co/neuralmagic/oBERT-3-downstream-dense-squadv1), [recipe](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT/recipes/30epochs_dense_squad.yaml), [script](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT/scripts/30epochs_gradual_pruning_squad.sh)<br>84.08 [model](https://huggingface.co/neuralmagic/oBERT-3-downstream-pruned-unstructured-80-squadv1), [recipe](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT/recipes/30epochs_init30_unstructured80_squad.yaml), [script](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT/scripts/30epochs_gradual_pruning_squad.sh)<br>82.50 [model](https://huggingface.co/neuralmagic/oBERT-3-downstream-pruned-unstructured-90-squadv1), [recipe](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT/recipes/30epochs_init30_unstructured90_squad.yaml), [script](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT/scripts/30epochs_gradual_pruning_squad.sh) | 84.66 [model](https://huggingface.co/neuralmagic/oBERT-3-downstream-dense-squadv1), [recipe](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT/recipes/30epochs_dense_squad.yaml), [script](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT/scripts/30epochs_gradual_pruning_squad.sh)<br>82.79 [model](https://huggingface.co/neuralmagic/oBERT-3-downstream-pruned-block4-80-squadv1), [recipe](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT/recipes/30epochs_init30_4block80_squad.yaml), [script](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT/scripts/30epochs_gradual_pruning_squad.sh) <br>80.69 [model](https://huggingface.co/neuralmagic/oBERT-3-downstream-pruned-block4-90-squadv1), [recipe](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT/recipes/30epochs_init30_4block90_squad.yaml), [script](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT/scripts/30epochs_gradual_pruning_squad.sh) | 84.25 [model](https://huggingface.co/neuralmagic/oBERT-3-downstream-dense-QAT-squadv1)<br>82.04 [model](https://huggingface.co/neuralmagic/oBERT-3-downstream-pruned-block4-80-QAT-squadv1)<br>79.66 [mode](https://huggingface.co/neuralmagic/oBERT-3-downstream-pruned-block4-90-QAT-squadv1) |


### Important notes
- the `OBSPruningModifier` will make use of all available GPUs during the pruning step to split the workload; use `CUDA_VISIBLE_DEVICES` to specify which GPUs can/should be used
- all experiments are designed to fit on a single 24GB RTX 3090 card, except the upstream ones which need to use more GPUs due to the large pre-training batch-size
- if an experiment doesn't fit on a single GPU, the multi-GPU mode via PyTorch DistributedDataParallel (DDP) should be used; the `OBSPruningModifier` will make use of all the available GPUs to split the workload
- results reported in the paper are obtained with the following versions of libraries:
    - `sparseml=0.2.0`
    - `transformers=4.5.1`
    - `datasets=1.6.1`
    - `torch=1.8.1`
- since then, we have improved and optimized `OBSPruningModifier` implementation, and to ease reproducibility, we have successfully reproduced results with newer versions of libraries:
    - `sparseml=0.12.0`
    - `transformers=4.18.0.dev0`
    - `datasets=2.0.0`
    - `torch=1.11.0`

# BibTeX entry and citation info
```bibtex
@article{kurtic2022optimal,
  title={The Optimal BERT Surgeon: Scalable and Accurate Second-Order Pruning for Large Language Models},
  author={Kurtic, Eldar and Campos, Daniel and Nguyen, Tuan and Frantar, Elias and Kurtz, Mark and Fineran, Benjamin and Goin, Michael and Alistarh, Dan},
  journal={arXiv preprint arXiv:2203.07259},
  year={2022}
}
```