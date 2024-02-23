# Evaluation Integrations

Some evaluation integrations require additional packages to be installed. The following instructions detail how to install the necessary packages for each integration.

## lm-eval

`lm-eval` is a package that provides a harness for evaluating language models. It is used by the `lm-evaluation-harness` integration.

This package has installs transformers and torch versions that are incompatible with the versions required by SparseML. To use this integration kindly follow the following steps to install the correct versions of the packages.

### Installation Instructions

Step 1: Make a new virtual environment + activate it

```bash
python3.11 -m venv sparseml-venv
source sparseml-venv/bin/activate
```

Step 2: Install `lm-eval` package

```bash
pip install lm-eval==0.4.1
```

Step 3: Uninstall transformers package, we will install the correct version later

```bash
pip uninstall transformers
```

Step 4: Install `sparseml` package with `[llm,torch]` extras
This will install the correct version of `transformers` package

```bash
pip install "sparseml[llm, torch]"
```

Step 5: Verify correct versions

```bash
pip show lm-eval nm-transformers
```

Note: If editable mode is used to install `sparseml` package, then the `nm-transformers` package will not be installed. In that case, the `nm-transformers-nightly` package is installed. Verify the correct version using the following command in that case.

```bash
pip show lm-eval nm-transformers-nightly
```

Output when installing in non-editable mode:

```bash
$ pip show lm-eval nm-transformers
Name: lm_eval
Version: 0.4.1
Summary: A framework for evaluating language models
Home-page: 
Author: 
Author-email: EleutherAI <contact@eleuther.ai>
License: MIT
Location: /home/ubuntu/venvs/sparseml-venv/lib/python3.11/site-packages
Requires: accelerate, datasets, evaluate, evaluate, jsonlines, numexpr, peft, pybind11, pytablewriter, rouge-score, sacrebleu, scikit-learn, sqlitedict, torch, tqdm-multiprocess, transformers, zstandard
Required-by: 
---
Name: nm-transformers
Version: 1.7.0
Summary: State-of-the-art Machine Learning for JAX, PyTorch and TensorFlow
Home-page: https://github.com/neuralmagic/transformers
Author: The Hugging Face team (past and future) with the help of all our contributors (https://github.com/huggingface/transformers/graphs/contributors)
Author-email: transformers@huggingface.co
License: Apache 2.0 License
Location: /home/ubuntu/venvs/sparseml-venv/lib/python3.11/site-packages
Requires: filelock, huggingface-hub, numpy, packaging, pyyaml, regex, requests, safetensors, tokenizers, tqdm
Required-by: 
```

Output when installing in editable mode:

```bash
$ pip show lm-eval nm-transformers-nightly
Name: lm_eval
Version: 0.4.1
Summary: A framework for evaluating language models
Home-page: 
Author: 
Author-email: EleutherAI <contact@eleuther.ai>
License: MIT
Location: /home/ubuntu/venvs/sparseml-venv/lib/python3.11/site-packages
Requires: accelerate, datasets, evaluate, evaluate, jsonlines, numexpr, peft, pybind11, pytablewriter, rouge-score, sacrebleu, scikit-learn, sqlitedict, torch, tqdm-multiprocess, transformers, zstandard
Required-by: 
---
Name: nm-transformers-nightly
Version: 1.7.0.20240131
Summary: State-of-the-art Machine Learning for JAX, PyTorch and TensorFlow
Home-page: https://github.com/neuralmagic/transformers
Author: The Hugging Face team (past and future) with the help of all our contributors (https://github.com/huggingface/transformers/graphs/contributors)
Author-email: transformers@huggingface.co
License: Apache 2.0 License
Location: /home/ubuntu/venvs/sparseml-venv/lib/python3.11/site-packages
Requires: filelock, huggingface-hub, numpy, packaging, pyyaml, regex, requests, safetensors, tokenizers, tqdm
Required-by: 
```

#### Test Command(s)

We will evaluate the model `mgoin/llama2.c-stories15M-quant-pt` on the `hellaswag` dataset using the `lm-evaluation-harness` integration and limit the number of samples to 1 for a quicker test. The following command can be used to do so:

##### Using the CLI

```bash
sparseml.evaluate \
    "mgoin/llama2.c-stories15M-quant-pt" \
    --dataset hellaswag \
    --integration lm-evaluation-harness \
    --limit 1 # for a quicker test
```

Example Output (Truncated):

```bash
sparseml.evaluate \                                                                                     (update-lm-eval-to-0.4.1|✚1…4⚑5)
    "mgoin/llama2.c-stories15M-quant-pt" \
    --dataset hellaswag \
    --integration lm-evaluation-harness \
    --limit 1 # for a quicker test
2024-02-23 12:05:55 sparseml.evaluation.cli INFO     Datasets to evaluate on: hellaswag
Batch size: 1
Additional integration arguments supplied: {}
2024-02-23 12:05:57 sparseml.evaluation.registry INFO     Auto collected lm-evaluation-harness integration for eval
2024-02-23:12:05:57,189 INFO     [registry.py:100] Auto collected lm-evaluation-harness integration for eval
2024-02-23:12:05:57,974 WARNING  [logging.py:61] Detected kernel version 5.4.0, which is below the recommended minimum of 5.5.0; this can cause the process to hang. It is recommended to upgrade the kernel to the minimum version or higher.
.
.
.
2024-02-23 12:05:58 sparseml.pytorch.model_load.helpers INFO     Applied an unstaged recipe to the model at mgoin/llama2.c-stories15M-quant-pt
.
.
2024-02-23:12:06:03,787 INFO     [lm_evaluation_harness.py:84] Selected Tasks: ['hellaswag']
2024-02-23:12:06:07,687 INFO     [task.py:363] Building contexts for task on rank 0...
2024-02-23:12:06:07,688 INFO     [evaluator.py:324] Running loglikelihood requests
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00,  5.28it/s]
2024-02-23 12:06:08 sparseml.evaluation.cli INFO     Evaluation done. Results:
formatted=[Evaluation(task='lm-evaluation-harness', dataset=Dataset(type=None, name='hellaswag', config={'model': 'mgoin/llama2.c-stories15M-quant-pt', 'model_args': None, 'batch_size': 1, 'batch_sizes': [], 'device': None, 
.
.
.
2024-02-23 12:06:08 sparseml.evaluation.cli INFO     Saving the evaluation results to /home/ubuntu/projects/sparseml/results.json
2024-02-23:12:06:08,494 INFO     [cli.py:172] Saving the evaluation results to /home/ubuntu/projects/sparseml/results.json
```

##### Using python API

```python
from sparseml import evaluate

results = evaluate(
    "mgoin/llama2.c-stories15M-quant-pt",
    datasets="hellaswag",
    integration="lm-evaluation-harness",
    limit=1, # for a quicker test
)
print(results)
```

Output (Truncated):

```bash
2024-02-23 12:12:43 sparseml.evaluation.registry INFO     Auto collected lm-evaluation-harness integration for eval
2024-02-23:12:12:43,968 INFO     [registry.py:100] Auto collected lm-evaluation-harness integration for eval
2024-02-23:12:12:44,782 WARNING  [logging.py:61] Detected kernel version 5.4.0, which is below the recommended minimum of 5.5.0; this can cause the process to hang. It is recommended to upgrade the kernel to the minimum version or higher.
2024-02-23:12:12:44,783 INFO     [huggingface.py:148] Using device 'cuda'
2024-02-23 12:12:45 sparseml.transformers.utils.helpers INFO     model_path is a huggingface model id. Attempting to download recipe from https://huggingface.co/
2024-02-23:12:12:45,398 INFO     [helpers.py:357] model_path is a huggingface model id. Attempting to download recipe from https://huggingface.co/
2024-02-23 12:12:45 sparseml.transformers.utils.helpers INFO     Found recipe: recipe.yaml for model id: mgoin/llama2.c-stories15M-quant-pt. Downloading...
2024-02-23:12:12:45,398 INFO     [helpers.py:363] Found recipe: recipe.yaml for model id: mgoin/llama2.c-stories15M-quant-pt. Downloading...
Logging all SparseML modifier-level logs to sparse_logs/23-02-2024_12.12.45.log
2024-02-23 12:12:45 sparseml.core.logger.logger INFO     Logging all SparseML modifier-level logs to sparse_logs/23-02-2024_12.12.45.log
2024-02-23:12:12:45,450 INFO     [logger.py:400] Logging all SparseML modifier-level logs to sparse_logs/23-02-2024_12.12.45.log
2024-02-23 12:12:45 sparseml.core.recipe.recipe INFO     Loading recipe from file /home/ubuntu/.cache/huggingface/hub/models--mgoin--llama2.c-stories15M-quant-pt/snapshots/863d525c4424fb315d1baaabf895320e9fcddae3/recipe.yaml
2024-02-23:12:12:45,451 INFO     [recipe.py:90] Loading recipe from file /home/ubuntu/.cache/huggingface/hub/models--mgoin--llama2.c-stories15M-quant-pt/snapshots/863d525c4424fb315d1baaabf895320e9fcddae3/recipe.yaml
manager stage: Model structure initialized
2024-02-23 12:12:45 sparseml.pytorch.model_load.helpers INFO     Applied an unstaged recipe to the model at mgoin/llama2.c-stories15M-quant-pt
2024-02-23:12:12:45,655 INFO     [helpers.py:120] Applied an unstaged recipe to the model at mgoin/llama2.c-stories15M-quant-pt
2024-02-23 12:12:45 sparseml.pytorch.model_load.helpers WARNING  Model state was not reloaded for SparseML: could not find model weights for mgoin/llama2.c-stories15M-quant-pt
2024-02-23:12:12:45,656 WARNING  [helpers.py:149] Model state was not reloaded for SparseML: could not find model weights for mgoin/llama2.c-stories15M-quant-pt
2024-02-23:12:12:46,777 WARNING  [lm_evaluation_harness.py:74] WARNING: --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT.
2024-02-23:12:12:48,789 WARNING  [__init__.py:194] Some tasks could not be loaded due to missing dependencies. Run with `--verbosity DEBUG` for full details.
2024-02-23:12:12:50,643 WARNING  [__init__.py:194] Some tasks could not be loaded due to missing dependencies. Run with `--verbosity DEBUG` for full details.
2024-02-23:12:12:50,644 INFO     [lm_evaluation_harness.py:84] Selected Tasks: ['hellaswag']
2024-02-23:12:12:54,388 INFO     [task.py:363] Building contexts for task on rank 0...
2024-02-23:12:12:54,389 INFO     [evaluator.py:324] Running loglikelihood requests
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00,  5.43it/s]
formatted=[Evaluation(task='lm-evaluation-harness', dataset=Dataset(type=None, name='hellaswag', config={'model': 'mgoin/llama2.c-stories15M-quant-pt', 'model_args': None, 'batch_size': 1, 'batch_sizes': [], 'device': None, 'use_cache': None, 'limit': 1, 'bootstrap_iters': 100000, 'gen_kwargs': None}, split=None), metrics=[Metric(name='acc,none', value=0.0), Metric(name='acc_norm,none', value=1.0)], samples=None)] raw={'results': {'hellaswag': {'acc,none': 0.0, 'acc_stderr,none': 'N/A', 'acc_norm,none': 1.0, 'acc_norm_stderr,none': 'N/A', 'alias': 'hellaswag'}}, 'configs': {'hellaswag': {'task': 'hellaswag', 'group': ['multiple_choice'], 'dataset_path': 'hellaswag', 'training_split': 'train', 'validation_split': 'validation', 'process_docs': 'def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:\n    def _process_doc(doc):\n        ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()\n        out_doc = {\n            "query": preprocess(doc["activity_label"] + ": " + ctx),\n            "choices": [preprocess(ending) for ending in doc["endings"]],\n            "gold": int(doc["label"]),\n        }\n        return out_doc\n\n    return dataset.map(_process_doc)\n', 'doc_to_text': '{{query}}', 'doc_to_target': '{{label}}', 'doc_to_choice': 'choices', 'description': '', 'target_delimiter': ' ', 'fewshot_delimiter': '\n\n', 'metric_list': [{'metric': 'acc', 'aggregation': 'mean', 'higher_is_better': True}, {'metric': 'acc_norm', 'aggregation': 'mean', 'higher_is_better': True}], 'output_type': 'multiple_choice', 'repeats': 1, 'should_decontaminate': False, 'metadata': {'version': 1.0}}}, 'versions': {'hellaswag': 1.0}, 'n-shot': {'hellaswag': 0}, 'samples': {'hellaswag': [{'doc_id': 0, 'doc': {'ind': 24, 'activity_label': 'Roof shingle removal', 'ctx_a': 'A man is sitting on a roof.', 'ctx_b': 'he', 'ctx': 'A man is sitting on a roof. he', 'endings': ['is using wrap to wrap a pair of skis.', 'is ripping level tiles off.', "is holding a rubik's cube.", 'starts pulling up roofing on a roof.'], 'source_id': 'activitynet~v_-JhWjGDPHMY', 'split': 'val', 'split_type': 'indomain', 'label': '3', 'query': 'Roof shingle removal: A man is sitting on a roof. He', 'choices': ['is using wrap to wrap a pair of skis.', 'is ripping level tiles off.', "is holding a rubik's cube.", 'starts pulling up roofing on a roof.'], 'gold': 3}, 'target': 3, 'arguments': [('Roof shingle removal: A man is sitting on a roof. He', ' is using wrap to wrap a pair of skis.'), ('Roof shingle removal: A man is sitting on a roof. He', ' is ripping level tiles off.'), ('Roof shingle removal: A man is sitting on a roof. He', " is holding a rubik's cube."), ('Roof shingle removal: A man is sitting on a roof. He', ' starts pulling up roofing on a roof.')], 'resps': [[(-114.10840606689453, False)], [(-82.98793029785156, False)], [(-93.36141967773438, False)], [(-93.36141967773438, False)]], 'filtered_resps': [(-114.10840606689453, False), (-82.98793029785156, False), (-93.36141967773438, False), (-93.36141967773438, False)], 'acc': 0.0, 'acc_norm': 1.0}]}, 'config': {'model': 'mgoin/llama2.c-stories15M-quant-pt', 'model_args': None, 'batch_size': 1, 'batch_sizes': [], 'device': None, 'use_cache': None, 'limit': 1, 'bootstrap_iters': 100000, 'gen_kwargs': None}, 'git_hash': 'ba5071079a'}
```
