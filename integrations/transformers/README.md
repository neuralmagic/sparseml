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
lim itations under the License.
-->
# Transformers-SparseML Integration
This folder contains an example on how to use sparseml with transformers. 
We focus on Question answering and use a modified implementation from the BERT SQuAD in transformers. 
Using various pruning configuration files we demostrate the effect unstructured pruning can have on SQuAD. The example code is absed on the transformers SQUAD implementation focused on BERT on the SQuAD1.0 dataset. It runs in 120 min (with BERT-base) a single tesla V100 16GB.
## Installation and Requirements
These example scripts require sparseml, transformers, torch, datasets and associated to libraries. To install run the following command

```bash
pip install torch sparseml transformers datasets
```

## Usage
To custom prune a model first go to the prune-config.yaml file and modify the parameters to your needs. We have provided a range of pruning configurations in the prune_config_files folder. 
!EpochRangeModifier controls how long the model trains for and Each !GMPruningModifier modifies controls how each portion is pruned. You can modify end_epoch to control how long the pruning regime lasts and final_sparsity and init_sparsity define the speed which the module is pruned and the final sparsity.
### Training 
```bash
python run_qa.py  \
 --model_name_or_path bert-base-uncased \
 --dataset_name squad \
 --do_train \
 --per_device_train_batch_size 12 \
 --learning_rate 3e-5 \
 --max_seq_length 384 \
 --doc_stride 128 \
 --output_dir bert-base-uncased-90-1shot/ \
 --overwrite_output_dir \
 --cache_dir cache \
 --preprocessing_num_workers 4 \
 --seed 42 \
 --num_train_epochs 2 \
 --nm_prune_config recipes/90sparsity1shot.yaml
```

#### Evaluation
```bash
python run_qa.py  \
 --model_name_or_path bert-base-uncased-99sparsity-10total8gmp/ \
 --dataset_name squad \
 --do_eval \
 --per_device_eval_batch_size 12 \
 --output_dir bert-base-uncased-99sparsity-10total8gmp/ \
 --overwrite_output_dir \
 --cache_dir cache \
 --preprocessing_num_workers 4 \
```
#### ONNX Export
```bash
python run_qa.py  \
 --model_name_or_path bert-base-uncased-99sparsity-10total8gmp/
 --do_eval  \
 --dataset_name squad \
 --do_onnx_export \
 --onnx_export_path bert-base-uncased-99sparsity-10total8gmp/ \
 --cache_dir cache \
 --preprocessing_num_workers 4 \
```

## Model Performance 
To demostrate the effect that various pruning regimes and techniques can have we prune the same bert-base-uncased model to 5 different sparsities(0,80,90,95,99) using 3 pruning methodologies: oneshot(prune to desired weights before fine tune then fine tune for 1 epoch), GMP 1 epoch(prune to desired sparsity over an epoch then stabilize over another epoch), and GMP 8 epochs (prune to desired sparsity over 8 epochs then stabilize over another 2 epochs). Its worth noting that we are pruning all layers uniformly and we believe further gains can be have by targeted pruning of individual layers.

| base model name       | sparsity 	| total train epochs    | prunned | one shot |pruning epochs| F1 Score 	| EM Score  |
|-----------------------|----------	|-----------------------|---------|----------|--------------|----------	|-----------|
| bert-base-uncased 	|0        	|1                  	|no       |no        |0            	|09.685     |3.614      |
| bert-base-uncased 	|0        	|2                  	|no       |no        |0            	|88.002     |80.634     |
| bert-base-uncased 	|0        	|10                 	|no       |no        |0            	|87.603     |79.130     |
| bert-base-uncased 	|80       	|1                  	|yes      |yes       |0          	|25.141     |15.998     |
| bert-base-uncased 	|80       	|2                   	|yes      |no        |0            	|06.068    	|00.312     |
| bert-base-uncased 	|80       	|10                  	|yes      |no        |8          	|83.951     |74.409     |
| bert-base-uncased 	|90       	|1                  	|yes      |yes       |0           	|16.064     |07.786     |
| bert-base-uncased 	|90       	|2                   	|yes      |no        |0            	|64.185     |50.946     |
| bert-base-uncased 	|90       	|10                 	|yes      |no        |8            	|79.091     |68.184     |
| bert-base-uncased 	|95       	|1                  	|yes      |yes       |0           	|10.501     |4.929      |
| bert-base-uncased 	|95       	|2                   	|yes      |no        |0            	|24.445     |14.437     |
| bert-base-uncased 	|95       	|10                 	|yes      |no        |8            	|72.761  	|60.407     |
| bert-base-uncased 	|99         |1                   	|yes      |yes       |0             |09.685     |03.614     |
| bert-base-uncased 	|99       	|2                   	|yes      |no        |0            	|17.433     |07.871     |
| bert-base-uncased 	|99         |10                    	|yes      |no        |8             |47.306    	|32.564     |


## Script origin and how to integrate sparseml with other Transformers projects
This script is based on the example BERT-QA implementation in transformers found [here](https://github.com/huggingface/transformers/blob/master/examples/question-answering/run_qa.py). 

For any other projects combining huggingface transformer's there are essentially four components to modify: imports and needed function, loading sparseml, modifying training script, and onnx export. 

First take your existing project and add the following imports and functions
```python
from transformers.optimization import (
    Adafactor,
    AdamW,
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)

from sparseml.pytorch.optim.manager import ScheduledModifierManager
from sparseml.pytorch.optim.optimizer import ScheduledOptimizer
from sparseml.pytorch.utils import ModuleExporter
from sparseml.pytorch.helpers import any_str_or_regex_matches_param_name


def get_sparsity_by_regex(module: Module, param_names: List[str]):
    """
    :param module: the module to get the matching layers and params from
    :param param_names: a list of names or regex patterns to match with full parameter
        paths. Regex patterns must be specified with the prefix 're:'
    ::return sparsity: a float representing the percetage of zero-weight parameters
    """
    param_count = 0
    param_set_to_zero = 0
    for layer_name, layer in module.named_modules():
        for param_name, param in layer.named_parameters():
            if "." in param_name:  # skip parameters of nested layers
                continue
            full_param_name = "{}.{}".format(layer_name, param_name)
            if any_str_or_regex_matches_param_name(full_param_name, param_names):
                param_count += np.prod(param.size())
                param_set_to_zero += torch.sum(param.data == 0)
    sparsity = float(param_set_to_zero / param_count)
    return sparsity


def load_optimizer(model, args):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer_cls = AdamW
    optimizer_kwargs = {
        "betas": (args.adam_beta1, args.adam_beta2),
        "eps": args.adam_epsilon,
    }
    optimizer_kwargs["lr"] = args.learning_rate
    return optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
```
Use the code below to load sparseml optimizers
```python
## Neural Magic Integration here. 
optim = load_optimizer(model, TrainingArguments) #We first create optimizers based on the method defined in transformers trainer class
steps_per_epoch = math.ceil(len(datasets["train"]) / (training_args.n_gpu * training_args.per_device_train_batch_size))
manager = ScheduledModifierManager.from_yaml(data_args.nm_prune_config) # Load a NM pruning config

optim = ScheduledOptimizer(
    optim, model, manager, steps_per_epoch=steps_per_epoch, loggers=None
)
```
Modify the hugging face trainer to take the sparseml optimzier as shown below
```python
# Initialize our Trainer and continue to use your regular transformers trainer
trainer = QuestionAnsweringTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset if training_args.do_train else None,
    eval_dataset=validation_dataset if training_args.do_eval else None,
    eval_examples=datasets["validation"] if training_args.do_eval else None,
    tokenizer=tokenizer,
    data_collator=data_collator,
    post_process_function=post_processing_function,
    compute_metrics=compute_metrics,
    optimizers=(optim, None), # This is what is new.
)
```
Finally, export the model. Its worth noting that you will have to create a sample batch which will be task dependent. The code shown below is specific for SQuAD style question answering
```python
exporter = ModuleExporter(
    model, output_dir=data_args.onnx_export_path
)
exporter.export_onnx(sample_batch=sample_batch)
```