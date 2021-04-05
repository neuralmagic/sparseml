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
# Transformers-SparseML Integration
This folder contains an example on how to use sparseml with transformers. 
We focus on Question answering and use a modified implementation from the BERT SQuAD in transformers. 
Using various pruning configuration files we demostrate the effect unstructured pruning can have on SQuAD. The example code is based on the transformers SQUAD implementation focused on BERT on the SQuAD1.0 dataset. It runs in 120 min (with BERT-base) on a single Tesla V100 16GB.
## Installation and Requirements
These example scripts require sparseml, transformers, torch, datasets and associated libraries. To install run the following command

```bash
pip install sparseml[torch] torch transformers datasets
```

## Usage
To custom prune a model first go to the prune-config.yaml file and modify the parameters to your needs. We have provided a range of pruning configurations in the prune_config_files folder. 
!EpochRangeModifier controls how long the model trains for and each !GMPruningModifier modifies controls how each portion is pruned. You can modify end_epoch to control how long the pruning regime lasts and final_sparsity and init_sparsity define the speed at which the module is pruned and the final sparsity.
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
To demostrate the effect that various pruning regimes and techniques can have we prune the same bert-base-uncased model to 5 different sparsities(0,80,90,95,99) using 3 pruning methodologies: oneshot(prune to desired weights before fine tune then fine tune for 1 epoch), GMP 1 epoch(prune to desired sparsity over an epoch then stabilize over another epoch), and GMP 8 epochs (prune to desired sparsity over 8 epochs then stabilize over another 2 epochs). Its worth noting that we are pruning all layers uniformly and we believe further gains can be achieved by targeted pruning of individual layers.

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

def convert_example_to_features(example, tokenizer, max_seq_length, doc_stride, max_query_length):
    Feature = collections.namedtuple(
        "Feature",
        [
            "unique_id",
            "tokens",
            "example_index",
            "token_to_orig_map",
            "token_is_max_context",
        ],
    )
    extra = []
    unique_id = 0
    query_tokens = tokenizer.tokenize(example["question"])[0:max_query_length]
    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(example["context"]):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)
    max_tokens_for_doc = max_seq_length - len(query_tokens) - 3
    _DocSpan = collections.namedtuple("DocSpan", ["start", "length"])
    doc_spans = []
    start_offset = 0
    while start_offset < len(all_doc_tokens):
        length = len(all_doc_tokens) - start_offset
        if length > max_tokens_for_doc:
            length = max_tokens_for_doc
        doc_spans.append(_DocSpan(start=start_offset, length=length))
        if start_offset + length == len(all_doc_tokens):
            break
        start_offset += min(length, doc_stride)
    for (doc_span_index, doc_span) in enumerate(doc_spans):
        tokens = []
        token_to_orig_map = {}
        token_is_max_context = {}
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in query_tokens:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)
        for i in range(doc_span.length):
            split_token_index = doc_span.start + i
            token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]
            is_max_context = _check_is_max_context(
                doc_spans, doc_span_index, split_token_index
            )
            token_is_max_context[len(tokens)] = is_max_context
            tokens.append(all_doc_tokens[split_token_index])
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
        feature = Feature(
            unique_id=unique_id,
            tokens=tokens,
            example_index=0,
            token_to_orig_map=token_to_orig_map,
            token_is_max_context=token_is_max_context,
        )
        extra.append(feature)
        unique_id += 1
        # extra is used as additional data but sparseml doesn't support it
    return (
        torch.from_numpy(np.array([np.array(input_ids, dtype=np.int64)])),
        torch.from_numpy(np.array([np.array(input_mask, dtype=np.int64)])),
        torch.from_numpy(np.array([np.array(segment_ids, dtype=np.int64)])),
    )


def _check_is_max_context(doc_spans, cur_span_index, position):
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index
    return cur_span_index == best_span_index
```
We add some sparseml arguments
```python
@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    ####################################################################################
    # Start SparseML Integration
    ####################################################################################
    nm_prune_config: Optional[str] = field(
        default='recipes/noprune1epoch.yaml', metadata={"help": "The input file name for the Neural Magic pruning config"}
    )
    do_onnx_export: bool = field(
        default=False, metadata={"help": "Export model to onnx"}
    )
    onnx_export_path: Optional[str] = field(
        default='onnx-export', metadata={"help": "The filename and path which will be where onnx model is outputed"}
    )
    ####################################################################################
    # End SparseML Integration
    ####################################################################################
```
Use the code below to load sparseml optimizers
```python
## Neural Magic Integration here. 
optim = load_optimizer(model, TrainingArguments)
steps_per_epoch = math.ceil(len(datasets["train"]) / (training_args.per_device_train_batch_size*training_args._n_gpu))
manager = ScheduledModifierManager.from_yaml(data_args.nm_prune_config)
optim = ScheduledOptimizer(optim, model, manager, steps_per_epoch=steps_per_epoch, loggers=None)
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
    model, output_dir='onnx-export'
)
sample_batch = convert_example_to_features(
    datasets["validation"][0],
    tokenizer,
    data_args.max_seq_length,
    data_args.doc_stride,
    data_args.max_query_length,
)
exporter.export_onnx(sample_batch=sample_batch)
```
