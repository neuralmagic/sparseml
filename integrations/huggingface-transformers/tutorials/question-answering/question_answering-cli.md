# Question Answering: Sparse Transfer Learning with the CLI

In this example, you will sparse transfer learn a 90% pruned BERT model onto some extraction question answering datasets using SparseML's CLI.

### **Sparse Transfer Learning Overview**

Sparse Transfer Learning is very similiar to the typical transfer learing process used to train NLP models, where we fine-tune a pretrained checkpoint onto a smaller downstream dataset. With Sparse Transfer Learning, we simply start the fine-tuning process from a pre-sparsified checkpoint and maintain sparsity while the training process occurs.

### Pre-Sparsified BERT

SparseZoo, Neural Magic's open source repository of pre-sparsified models, contains a 90% pruned version of BERT, which has been sparsified on the upstream Wikipedia and BookCorpus datasets with the masked language modeling objective. [Check out the model card](https://sparsezoo.neuralmagic.com/models/nlp%2Fmasked_language_modeling%2Fobert-base%2Fpytorch%2Fhuggingface%2Fwikipedia_bookcorpus%2Fpruned90-none). We will use this model as the starting point for the transfer learning process.

**Let's dive in!**

## Installation

Install SparseML via `pip`.

```bash
pip install sparseml[torch]
```

## Sparse Transfer Learning onto MNLI (GLUE Task)

SparseML's CLI enables you to kick-off sparsification workflows with various utilities like creating training pipelines, dataset loading, checkpoint saving, metric reporting, and logging handled for you. 

All we have to do is pass a couple of key arguments: 
- `--model_name_or_path` specifies the starting checkpoint to load for training
- `--task` specifies a glue dataset to train with 
- `--recipe` specifies path a recipe to use to apply sparsification algorithms or sparse transfer learning to the model. For Sparse Transfer Learning, we will use a recipe that instructs SparseML to maintain sparsity during the training process and to apply quantization over the final few epochs. 

### Run Transfer Learning

We will fine-tune a [90% pruned version of BERT](zoo:nlp/masked_language_modeling/obert-base/pytorch/huggingface/wikipedia_bookcorpus/pruned90-none) onto MNLI.

Run the following:
```bash
sparseml.transformers.train.question_answering \
  --dataset_name squad \
  --model_name_or_path zoo:nlp/masked_language_modeling/obert-base/pytorch/huggingface/wikipedia_bookcorpus/pruned90-none \
  --recipe zoo:nlp/question_answering/obert-base/pytorch/huggingface/squad/pruned90_quant-none \
  --distill_teacher zoo:nlp/question_answering/obert-base/pytorch/huggingface/squad/base-none \
  --output_dir obert_base_pruned90_quant_squad \
  --do_train --do_eval --evaluation_strategy epoch --logging_steps 1000 --save_steps 1000 \
  --per_device_train_batch_size 8 --per_device_eval_batch_size 32 --gradient_accumulation_steps 2 --preprocessing_num_workers 32 \
  --max_seq_length 384 --doc_stride 128 \
  --seed 42
```

Let's discuss the key arguments:
- `--dataset_name squad` instructs SparseML to download and fine-tune onto the SQuAD dataset. The data is downloaded from the Hugging Face hub. You can pass any extractive QA dataset from the Hugging Face hub, provided it conforms to the SQuAD format (see below for details).

- `--zoo:nlp/masked_language_modeling/obert-base/pytorch/huggingface/wikipedia_bookcorpus/pruned90-none` specifies the starting checkpoint for the fine tuning. Here, we passed a SparseZoo stub identifying the 90% pruned version of BERT trained with masked language modeling, which SparseML downloads when the script starts.

- `--recipe zoo:nlp/question_answering/obert-base/pytorch/huggingface/squad/pruned90_quant-none` specifies the recipe to be applied by SparseML. Here, we passed a SparseZoo stub identifying the transfer learning recipe for the SQuAD dataset, which SparseML downloads when the script starts. See below for the details of what this recipe looks like.

- `--distill_teacher zoo:nlp/question_answering/obert-base/pytorch/huggingface/squad/base-none` is an optional argument that specifies a model to use for as a teacher to apply distillation during the training process. We passed a SparseZoo stub identifying a dense BERT model trained on SQuAD, which SparseML downloads when the script starts.

The model trains for 13 epochs, converging to and F1 score ~88% on the validation set. Because we applied a sparse transfer recipe, which instructs SparseML to maintain the sparsity of the starting pruned checkpoint and apply quantization, the final model is 90% pruned and quantized!

#### Transfer Learning Recipe

SparseML's recipes are YAML files that specify the sparsity related algorithms and parameters. SparseML parses the recipes and updates the training loops to apply the 
to apply sparsification algorithms or sparse transfer learning to the model.

In the case of SQuAD, we used a [premade recipe from the SparseZoo](https://sparsezoo.neuralmagic.com/models/nlp%2Fquestion_answering%2Fbert-large%2Fpytorch%2Fhuggingface%2Fsquad%2Fpruned90_quant-none). 

<details>
    <summary>Click to inspect the recipe</summary>

The `Modifiers` are the important items that encode how SparseML should modify the training process for Sparse Transfer Learning:
- `ConstantPruningModifier` tells SparseML to pin weights at 0 over all epochs, maintaining the sparsity structure of the network
- `QuantizationModifier` tells SparseML to quanitze the weights with quantization aware training over the last 5 epochs
- `DistillationModifier` tells SparseML how to apply distillation during the trainign process, targeting the logits

SparseML parses the modifiers and updates the training process to implement the algorithms and hyperparameters specified in the recipes.

```yaml
version: 1.1.0

# General Variables
num_epochs: 13

init_lr: 1.75e-4
final_lr: 0

qat_start_epoch: 8.0
observer_epoch: 12.0
quantize_embeddings: 1

distill_hardness: 1.0
distill_temperature: 5.0

weight_decay: 0.0

# Modifiers:
training_modifiers:
  - !EpochRangeModifier
      end_epoch: eval(num_epochs)
      start_epoch: 0.0

  - !LearningRateFunctionModifier
      start_epoch: 0
      end_epoch: eval(num_epochs)
      lr_func: linear
      init_lr: eval(init_lr)
      final_lr: eval(final_lr)

quantization_modifiers:
  - !QuantizationModifier
      start_epoch: eval(qat_start_epoch)
      disable_quantization_observer_epoch: eval(observer_epoch)
      freeze_bn_stats_epoch: eval(observer_epoch)
      quantize_embeddings: eval(quantize_embeddings)
      quantize_linear_activations: 0
      exclude_module_types: ['LayerNorm']
      submodules:
        - bert.embeddings
        - bert.encoder
        - qa_outputs

distillation_modifiers:
  - !DistillationModifier
     hardness: eval(distill_hardness)
     temperature: eval(distill_temperature)
     distill_output_keys: [start_logits, end_logits]

constant_modifiers:
  - !ConstantPruningModifier
      start_epoch: 0.0
      params: __ALL_PRUNABLE__

regularization_modifiers:
  - !SetWeightDecayModifier
      start_epoch: 0.0
      weight_decay: eval(weight_decay)
```

</details>

You can download the recipe with the following code:

```python
from sparsezoo import Model
transfer_stub = "zoo:nlp/question_answering/obert-base/pytorch/huggingface/squad/pruned90_quant-none"
download_dir = "./transfer_recipe-squad"
zoo_model = Model(transfer_stub, download_path=download_dir)
recipe_path = zoo_model.recipes.default.path
print(recipe_path)
```

### **Export to ONNX**

Once you have trained your model, export to ONNX in order to deploy with DeepSparse. The artifacts of the training process 
are saved to your local filesystem. 

Run the following to convert your PyTorch checkpoint to ONNX:

```bash
sparseml.transformers.export_onnx \
  --model_path obert_base_pruned90_quant_squad \
  --task question_answering
```

A `deployment` folder is created in your local directory, which has all of the files needed for deployment with DeepSparse including the `model.onnx`, `config.json`, and `tokenizer.json` files.

## Sparse Transfer Learning with a Custom Dataset (SquadShifts Amazon)

Beyond the SQuAD task, we can also pass a dataset from the Hugging Face Hub or pass via local files. Let's try an example with  for the extractive question answering using [Squadshifts Amazon Dataset](https://huggingface.co/datasets/squadshifts), which containing ~10,000 question answer pairs from the Amazon product reviews.

For simplicity, we will perform the fine-tuning without distillation. Although the transfer learning recipe contains distillation
modifiers, by setting `--distill_teacher disable` we instruct SparseML to skip distillation.

### Squadshifts Dataset Inspection

Run the following to inspect the Squadshifts Amazon dataset.

```python
from datasets import load_dataset
from pprint import pprint 

squadshifts = load_dataset("squadshifts", "amazon")["test"].train_test_split(test_size=.2)
pprint(squadshifts["train"][0])
```

Output:
```bash
{'answers': {'answer_start': [490, 490], 'text': ['very large', 'very large']},
 'context': 'This item is lightweight and very slim in design. In a '
            'kitchen,where space is limited, we found the scale was easy to '
            'use and quickly store, in an upright position, for instance. We '
            'love that it is flat and easy to clean. For instance, if '
            'something spills on it, it is very easy to wipe off-no nooks and '
            "cranies to worry about. We've also used this for shipping the "
            'occasional package,over the last week or so, and the flat surface '
            'is excellent for balancing small packages. The readout is very '
            'large and extremely crisp and clear, which makes assessing weight '
            'a snap. It also fits in nicely with the design of many kitchens '
            'and appliances on the market today (stainless steel, or '
            'black/white appliances. Highly recommended, multi-use tool!',
 'id': '5dd4b482cc027a086d65f11b',
 'question': 'how large is the display according to the writer?',
 'title': 'Amazon_Reviews_1525'}
```

We can see that each row dataset contains the following:
- A `context` field which is a string representing the text which contains the answer
- A `question` field which is a string representing the query
- An `answers` dictionary, which contains a `answers_start` (a list of ints) and `text` (a list of strings). `text` is the raw strings that are the correct answers
and `answer_start` are the index of the first character in the `context`. For the example above, the `v` in `very large` is the 490th character of `context`.

The `question_answering` training script accepts JSON files in the form:

### Using Local JSON Files

Let's walk through how to pass this dataset in JSON dataset to the CLI.

#### Save Dataset as a JSON File

The `question_answering` training script accepts JSON files in the form:

```bash
{
  'data': [
    {'question': 'What is my Name?', 'context': 'My name is Robert', "answers":{'answer_start':[11], 'text':['Robert']}},
    {'question': 'What is my Name?', 'context': 'My name is Mark', "answers":{'answer_start':[11], 'text':['Mark']}},
    {'question': 'What is my Name?', 'context': 'My name is Ben', "answers":{'answer_start':[11], 'text':['Ben']}},
    ...
  ]
}
```

Run the following to convert the dataset to this format and dump to a json file.

```python
# load dataset
from datasets import load_dataset
squadshifts = load_dataset("squadshifts", "amazon")["test"].train_test_split(test_size=.2)

# wrap dataset
train_dict = {"data":[]}
val_dict = {"data":[]}
for row in squadshifts["train"]:
  train_dict["data"].append(row)
for row in squadshifts["test"]:
  val_dict["data"].append(row)

# dump to json files
import json
def dict_to_json_file(path, dictionary):
  with open(path, 'w') as file:
      json_string = json.dumps(dictionary, default=lambda o: o.__dict__, sort_keys=True, indent=2)
      file.write(json_string)
dict_to_json_file("squadshifts-train.json", train_dict)
dict_to_json_file("squadshifts-val.json", val_dict)
```

#### **Kick off Training**

To use the local files with the CLI, pass `--train_file squadshifts-train.json --validation_file squadshifts-val.json`:

Run the following:
```bash
sparseml.transformers.train.question_answering \
  --output_dir obert_base_pruned90_quant_squadshifts \
  --recipe zoo:nlp/question_answering/obert-base/pytorch/huggingface/squad/pruned90_quant-none \
  --recipe_args '{"num_epochs":8, "qat_start_epoch":4.0, "observer_epoch":7.0}' \
  --model_name_or_path zoo:nlp/masked_language_modeling/obert-base/pytorch/huggingface/wikipedia_bookcorpus/pruned90-none \
  --distill_teacher disable \
  --train_file squadshifts-train.json --validation_file squadshifts-val.json \
  --do_train --do_eval --evaluation_strategy epoch --logging_strategy epoch --save_steps 1000 --preprocessing_num_workers 32 \
  --per_device_train_batch_size 8 --per_device_eval_batch_size 32 --gradient_accumulation_steps 2 \
  --max_seq_length 384 --doc_stride 128 \
  --seed 42
```

Without doing any hyperparameter search, the script runs for 8 epochs and converges to ~68% F1 score.

Note that in this case, we used the SQuAD transfer learning recipe (identified by 
`zoo:nlp/question_answering/obert-base/pytorch/huggingface/squad/pruned90_quant-none`). Since the Squadshifts dataset is similiar to the SQuAD dataset, 
we chose the same hyperparameters. While you are free to download and modify the recipe manually (and then pass to SparseML as a local file), you can also use `--recipe_args` to modify the recipe on the fly. 

In this case, we passed `--recipe_args '{"num_epochs":8, "qat_start_epoch":4.0, "observer_epoch":7.0}'`. This updates the recipe to run
for 8 epochs with QAT running over the final 4 epochs.

### Sparse Transfer Learning with a Custom Teacher

To support the transfer learning process, we can apply model distillation, just like we did for the SQuAD case.
You are free to use the native Hugging Face workflows to train the dense teacher model (and can even
pass a Hugging Face model identifier to the command), but you can also use the SparseML CLI as well. 

#### Train The Dense Teacher

Run the following to train a dense teacher model on SquadShifts:

```bash
sparseml.transformers.train.question_answering \
  --output_dir dense_teacher \
  --recipe zoo:nlp/question_answering/obert-base/pytorch/huggingface/squad/base-none \
  --recipe_args '{"num_epochs":5, "init_lr":0.0002}' \
  --model_name_or_path zoo:nlp/masked_language_modeling/obert-base/pytorch/huggingface/wikipedia_bookcorpus/base-none \
  --distill_teacher disable \
  --train_file squadshifts-train.json --validation_file squadshifts-val.json \
  --do_train --do_eval --evaluation_strategy epoch --logging_strategy epoch --save_steps 1000 --preprocessing_num_workers 32 \
  --per_device_train_batch_size 8 --per_device_eval_batch_size 32 --gradient_accumulation_steps 2 \
  --max_seq_length 384 --doc_stride 128 \
  --seed 42
```

Note that used the dense version of BERT (the stub ends in `base-none`) as the starting point for the training 
and passed a recipe from [SparseZoo](https://sparsezoo.neuralmagic.com/models/nlp%2Fquestion_answering%2Fbert-large%2Fpytorch%2Fhuggingface%2Fsquad%2Fbase-none) which was used to train the 
dense teacher for the SQuAD task. Since the SQuAD task is similiar to the Squadshifts Amazon task, these hyperparameters are a solid starting point. This recipe contains no sparsity related modifiers and only controls the learning rate and number of epochs. As such, the script
will run typical fine-tuning, resulting in a dense model.

Here's what the recipe looks like:
```yaml
version: 1.1.0

# General Variables
num_epochs: 3
init_lr: 5e-5 
final_lr: 0

warmup_epoch: 0.033

# Modifiers:
training_modifiers:
  - !EpochRangeModifier
      end_epoch: eval(num_epochs)
      start_epoch: 0.0

  - !LearningRateFunctionModifier
    start_epoch: 0
    end_epoch: eval(warmup_epoch)
    lr_func: linear
    init_lr: 0.0
    final_lr: eval(init_lr)

  - !LearningRateFunctionModifier
    start_epoch: eval(warmup_epoch)
    end_epoch: eval(num_epochs)
    lr_func: linear
    init_lr: eval(init_lr)
    final_lr: eval(final_lr)
```

While you are free to download and modify the recipe manually (and then pass to SparseML as a local file), you
can also use `--recipe_args` to modify the recipe on the fly.

In this case, we passed `--recipe_args '{"num_epochs":5, "init_lr":0.0002}'`. This updates the recipe to run
for 5 epochs instead of 3 and to use an initial learning rate of `0.0002` instead of `5e-5`.

The model converges to ~70% accuracy without any hyperparameter search.

#### Sparse Transfer Learning with a Custom Teacher

With the dense teacher trained, we can sparse transfer learn with the help of the teacher by passing
`--distill_teacher ./dense_teacher`.

Run the following to kick-off training with the teacher:

```bash
sparseml.transformers.train.question_answering \
  --output_dir obert_base_pruned90_quant_squadshifts \
  --recipe zoo:nlp/question_answering/obert-base/pytorch/huggingface/squad/pruned90_quant-none \
  --recipe_args '{"num_epochs":8, "qat_start_epoch":4.0, "observer_epoch":7.0}' \
  --model_name_or_path zoo:nlp/masked_language_modeling/obert-base/pytorch/huggingface/wikipedia_bookcorpus/pruned90-none \
  --distill_teacher ./dense_teacher \
  --train_file squadshifts-train.json --validation_file squadshifts-val.json \
  --do_train --do_eval --evaluation_strategy epoch --logging_strategy epoch --save_steps 1000 --preprocessing_num_workers 32 \
  --per_device_train_batch_size 8 --per_device_eval_batch_size 32 --gradient_accumulation_steps 2 \
  --max_seq_length 384 --doc_stride 128 \
  --seed 42
```

The model converges to ~69% F1 without any hyperparameter search after 8 epochs.