# [Pipeline Template Example] SparseML Transformer Question Answer Integration

*This section serves as a short overview of the functionality of the pipeline. Should contain a brief intro plus a compact description of the use cases.*

SparseML Question Answer pipeline integrates with Hugging Face’s Transformers library to enable the sparsification of any transformer model.
Sparsification is a powerful feature that results in faster, smaller, and cheaper deployable Transformer models. 
The sparse model can be eventually deployed to Neural Magic's DeepSparse Engine. This allows running the inference with GPU-class performance directly on your CPU.

This integration enables spinning up one of the following end-to-end functionalities:
- **Sparsification of Popular Transformer Models** - easily sparsify any of the popular Hugging Face transformer models. 
- **Sparse Transfer Learning** - fine-tune a sparse backbone model (or use one of our [sparse pre-trained models](https://sparsezoo.neuralmagic.com/?page=1&domain=nlp&sub_domain=question_answering)) on your own, private dataset.

## Installation
*Description of minimal requirements to run the pipeline.*

```pip install sparseml[torch]```

It is recommended to run Python 3.8 as some of the scripts within the transformers repository require it.

Note: Transformers will not immediately install with this command. Instead, a sparsification-compatible version of Transformers will install on the first invocation of the Transformers code in SparseML.

## Tutorials
*Pointer to any existing tutorials published on neuralmagic.com.*
- [Sparsifying BERT Models Using Recipes](https://github.com/neuralmagic/sparseml/blob/main/integrations/huggingface-transformers/tutorials/sparsifying_bert_using_recipes.md)
- [Sparse Transfer Learning With BERT](https://github.com/neuralmagic/sparseml/blob/main/integrations/huggingface-transformers/tutorials/bert_sparse_transfer_learning.md)

## Getting started

*This should explain to the user in the learn-by-doing fashion about the functionalities of the pipeline.*

### Sparsification of Popular Transformer Models

Sparse ML Hugging Face Question Answer integration allows sparsifying any dense transformer.

In the example below, a dense BERT model is trained on the SQuAD dataset. By passing the recipe `zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned-aggressive_98` (located in [SparseZoo](https://sparsezoo.neuralmagic.com/models/nlp%2Fquestion_answering%2Fbert-base%2Fpytorch%2Fhuggingface%2Fsquad%2Fpruned-aggressive_98)) we modify (sparsify) the training process and/or the model.

```python
sparseml.transformers.question_answering \
  --model_name_or_path bert-base-uncased \          # name of the Hugging Face dense model
  --dataset_name squad \                            # name of the dataset we want to sparse train on
  --do_train \                                      # run training
  --do_eval \                                       # run evaluation on validation set 
  --output_dir './output' \                         # output directory of the saved model
  --cache_dir cache \                               # local directory to store the downloaded hugging face model."   
  --distill_teacher disable \                       # disable knowledge destillation
  --recipe zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned-aggressive_98          
```

### Sparse Transfer Learning

Once you sparsify a model using SparseML, you can easily sparse fine-tune it on a new dataset.
While you are free to use your backbone, we encourage you to leverage one of our [sparse pre-trained models](https://sparsezoo.neuralmagic.com) to boost your productivity!

In the example below, we fetch a pruned, quantized BERT model, pre-trained on Wikipedia and Bookcorpus datasets. We then fine tune the model to the SQuAD dataset. 
```python
sparseml.transformers.question_answering \
    --model_name_or_path zoo:nlp/masked_language_modeling/bert-base/pytorch/huggingface/wikipedia_bookcorpus/12layer_pruned80_quant-none-vnni \
    --dataset_name squad \
    --do_train \
    --do_eval \
    --output_dir './output' \ 
    --distill_teacher disable \
    --recipe zoo:nlp/masked_language_modeling/bert-base/pytorch/huggingface/wikipedia_bookcorpus/12layer_pruned80_quant-none-vnni?recipe_type=transfer-question_answering \
```

#### Knowledge Distillation
By modifying the `distill_teacher` argument, you can enable [Knowledge Destillation](https://neptune.ai/blog/knowledge-distillation) (KD) functionality.

In this example, the `--distill_teacher` argument is set to pull a dense SQuAD model from the SparseZoo to enable it to run independently of the dense teacher step:

```bash
--distill_teacher zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/base-none
```

Alternatively, the user may decide to train their own dense teacher model. The following command will use the dense BERT base model from the SparseZoo and fine-tune it on the SQuAD dataset.
```bash
sparseml.transformers.question_answering \
    --model_name_or_path zoo:nlp/masked_language_modeling/bert-base/pytorch/huggingface/wikipedia_bookcorpus/base-none \
    --dataset_name squad \
    --do_train \
    --do_eval \
    --output_dir models/teacher \
    --recipe zoo:nlp/masked_language_modeling/bert-base/pytorch/huggingface/wikipedia_bookcorpus/base-none?recipe_type=transfer-question_answering 
```

Once the dense teacher is trained we may reuse it for KD. This only requires passing the path to the directory with the model:

```bash
--distill_teacher models/teacher
```

## SparseML CLI
*This section should explain the functionality of the most important parameters of the CLI. The goal is not to teach the user about every argument, but to make it easy for them to tweak the crucial ones.*

The SparseML installation provides a CLI for sparsifying your models for a specific task; appending the `--help` argument will provide a full list of options for training in SparseML:
```bash
sparseml.transformers.question_answering --help
```
output:
```bash
  --model_name_or_path MODEL_NAME_OR_PATH
                        Path to pre-trained model or model identifier from huggingface.co/models
  --distill_teacher DISTILL_TEACHER
                        Teacher model which needs to be a trained QA model
  --cache_dir CACHE_DIR 
                        Directory path to store the pre-trained models downloaded from huggingface.co
  --recipe RECIPE       
                        Path to a SparseML sparsification recipe, see https://github.com/neuralmagic/sparseml for more information
  --dataset_name DATASET_NAME
                        The name of the dataset to use (via the datasets library).
  ...
```

To learn about the Hugging Face Transformers parameters in more detail, please also refer to [Hugging Face Transformers documentation](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments).

## Once the Training is Done...

*This section should inform the user what to do with the train user. It should serve as a walkthrough for generated artifacts and a pointer to next actions such as export to onnx, deployment on SparseEngine, GPU*

The artifacts of the training process are saved to the directory `--output_dir`. Once the script terminates, you should find there everything required to deploy or further modify the model. 

This includes the recipe (with the full description of the sparsification attributes), checkpoint files (saved in the appropriate framework format), Hugging Face Transformer's specific files (e.g. the tokenizer), etc.

### Exporting the Sparse Model to ONNX

*This subsection should explain to the user how to export their trained, sparse model to ONNX format.*

The DeepSparse Engine uses the ONNX format to load neural networks and then deliver breakthrough performance for CPUs by leveraging the sparsity and quantization within a network.

The SparseML installation provides a `sparseml.transformers.export_onnx` command that you can use to load the training model folder and create a new `model.onnx` file within. Be sure the `--model_path` argument points to your trained model. 
```bash
sparseml.transformers.export_onnx \
    --model_path './output' \
    --task 'question-answering' \
```

### DeepSparse Engine Deployment

*No pipeline is complete without its integration with the DeepSparse Engine. This chapter should communicate to the user,
that they can eventually use their ONNX model to deploy the network using the DeepSparse Engine and benefit from GPU-class performance
on their CPU! Let's make this section short, referencing internal DeepSparse docs as much as possible.* 

Once the model is exported an ONNX format, it is ready for deployment with the DeepSparse Engine. 

The deployment is intuitive and simple thanks to DeepSparse Python API.

```python
from deepsparse.transformers import pipeline

model_path = "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/12layer_pruned80_quant-none-vnni"

qa_pipeline = pipeline(
  task="question-answering", 
  model_path='./output'
)

inference = qa_pipeline(question="What's my name?", context="My name is Snorlax")

>> {'score': 0.9947717785835266, 'start': 11, 'end': 18, 'answer': 'Snorlax'}
```


To learn more, please refer to the [appropriate documentation in the DeepSparse repository](https://github.com/neuralmagic/deepsparse/tree/main/examples/huggingface-transformers)

## Support

For Neural Magic Support, sign up or log in to our [Deep Sparse Community Slack](https://join.slack.com/t/discuss-neuralmagic/shared_invite/zt-q1a1cnvo-YBoICSIw3L1dmQpjBeDurQ). Bugs, feature requests, or additional questions can also be posted to our [GitHub Issue Queue](https://github.com/neuralmagic/sparseml/issues).
