# [Pipeline Template Example] Sparse Transformer Question Answer

*This section serves as a short overview of the functionality of the pipeline. Should contain a brief intro plus a compact description of the use cases.*

Neural Magic’s Sparse Question Answer pipeline integrates with Hugging Face’s Transformers library to enable sparsifying any transformer model.
Sparsification is a powerful feature that results in faster, smaller, and cheaper deployable extractive Question Answer models. 
The sparse model can be eventually easily deployed to Neural Magic's DeepSparse Engine, which enables running the inference with GPU-class performance directly on your CPU.

This pipeline allows kicking off multiple end-to-end experiences:
- **Sparsification of Popular Transformer Models** - easily sparsify any of the popular Hugging Face transformer models.
- **Sparse Transfer Learning** - fine-tune one of our [sparse pre-trained models](https://sparsezoo.neuralmagic.com/?page=1&domain=nlp&sub_domain=question_answering) on your own, private dataset.
- **Knowledge Destillation** - any of the above-described scenarios can be further enhanced through the built-in [Knowledge Destillation](https://neptune.ai/blog/knowledge-distillation) functionality.

## Installation

*Description of minimal requirements to run the pipeline.*

## Getting started

*This should explain to the user in the learn-by-doing fashion about the functionalities of the pipeline.*

### Sparsification of Popular Transformer Models

To run a simple Question Answering pipeline, all you need is one command:

```python
python transformers/question_answering.py \
  --model_name_or_path bert-base-uncased \          # name of the Hugging Face dense model
  --dataset_name squad \                            # name of the dataset we want to sparse train on
  --do_train \                                      # run training
  --do_eval \                                       # run evaluation on validation set 
  --output_dir './output' \                         # output directory
  --cache_dir cache \                               # local directory to store the downloaded hugging face model."
  --num_train_epochs 30 \                           # total number of training epochs
  --recipe recipe.yaml \                            # sparsification recipe for sparse training
```

The script fetches dense `bert-base-uncased` and trains the model on the `squad` dataset. However, by passing the recipe `recipe.yaml` we modify (sparsify) the training process or the model.

### Sparse Transfer Learning
...

### Knowledge Destillation
...

Knowledge Distillation (KD) speeds up inference and maintains accuracy while transferring knowledge from a pre-trained cumbersome teacher model to a compact student model



## Parameters
*This section should explain the functionality of the most important parameters of the CLI. The goal is not to teach the user about every argument, but to make it easy for them to tweak the crucial ones.*

We recommend the user interact with the sparse question answering transformer pipeline using the built-in CLI.
You can easily adapt the pipeline by choosing an adequate set of parameters. Note: Many arguments should be familiar to the users experienced with Hugging Face Transformer API. 

The most important parameters are explained below:
- `test` - test
- `...` - ...

To learn about the parameters in more detail, please also refer to [Hugging Face Transformers documentation](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments)

## Once the Training is Done...

*This section should inform the user what to do with the train user. It should serve as a walkthrough for generated artifacts and a pointer to next actions such as export to onnx, deployment on SparseEngine, GPU*

### Exporting the Sparse Model to ONNX

*This subsection should explain to the user how to export their trained, sparse model to ONNX format.*

### DeepSparse Engine Deployment

*No pipeline is complete without its integration with the DeepSparse Engine. This chapter should communicate to the user,
that they can eventually use their ONNX model to deploy the network using the DeepSparse Engine and benefit from GPU-class performance
on their CPU! Let's make this section short, referencing internal DeepSparse docs as much as possible.* 

Once the model is saved in an ONNX format, it is ready for deployment with the DeepSparse Engine. 

The deployment is intuitive and simple thanks to DeepSparse Python API.

```python
from deepsparse.transformers import pipeline

model_path = "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/12layer_pruned80_quant-none-vnni"

qa_pipeline = pipeline(
  task="question-answering", 
  model_path=model_path
)

inference = qa_pipeline(question="What's my name?", context="My name is Snorlax")
print(inference)
```
printout:

    {'score': 0.9947717785835266, 'start': 11, 'end': 18, 'answer': 'Snorlax'}

To learn more, please refer to the [appropriate documentation in the DeepSparse repository](https://github.com/neuralmagic/deepsparse/tree/main/examples/huggingface-transformers)

