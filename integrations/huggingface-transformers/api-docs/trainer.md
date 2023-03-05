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

# **Trainer**

SparseML's `Trainer` extends the [Hugging Face `Trainer`](https://huggingface.co/docs/transformers/v4.26.1/en/main_classes/trainer#transformers.Trainer) to enable usage of Recipes and Model Distillation during the training loop.

The `Trainer` class parses the `recipe` passed to the constructor and modifies the training loop with the specified instructions. If the `recipe` includes
a `DistillationModifier`, specifying how model distillation should be applied, you can also pass a trained dense model to `distill_teacher`, enabling
you to apply distillation during training.

Other than these modification, SparseML's `Trainer` works just like the Hugging Face `Trainer` - this makes it very easy to add SparseML to your
existing training loops to prune models from scratch and to fine-tune pre-sparsified checkpoints onto downstream tasks.

### Signature

The class is called `sparseml.transformers.sparsification.Trainer`:

#### Parameters

The class accepts the following arguments:

- `model` - transformers compatible `Module` which should be trained

- `model_state_path` - the state path of the model
- `recipe`
- `recipe_args`
- `metadata_args`
- `teacher`
- `kwargs`

```
class sparseml.transformers.sparsification.Trainer
    :param model: the model to use with the trainer and apply sparsification to
    :param model_state_path: the state path to the model,
        used to load config and tokenizer settings
    :param recipe: the recipe, if any, to apply to the model and training
        process
    :param recipe_args: A json string, csv key=value string, or dictionary containing
        arguments to override the root arguments within the recipe such as
        learning rate or num epochs
    :param metadata_args A list of arguments to be extracted from training_args
        and passed as metadata for the final, saved recipe.
    :param teacher: teacher model for distillation. Set to 'self' to distill
        from the loaded model or 'disable' to turn off distillation
    :param kwargs: key word arguments passed to the parent class
```
