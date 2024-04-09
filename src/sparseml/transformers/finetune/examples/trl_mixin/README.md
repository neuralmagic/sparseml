# Sparse Finetuning with TRL's SFTTrainer

The `SessionManagerMixin` can be added to other Trainer classes that inherit from 
[Hugging Face's Trainer](https://huggingface.co/docs/transformers/en/main_classes/trainer).

For example, we can add SparseML support to TRL's SFTTrainer like so: 

```python
from trl import SFTTrainer as TRLSFTTrainer

class SFTTrainer(SessionManagerMixIn, TRLSFTTrainer):
    ...
```

The new `SFTTrainer` class can now apply SparseML recipes and modifiers during 
supervised finetuning, will full support for all of the original TRL features. 

### Examples

[test_trl_sft_data.py](test_trl_sft_data.py): finetunes a 50% sparse Llama-7b model,
using TRL's dataset preprocessing. Sparsity is maintained throughout training by 
applying a `ConstantPruningModifier` recipe to the `SFTTrainer` 

[test_trl_distillation.py](test_trl_distillation.py): finetunes a 50% sparse Llama-7b 
model using knowledge distillation from a dense Llama-7b model. Sparsity is maintained 
throughout training with a `ConstantPruningModifier` and layer-wise knowledge 
distillation is handled by the `OutputDistillationModifier`