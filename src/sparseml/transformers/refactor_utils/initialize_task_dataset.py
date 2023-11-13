from sparsezoo.utils.registry import RegistryMixin
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from transformers import PreTrainedTokenizerBase
from sparseml.transformers.masked_language_modeling import DataTrainingArguments

TEXT_CLASSIFICATION_TASKS = ["sequence-classification", "glue", "sentiment-analysis", "text-classification"]

class TaskDatasetRegistry(RegistryMixin):
    @classmethod
    def load_from_registry(cls, name: str) -> Callable[..., Any]:
        return cls.get_value_from_registry(name=name)

@TaskDatasetRegistry.register(name=["masked-language-modeling", "mlm"])
def dataset_function():
    from sparseml.transformers.masked_language_modeling import (
        get_tokenized_mlm_dataset,
    )
    return get_tokenized_mlm_dataset

@TaskDatasetRegistry.register(name=["question-answering","qa"])
def dataset_function():
    from sparseml.transformers.question_answering import (
        get_tokenized_qa_dataset,
    )

    return get_tokenized_qa_dataset

@TaskDatasetRegistry.register(name=["token-classification", "ner"])
def dataset_function():
    from sparseml.transformers.token_classification import (
        get_tokenized_token_classification_dataset,
    )

    return get_tokenized_token_classification_dataset

@TaskDatasetRegistry.register(name=TEXT_CLASSIFICATION_TASKS)
def dataset_function():
    from sparseml.transformers.text_classification import (
        get_tokenized_text_classification_dataset,
    )

    return get_tokenized_text_classification_dataset

def initialize_task_dataset(task:str, tokenizer: PreTrainedTokenizerBase, model: Optional[Any]=None, config: Optional[Any] = None, data_args: Dict[str, Any]= {}):
    tokenized_task_dataset = TaskDatasetRegistry.load_from_registry(task)()
    if task in TEXT_CLASSIFICATION_TASKS:
        return tokenized_task_dataset(tokenizer=tokenizer, model=model, config=config, data_args=DataTrainingArguments(**data_args))

    return tokenized_task_dataset(tokenizer=tokenizer, data_args=DataTrainingArguments(**data_args))



