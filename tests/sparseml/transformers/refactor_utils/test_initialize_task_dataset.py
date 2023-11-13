from src.sparseml.transformers.refactor_utils.initialize_task_dataset import initialize_task_dataset
from src.sparseml.transformers.refactor_utils.initialize_model import initialize_tokenizer
import pytest
from sparsezoo import Model

@pytest.mark.parametrize(
    "task, stub, data_args",
    [
     #("masked-language-modeling", "zoo:obert-small-wikipedia_bookcorpus-pruned80.4block_quantized", dict(dataset_name = "squad", max_seq_length=384)),
     # ("question-answering", "zoo:mobilebert-squad_wikipedia_bookcorpus-14layer_pruned50.4block_quantized", dict(dataset_name = "squad", max_seq_length=384)),
     ("text-classification","zoo:distilbert-qqp_wikipedia_bookcorpus-pruned80.4block_quantized", dict(dataset_name = "qqp", max_seq_length=384)),
     #("token-classification", None)
    ]
)
def test_initialize_task_dataset(task, stub, data_args):
    dataset = initialize_task_dataset(task=task,
                            tokenizer=initialize_tokenizer(model_path = Model(stub).training.path, task=task, sequence_length=384),
                            data_args=data_args
                            )
    assert dataset.get('validation')