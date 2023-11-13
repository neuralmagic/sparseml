import pytest
import os
import numpy as np
from sparsezoo import Model
from src.sparseml.transformers.refactor_utils.export_samples import export_samples
from src.sparseml.transformers.refactor_utils.initialize_task_dataset import initialize_task_dataset
from src.sparseml.transformers.refactor_utils.initialize_model import initialize_transformer_model


# @pytest.fixture()
# def model_path(tmp_path):
#     return Model(
#         "zoo:mobilebert-squad_wikipedia_bookcorpus-14layer_pruned50.4block_quantized", tmp_path
#     ).training.path

@pytest.fixture()
def model_path():
    return Model(
        "zoo:mobilebert-squad_wikipedia_bookcorpus-14layer_pruned50.4block_quantized"
    ).training.path


@pytest.fixture()
def sequence_length():
    return 384


@pytest.fixture()
def task():
    return "qa"

def test_real_export_samples(model_path, task):
    model, trainer, config, tokenizer = initialize_transformer_model(model_path=model_path, sequence_length=384, task=task)
    dataset = initialize_task_dataset(task=task, tokenizer=tokenizer, data_args=dict(dataset_name = "squad", max_seq_length=384))
    trainer.eval_dataset = dataset
    with pytest.raises(ValueError):
        export_samples(trainer=trainer, tokenizer=tokenizer, num_samples=5, real_samples = True)

def test_real_export_samples_but_no_dataset(model_path, task):
    model, trainer, config, tokenizer = initialize_transformer_model(model_path=model_path, sequence_length=384, task=task)
    with pytest.raises(ValueError):
        export_samples(trainer=trainer, tokenizer=tokenizer, num_samples=5, real_samples = True)

@pytest.mark.parametrize(
    "num_samples",
    [0, 5, 10],
)
def test_export_samples(model_path, task, num_samples):
    model, trainer, config, tokenizer = initialize_transformer_model(model_path=model_path, sequence_length=384, task=task)
    export_samples(trainer=trainer, tokenizer=tokenizer, num_samples=num_samples)

    assert "sample-inputs" in os.listdir(os.path.dirname(model_path))
    assert "sample-outputs" in os.listdir(os.path.dirname(model_path))

    outputs_dir = os.path.join(os.path.dirname(model_path), "sample-outputs")
    inputs_dir = os.path.join(os.path.dirname(model_path), "sample-inputs")

    assert len(os.listdir(inputs_dir)) == len(os.listdir(outputs_dir)) == num_samples

    if num_samples:
        assert np.load(os.path.join(inputs_dir, "inp-0000.npz"))
        assert np.load(os.path.join(outputs_dir, "out-0000.npz"))

