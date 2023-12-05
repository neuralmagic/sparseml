from sparsezoo.utils.registry import RegistryMixin
from pydantic import BaseModel, Field
from typing import Callable, Optional

class IntegrationHelperFunctions(BaseModel, RegistryMixin):
    """
    Registry that maps integration names to helper functions
    for creation/export/manipulation of models for a specific
    integration.
    """

    create_model: Optional[Callable] = Field(
        description="A function that creates a (sparse) "
        "PyTorch model from a source path."
    )
    create_dummy_input: Optional[Callable] = Field(
        description="A function that creates a dummy input "
        "given a (sparse) PyTorch model."
    )
    export_model: Optional[Callable] = Field(
        description="A function that exports a (sparse) PyTorch "
        "model to an ONNX format appropriate for a "
        "deployment target."
    )
    apply_optimizations: Optional[Callable] = Field(
        description="A function that takes a set of "
        "optimizations and applies them to an ONNX model."
    )
    export_sample_inputs_outputs: Optional[Callable] = Field(
        description="A function that exports input/output samples given "
        "a (sparse) PyTorch model."
    )
    create_deployment_folder: Optional[Callable] = Field(
        description="A function that creates a "
        "deployment folder for the exporter ONNX model"
        "with the appropriate structure."
    )



