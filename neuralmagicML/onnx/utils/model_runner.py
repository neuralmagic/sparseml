from abc import ABC, abstractmethod
from time import time
from typing import Callable, Dict, List, Union

import numpy as np
import onnx
import onnxruntime as rt

__all__ = ["ModelRunner", "LossRunner", "ORTModelRunner"]


class ModelRunner(ABC):
    @abstractmethod
    def run(self, inputs: List, **kwargs):
        pass


class LossRunner(ModelRunner):
    def __init__(self, model, loss_function, model_runner=None):
        if model_runner is None:
            model_runner = ORTModelRunner
        self._model_runner = model_runner(model)
        self._loss_function = loss_function

    def run(self, inputs: List, expected: List):
        return [
            dict(output, loss=self._loss_function(output["output"], expected[index]))
            for index, output in enumerate(self._model_runner.run(inputs))
        ]

        return self._loss_function(self._model_runner.run(inputs), expected)


class ORTModelRunner(ModelRunner):
    def __init__(self, model):
        self._model = model
        sess_options = rt.SessionOptions()
        sess_options.log_severity_level = 3
        self._session = rt.InferenceSession(
            self._model.SerializeToString(), sess_options
        )

    def run(
        self, inputs: List,
    ):
        outputs = []
        for sample_inputs in inputs:
            sess_inputs = {}
            for inp_index, inp_name in enumerate(self._session.get_inputs()):
                sess_inputs[inp_name.name] = sample_inputs[inp_index]

            start_time = time()
            sess_out = self._session.run(
                [out.name for out in self._session.get_outputs()], sess_inputs
            )
            total_time = time() - start_time

            outputs.append({"output": sess_out, "timing": total_time})
        return outputs
