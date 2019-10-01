from typing import Union, Tuple, List
import time
from tqdm import tqdm
import torch
from torch import Tensor
from torch.nn import Module

from ..models import model_to_device


__all__ = ['BenchmarkResults', 'ModelBenchmarker']


class BenchmarkResults(object):
    def __init__(self):
        self._model_total = 0.0
        self._e2e_total = 0.0
        self._num_batches = 0
        self._num_items = 0

    def __str__(self) -> str:
        return 'BenchmarkResults(e2e=[items/sec={}, ms/item={}], mod=[items/sec={}, ms/item={}])' \
            .format(self.e2e_items_per_second, self.e2e_ms_per_item,
                    self.model_items_per_second, self.model_ms_per_item)

    @property
    def model_items_per_second(self) -> float:
        res = float(self._num_items) / self._model_total

        return res

    @property
    def model_items_per_ms(self) -> float:
        res = self.model_items_per_second
        res = res / 1000.0

        return res

    @property
    def model_sec_per_item(self) -> float:
        res = 1.0 / self.model_items_per_second

        return res

    @property
    def model_ms_per_item(self) -> float:
        res = 1.0 / self.model_items_per_ms

        return res

    @property
    def model_batches_per_second(self) -> float:
        res = float(self._num_batches) / self._model_total

        return res

    @property
    def model_batches_per_ms(self) -> float:
        res = self.model_batches_per_second
        res = res / 1000.0

        return res

    @property
    def model_sec_per_batch(self) -> float:
        res = 1.0 / self.model_batches_per_second

        return res

    @property
    def model_ms_per_batch(self) -> float:
        res = 1.0 / self.model_batches_per_ms

        return res

    @property
    def e2e_items_per_second(self) -> float:
        res = float(self._num_items) / self._e2e_total

        return res

    @property
    def e2e_items_per_ms(self) -> float:
        res = self.e2e_items_per_second
        res = res / 1000.0

        return res

    @property
    def e2e_sec_per_item(self) -> float:
        res = 1.0 / self.e2e_items_per_second

        return res

    @property
    def e2e_ms_per_item(self) -> float:
        res = 1.0 / self.e2e_items_per_ms

        return res

    @property
    def e2e_batches_per_second(self) -> float:
        res = float(self._num_batches) / self._e2e_total

        return res

    @property
    def e2e_batches_per_ms(self) -> float:
        res = self.e2e_batches_per_second
        res = res / 1000.0

        return res

    @property
    def e2e_sec_per_batch(self) -> float:
        res = 1.0 / self.e2e_batches_per_second

        return res

    @property
    def e2e_ms_per_batch(self) -> float:
        res = 1.0 / self.e2e_batches_per_ms

        return res

    def add_result(self, model_sec: float, e2e_sec, batch_size: int):
        self._model_total += model_sec
        self._e2e_total += e2e_sec
        self._num_batches += 1
        self._num_items += batch_size


class ModelBenchmarker(object):
    def __init__(self, name: str, model: Module, device: str):
        self._name = name
        model, device, device_ids = model_to_device(model, device)
        self._model = model
        self._device = device
        self._device_ids = device_ids

    def benchmark_batch(self, batch: Union[Tensor, List[Tensor], Tuple[Tensor, ...]], full_precision: bool,
                        test_size: int = 100, warmup_size: int = 10) -> BenchmarkResults:
        model = self._model.eval()
        ModelBenchmarker._convert_precisions(model, batch, full_precision)

        for _ in tqdm(range(warmup_size), desc='warmup', total=warmup_size):
            ModelBenchmarker._execute_batch_for_time(batch, model, self._device)

        results = BenchmarkResults()

        for _ in tqdm(range(test_size), desc='test', total=test_size):
            model_sec, e2e_sec, batch_size = ModelBenchmarker._execute_batch_for_time(batch, model, self._device)
            results.add_result(model_sec, e2e_sec, batch_size)

        return results

    @staticmethod
    def _convert_precisions(model: Module, batch: Union[Tensor, List[Tensor], Tuple[Tensor, ...]], full: bool):
        if full:
            if isinstance(batch, Tensor):
                batch.float()
            else:
                for tens in batch:
                    tens.float()

            model.float()

            return

        if isinstance(batch, Tensor):
            batch.half()
        else:
            for tens in batch:
                tens.half()

        model.half()

    @staticmethod
    def _execute_batch_for_time(batch: Union[Tensor, List[Tensor], Tuple[Tensor, ...]],
                                model: Module, device: str) -> Tuple[float, float, int]:
        with torch.no_grad():
            batch = ModelBenchmarker._batch_to_device(batch, 'cpu')

            if 'cuda' in device:
                torch.cuda.synchronize()

            e2e_start = time.time()

            x_tens = ModelBenchmarker._batch_to_device(batch, device)

            if 'cuda' in device:
                torch.cuda.synchronize()

            model_start = time.time()
            y_pred = model(*x_tens)

            if not isinstance(y_pred, Tensor):
                # returning multiple outputs (like logits and classes)
                # assume first index is supposed to be the logits
                y_pred = y_pred[0]

            if 'cuda' in device:
                torch.cuda.synchronize()

            model_end = time.time()
            y_pred_local = y_pred.to('cpu')

            if 'cuda' in device:
                torch.cuda.synchronize()

            e2e_end = time.time()
            batch_size = y_pred_local.shape[0]

            del x_tens
            del y_pred
            del y_pred_local

            if 'cuda' in device:
                torch.cuda.synchronize()

            return model_end - model_start, e2e_end - e2e_start, batch_size

    @staticmethod
    def _batch_to_device(batch: Union[Tensor, List[Tensor], Tuple[Tensor, ...]],
                         device: str) -> Union[Tensor, List[Tensor], Tuple[Tensor, ...]]:
        if isinstance(batch, Tensor):
            batch = batch.to(device)
        else:
            batch = [tens.to(device) for tens in batch]

        return batch
