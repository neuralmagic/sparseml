import json
import logging
import os
from typing import Dict, Iterable, List
import uuid
import shutil

from neuralmagicML.onnx.project.config import ProjectConfig
from neuralmagicML.onnx.project.models import RecalModel

from neuralmagicML.onnx.utils import Dataloader, get_path, allowed_file

__all__ = ["RecalProject"]

INPUTS = "inputs"

DEFAULT_INPUT_FOLDER = "_sample-inputs"


class RecalProject:
    def __init__(self, path: str):
        self._model = RecalModel(os.path.join(path, "model.onnx"))
        self._path = path
        self._config = ProjectConfig(self._path)

        os.makedirs(self.perf_folder, exist_ok=True)
        os.makedirs(self.loss_folder, exist_ok=True)

    @staticmethod
    def register_project(
        model_path: str, config_settings: Dict, project_root: str = None
    ):
        project_path = None
        try:
            project_id = str(uuid.uuid4())
            project_path = get_path(project_id, project_root=project_root)
            model_path = os.path.expanduser(model_path)
            logging.info(f"Saving {model_path} to path {project_path}")

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Path {model_path} does not exist")
            elif not os.path.isfile(model_path) or not allowed_file(model_path):
                raise Exception(f"Path {model_path} is not a valid file")

            os.makedirs(project_path, exist_ok=True)
            config_settings["projectId"] = project_id
            project_config = ProjectConfig(project_path)
            project_config.write(config_settings)

            target_file = os.path.join(project_path, "model.onnx")
            shutil.copy(model_path, target_file)

            return project_config
        except Exception as e:
            if project_path is not None and os.path.exists(project_path):
                shutil.rmtree(project_path)
            logging.exception(e)
            raise e

    def get_input_loader(self, batch_size: int):
        input_folder = (
            self.config.get_setting(INPUTS)
            if self.config.get_setting(INPUTS)
            else os.path.join(self._path, DEFAULT_INPUT_FOLDER)
        )
        if os.path.exists(input_folder):
            return Dataloader(
                os.path.join(input_folder, "*.npz"), batch_size=batch_size
            )
        else:
            logging.warning(f"No input folder found for {self._path}")
            return []

    @property
    def model(self):
        return self._model

    @property
    def config(self) -> ProjectConfig:
        return self._config

    @property
    def prunable_layers(self) -> List[Dict]:
        return self._model.prunable_layers

    @property
    def sparse_analysis_loss_approx(self) -> List[Dict]:
        return self._model.sparse_analysis_loss_approx

    @property
    def sparse_analysis_perf_approx(self) -> List[Dict]:
        return self._model.sparse_analysis_perf_approx

    @property
    def sparse_analysis_perf(self) -> List[Dict]:
        return self.get_sparse_analysis_perf()

    @property
    def sparse_analysis_loss(self) -> List[Dict]:
        return self.get_sparse_analysis_loss()

    @property
    def perf_folder(self) -> str:
        return os.path.join(self._path, "sparse-analysis/perf")

    def perf_file_path(self, perf_file: str):
        if perf_file.endswith(".json"):
            return os.path.join(self.perf_folder, perf_file)
        else:
            return os.path.join(self.perf_folder, f"{perf_file}.json")

    @property
    def loss_folder(self) -> str:
        return os.path.join(self._path, "sparse-analysis/loss")

    def loss_file_path(self, loss_file: str):
        if loss_file.endswith(".json"):
            return os.path.join(self.loss_folder, loss_file)
        else:
            return os.path.join(self.loss_folder, f"{loss_file}.json")

    @property
    def onnx_path(self) -> str:
        return os.path.join(self._path, "model.onnx")

    @property
    def path(self) -> str:
        return self._path

    def run_sparse_analysis_perf(
        self,
        perf_file: str,
        batch_size: int = 1,
        sparsity_levels: List[float] = None,
        optimization_level=0,
        num_cores=-1,
        num_warmup_iterations=5,
        num_iterations=30,
    ) -> List[Dict]:
        inputs = self.get_input_loader(batch_size)
        perf_path = self.perf_file_path(perf_file)
        self.model.run_sparse_analysis_perf(
            perf_path,
            inputs,
            sparsity_levels=sparsity_levels,
            optimization_level=optimization_level,
            num_cores=num_cores,
            num_warmup_iterations=num_warmup_iterations,
            num_iterations=num_iterations,
        )
        return self.get_sparse_analysis_perf(perf_file)

    def write_sparse_analysis_perf(self, perf_file: str, content: dict) -> List[Dict]:
        if perf_file == "approx":
            raise Exception("Cannot name perf file: approx")
        perf_path = self.perf_file_path(perf_file)
        with open(perf_path, "w+") as json_data:
            json_data.write(json.dumps(content))

        return self.get_sparse_analysis_perf(perf_file)

    def write_sparse_analysis_loss(self, loss_file: str, content: dict) -> List[Dict]:
        if loss_file == "approx":
            raise Exception("Cannot name loss file: approx")
        loss_path = self.loss_file_path(loss_file)
        with open(loss_path, "w+") as json_data:
            json_data.write(json.dumps(content))

        return self.get_sparse_analysis_loss(loss_file)

    def get_sparse_analysis_perf(self, perf_file: str = None) -> List[Dict]:
        if perf_file is None:
            perf_files = sorted(
                [
                    os.path.join(self.perf_folder, perf_json)
                    for perf_json in os.listdir(self.perf_folder)
                    if ".json" in perf_json
                ],
                key=lambda perf_file: os.path.getctime(perf_file),
            )
            if len(perf_files) > 0:
                with open(perf_files[0]) as json_data:
                    return json.load(json_data)
            else:
                raise FileNotFoundError(f"No perf files exist for {self.config.id}")
        else:
            with open(os.path.join(self.perf_folder, f"{perf_file}.json")) as json_data:
                return json.load(json_data)

    def run_sparse_analysis_loss(
        self,
        loss_file: str,
        batch_size: int = 1,
        sparsity_levels: List[float] = None,
        samples_per_measurement: int = 5,
    ) -> List[Dict]:
        inputs = self.get_input_loader(batch_size)
        loss_path = self.loss_file_path(loss_file)
        self.model.one_shot_ks_loss_sensitivity(
            loss_path,
            inputs,
            sparsity_levels=sparsity_levels,
            samples_per_measurement=samples_per_measurement,
        )
        return self.get_sparse_analysis_loss(loss_file)

    def get_sparse_analysis_loss(self, loss_file: str = None) -> List[Dict]:
        if loss_file is None:
            loss_files = sorted(
                [
                    os.path.join(self.loss_folder, loss_json)
                    for loss_json in os.listdir(self.loss_folder)
                    if ".json" in loss_json
                ],
                key=lambda loss_file: os.path.getctime(loss_file),
            )
            if len(loss_files) > 0:
                with open(loss_files[0]) as json_data:
                    return json.load(json_data)
            else:
                raise FileNotFoundError(f"No loss files exist for {self.config.id}")
        else:
            with open(os.path.join(self.loss_folder, f"{loss_file}.json")) as json_data:
                return json.load(json_data)
