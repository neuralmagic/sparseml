# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import logging
from contextlib import suppress
from pathlib import Path
from typing import Callable, Dict

import yaml

from sparsezoo.evaluation import EvaluationRegistry
from sparsezoo.evaluation.results import Result
from sparsezoo.utils.registry import standardize_lookup_name


__all__ = ["SparseMLEvaluationRegistry", "collect_integrations"]

_LOGGER = logging.getLogger(__name__)

INTEGRATION_CONFIG_NAME: str = "integrations_config.yaml"
INTEGRATION_CONFIG_PATH: Path = Path(__file__).parent / INTEGRATION_CONFIG_NAME


class SparseMLEvaluationRegistry(EvaluationRegistry):
    """
    This class is used to register and retrieve evaluation integrations for
    SparseML. It is a subclass of the SparseZoo EvaluationRegistry class.
    """

    @classmethod
    def resolve(cls, name: str, *args, **kwargs) -> Callable[..., Result]:
        """
        Resolve an evaluation integration by name.

        :param name: The name of the evaluation integration to resolve
        :param args: The arguments to pass to the evaluation integration
        :param kwargs: The keyword arguments to pass to the evaluation integration
        :return: The evaluation integration associated with the name
        """

        collect_integrations(
            name=name,
            integration_config_path=kwargs.get(
                "integration_config_path", INTEGRATION_CONFIG_PATH
            ),
        )
        return cls.get_value_from_registry(name=name)


def collect_integrations(
    name: str, integration_config_path: Path = INTEGRATION_CONFIG_PATH
):
    """
    This function is used to collect integrations based on name, this method
    is responsible for triggering the registration with the SparseML Evaluation
    registry.

    This is done by importing the module associated with each integration.
    This function is called automatically when the registry is accessed.
    The specific integration(s) `callable` must be decorated with the
    `@SparseMLEvaluationRegistry.register` decorator.

    :param name: The name of the integration to collect, is case insentitive
    :param integration_config_path: The path to the integrations config file
    """
    with suppress(KeyError):
        # exit early if the integration is already registered
        SparseMLEvaluationRegistry.get_value_from_registry(name)
        _LOGGER.info(f"Integration {name} already registered")
        return

    integrations = _standardize_integration_dict(
        integrations_location_dict=_load_yaml(path=integration_config_path),
        integration_config_path=integration_config_path,
    )

    name = standardize_lookup_name(name)

    if name in integrations:
        location = integrations[name]
        _LOGGER.debug(f"Auto collecting {name} integration for eval from {location}")
        try:
            spec = importlib.util.spec_from_file_location(
                f"eval_plugin_{name}", location
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            _LOGGER.info(f"Auto collected {name} integration for eval")
        except ImportError as import_error:
            _LOGGER.warning(f"Collection of {name} integration for eval failed")
            raise import_error
    else:
        raise ValueError(f"No registered integrations found for the given name {name}")


def _load_yaml(path: Path):
    """
    Load a yaml file from the given path.

    :param path: The path to the yaml file
    :return: The loaded yaml file
    """
    with path.open("r") as file:
        return yaml.safe_load(file)


def _standardize_integration_dict(
    integrations_location_dict: Dict[str, str],
    integration_config_path: Path = INTEGRATION_CONFIG_PATH,
) -> Dict[str, str]:
    """
    Standardize the names of the integrations in the given dictionary.

    :param integrations_location_dict: Dictionary of integration names to
        their locations
    :param integration_config_path: The path to the integrations config file
        (Used to resolve relative paths to absolute paths)
    :return: A copy of the dictionary with the standardized integration names
        and the resolved absolute paths to their locations
    """
    return {
        standardize_lookup_name(name): _resolve_relative_path(
            relative_path=location, relative_to=integration_config_path
        )
        for name, location in integrations_location_dict.items()
    }


def _resolve_relative_path(
    relative_path: Path, relative_to: Path = INTEGRATION_CONFIG_PATH
) -> Path:
    """
    Resolve the given path to an absolute path.

    :param relative_path: The path to resolve
    :param relative_to: The path to resolve the given path relative to, defaults
        to the integrations config path
    :return: The resolved absolute path
    """
    absolute_path = (relative_to.resolve().parent / relative_path).resolve()
    return absolute_path
