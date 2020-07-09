import os
from typing import Dict

import yaml

CONFIG_FILE = "config.yaml"

__all__ = ["ProjectConfig"]

ID_KEY = "projectId"


class ProjectConfig(dict):
    def __init__(self, project_path: str):
        self._config_path = os.path.join(project_path, CONFIG_FILE)
        self._config_settings = {}
        if os.path.exists(self._config_path):
            with open(self._config_path) as yml_file:
                self._config_settings = yaml.safe_load(yml_file)
        else:
            self._config_settings = {ID_KEY: project_path.split("/")[-1]}
            self.save()

    def overwrite(self, config_settings: Dict):
        self._config_settings = config_settings
        self.save()

    def write(self, config_settings: Dict, save: bool = True):
        for key in config_settings:
            self.set_setting(key, config_settings[key])

        if save:
            self.save()

    def save(self):
        with open(self._config_path, "w+") as yml_file:
            yml_file.write(self.yaml)

    @property
    def yaml(self) -> str:
        return yaml.dump(self._config_settings)

    @property
    def config_settings(self) -> Dict:
        return self._config_settings

    def set_setting(self, key, value):
        self._config_settings[key] = value

    def get_setting(self, key):
        try:
            return self._config_settings[key]
        except KeyError:
            return None

    @property
    def id(self) -> str:
        return self.get_setting(ID_KEY)
