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
from pathlib import Path

import click


__all__ = [
    "export",
    "train",
    "val",
    "download",
]


def export():
    from yolact.export import main as run_export

    run_export()


def train():
    from yolact.train import main as run_train

    run_train()


def val():
    from yolact.eval import main as run_val

    run_val()


@click.command()
@click.option(
    "--test",
    default=False,
    is_flag=True,
    show_default=True,
    help="Download COCO test data",
)
def download(test: bool = False):
    """
    A command line callable to download training/test coco dataset for yolact
    """
    import os as _os
    import subprocess as _subprocess

    try:

        yolact_folder = Path(_os.path.abspath(__file__)).parent.resolve()
        bash_script = "COCO_test.sh" if test else "COCO.sh"
        _subprocess.check_call(
            [
                "bash",
                _os.path.join(yolact_folder, bash_script),
            ]
        )
    except Exception as data_download_exception:
        raise ValueError(
            "Unable to download coco with the "
            f"following exception {data_download_exception}"
        )
