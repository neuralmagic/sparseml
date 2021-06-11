# neuralmagic: no copyright
# flake8: noqa
# fmt: off
# isort: skip_file
#!/usr/bin/env python
# coding=utf-8
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
import json

class DenseDocument:
    title: str
    body: str
    chunks: list
    def __init__(self, title: str, body:str, chunks:list):
        self.title = title
        self.body = body
        self.chunks = chunks

    def to_dict(self):
        return {'title': self.title, 'body': self.body, 'chunks': self.chunks}

    def __repr__(self):
        pretty = json.dumps(self.to_dict())
        return pretty
