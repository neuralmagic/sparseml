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
import re
import logging
from typing import List
from transformers import DPRReaderTokenizer

class DocumentChunker:
    def __init__(self, tokenizer, max_tokens = 512, max_query_tokens=30, document_chunks=5):
        self.tokenizer = tokenizer #DPRReaderTokenizer.from_pretrained('facebook/dpr-reader-single-nq-base')
        self.max_tokens = max_tokens
        self.document_chunks = document_chunks
        self.max_query_tokens = max_query_tokens
        self.max_doc_tokens = self.max_tokens - self.max_query_tokens -2 # [SEP] and [CLS]

    def chunk_doc(self, document ):
        #This is where logic to split documents goes. We keep the first chunk, last chunk, and a classifed best chunk
        chunks = []
        return chunks
