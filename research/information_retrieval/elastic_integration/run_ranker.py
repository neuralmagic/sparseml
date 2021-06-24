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

"""
Example script for integrating spare ranking models with Elasticsearch.
##########
Command help:
usage: run_ranker.py [-h] \

##########
Example command for generating a Dense Elastic Search compatible index using sparse Bi-encoder Model:
python research/information_retrieval/run_ranker.py \
    
"""

    context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(
        'facebook/dpr-ctx_encoder-single-nq-base')
    context_model = DPRContextEncoder.from_pretrained(
        'facebook/dpr-ctx_encoder-single-nq-base', return_dict=True)
    question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
        'facebook/dpr-question_encoder-single-nq-base')
    question_model = DPRQuestionEncoder.from_pretrained(
        'facebook/dpr-question_encoder-single-nq-base', return_dict=True)