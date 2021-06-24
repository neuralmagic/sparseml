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
import torch
import faiss
from tqdm import tqdm
from typing import List
from elasticsearch import Elasticsearch
from transformers import (DPRContextEncoder, DPRContextEncoderTokenizer,
                          DPRQuestionEncoder, DPRQuestionEncoderTokenizer)
from chunker import DocumentChunker
from dense_document import DenseDocument


class DenseIndex():
    def __init__(self, documents, context_tokenizer, context_model, query_tokenizer, query_model, index_name='dense-index', dimension=768):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.index_name = index_name
        self.faiss_index = faiss.IndexFlatIP(self.dimension)
        self.prep_index()
        self.generate_index(documents)

    def prep_index(self):
        self.es = Elasticsearch()
        if self.es.indices.exists(self.index_name):
            logging.warning(f'Deleting old index for {self.index_name}.')
            self.es.indices.delete(self.index_name)
        self.es.indices.create(index=self.index_name)

    def generate_index(self, documents, max_passages: int = 5): #Works for passages because passages dataset will only have 1 chunk
        self.documents = documents
        self.doc_bodies = [doc.body for doc in self.documents]
        self.passage_to_doc = {} #pasage_id to doc_id
        self.passages = []
        doc_id = 0
        passage_id = 0 
        for doc_counter, doc_body in tqdm(enumerate(self.doc_bodies),total=len(self.doc_bodies)):
            self.es.create(self.index_name, id=doc_id, body={'document': doc_body})
            passages = self.chunk_document(doc_body)
            for i in range(min(len(passages),max_passages)): #NEED to add a chunking strategy first P, Last P, Best P 
                self.passages.append(passages[i])
                input_ids = self.context_tokenizer(passages[i], return_tensors='pt')['input_ids']
                self.faiss_index.add(self.context_model(input_ids).pooler_output.detach().numpy())
                self.passage_to_doc[passage_id] = doc_id
                passage_id += 1
            doc_id += 1

    def dense_search(self, query: str, k: int = 10):
        input_ids = self.query_tokenizer(query, return_tensors='pt')['input_ids']
        vec_dists, vec_ids = self.faiss_index.search(self.query_model(input_ids).pooler_output.detach().numpy(), k=k)
        vec_dists, vec_ids = list(vec_dists[0]), list(vec_ids[0])
        vec_dists= list(map(float, vec_dists))
        results = []
        for dist, passage_id in zip(vec_dists, vec_ids):
            document_id = self.passage_to_doc[passage_id]
            result = {
                'document': self.documents[document_id],
                'document_id': document_id,
                'passage': self.passages[passage_id],
                'passage_id': int(passage_id),
                'faiss_dist': dist
            }
            results.append(result)
        return results

    def sparse_search(self, query: str, k: int = 10):
        body = {
            'size': k,
            'query': {
                'match': {
                    'chunk': query
                }
            }
        }
        results = self.es.search(index=self.index_name, body=body)
        hits = results['hits']['hits']
        return hits

    def hybrid_search(self, query: str, k: int = 10, dense_weight: float = 0.5):
        results_index = {}
        for sparse_result in self.sparse_search(query):
            id, score = sparse_result['_id'], sparse_result['_score']
            id = int(id)
            results_index[id] = {'elastic_score': score}
        for dense_result in self.dense_search(query):
            id, score = dense_result['passage_id'], dense_result['faiss_dist']
            if id in results_index:
                results_index[id]['faiss_dist'] = score
            else:
                results_index[id] = {'faiss_dist': score, 'elastic_score': 0}
        results = []
        for passage_id, scores in results_index.items():
            document_id = self.passage_to_doc[passage_id]
            document = self.documents[document_id]
            doc_profile = document.to_dict()
            result = {
                'document': doc_profile,
                'document_id': document_id,
                'passage': self.passages[passage_id],
                'passage_id': int(passage_id),
                'scores': scores
            }
            results.append(result)
        return results
