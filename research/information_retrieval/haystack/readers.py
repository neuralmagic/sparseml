"""
The Reader, also known as Open-Domain QA systems in Machine Learning speak,
is the core component that enables Haystack to find the answers that you need.

https://haystack.deepset.ai/components/reader
"""
from deepsparse.transformers import pipeline
from haystack.nodes.reader.base import BaseReader
from haystack.schema import Document, Answer
from typing import List, Optional


class DeepSparseReader(BaseReader):
    """
    Minimal integration requires a
        - constructor
        - predict and predict_batch functions
    """

    def __init__(self, model_stub):
        self.pipeline = pipeline(
            task="question-answering",
            model_path=model_stub,

        )

    def predict(self, query, documents, top_k):
        pred = self.pipeline(question=query, context=[document.content for document in documents], top_k=top_k)
        answers = [Answer(answer=pred["answer"])]
        results = {"query": query, "answers": answers}
        return results

    def predict_batch(self, query_doc_list, top_k, batch_size=None):
        raise NotImplementedError("Batch prediction not yet available in DeepSparseReader.")
