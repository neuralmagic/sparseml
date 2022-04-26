"""
Executor is a self-contained component and performs a group of tasks on a DocumentArray.
This is the component, which in our integration should wrap around DeepSpare pipeline.

Source: https://docs.jina.ai/fundamentals/executor/
"""

from deepsparse.transformers import pipeline
from docarray import Document, DocumentArray
from jina import Executor, requests


class DeepSparseExecutor(Executor):
    """
    Minimal integration requires a
        - constructor
        - at least one function decorated by @request (endpoint)
    """
    def __init__(self, model_stub, **kwargs):
        self.pipeline = pipeline(
            task="question-answering",
            model_path=model_stub
        )
        super().__init__(**kwargs)

    @requests(on='/question_answering')
    def run(self, docs, **kwargs):
        question = docs[0].content
        context = docs[1].content

        answer = self.pipeline(question=question, context=context)
        """
        Endpoint function returns one of three:
            - if return None, will return (potentially mutated) docs variable
            - DocumentArray
            - dict (will be passed along with the potentially mutated docs variable)
            """
        return DocumentArray(Document(text=answer['answer']))
