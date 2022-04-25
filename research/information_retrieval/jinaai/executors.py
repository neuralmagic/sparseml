from deepsparse import pipeline
from docarray import Document, DocumentArray
from jina import Executor, requests


class DeepSparseExecutor(Executor):
    def __init__(self, model_stub, **kwargs):
        self.pipeline = pipeline(
            task="question-answering",
            model_path=model_stub
        )
        super().__init__(**kwargs)

    @requests(on='/question_answering')
    def run(self, docs, **kwargs):
        pass
        question = docs[0].content
        context = docs[1].content

        answer = self.pipeline(question=question, content=context)
        return DocumentArray(Document(text=question))
