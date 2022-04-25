from docarray import Document, DocumentArray
from jina import Flow

from executors import DeepSparseExecutor

# Create a document with a query and context
query, context = "What's my name?", "My name is Mario!"
docs = DocumentArray([Document(text=query),
                      Document(text=context)
                      ])
# Pass the SparseZoo model
model_stub = "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned_3layers-aggressive_90"

# Create a new Flow that includes our custom Executor
flow = (Flow().add(uses=DeepSparseExecutor,
                   uses_with={'model_stub': model_stub},
                   name="executor",
                   install_requirements=True))

# Open Flow as context manager
with flow:
    answer = flow.post(on='/question_answering', inputs=docs)

print(f"Input question: {query}, input context: {context}")
print(f"Response: {answer.contents[0]}")
