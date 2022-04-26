"""
To run:
1. pip install deepsparse
2. pip install -U jina
3. python minimal_example.py
"""

from docarray import Document, DocumentArray
from jina import Flow
from executors import DeepSparseExecutor
import time

query, context = "What's my name?", "My name is Mario!"
# JinaAI uses DocumentArray as its native data structure
docs = DocumentArray([Document(text=query),
                      Document(text=context)
                      ])

# Pass the SparseZoo model
model_stub = "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned_3layers-aggressive_90"

# Create a new Flow that includes our custom DeepSparseExecutor
flow = (Flow().add(uses=DeepSparseExecutor,
                   uses_with={'model_stub': model_stub},
                   name="deepsparse_executor"))

# Open Flow as context manager
time1 = time.time()
with flow:
    answer = flow.post(on='/question_answering', inputs=docs)
time_delta = time.time() - time1

print(f"Input question: {query}, input context: {context}")
print(f"Response: {answer.contents[0]} (took {time_delta}s)")
