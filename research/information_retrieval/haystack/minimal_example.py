"""
To run:
1. pip install deepsparse
2. pip install farm-haystack
3. python minimal_example.py

Minimal example from: https://github.com/deepset-ai/haystack/blob/master/tutorials/Tutorial3_Basic_QA_Pipeline_without_Elasticsearch.ipynb
https://github.com/deepset-ai/haystack/issues/248#issuecomment-661977237
https://haystack.deepset.ai/components/reader

"""
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import TfidfRetriever
from haystack.pipelines import ExtractiveQAPipeline
from haystack.utils import print_answers

from readers import DeepSparseReader

# Pass the SparseZoo model
model_stub = "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned_3layers-aggressive_90"

query, contexts = "Who's brother is holding a sword?", ["Mario has red pants", "Wario wields a fire sword", "Wario is Mario's brother"]

# Haystack finds answer to query within the documents stored in a DocumentStore
# e.g. ElasticsearchDocumentStore, SQLDocumentStore etc.
# Let's just use our memory as a store.
document_store = InMemoryDocumentStore()
docs = [
    {
        'content': contexts[0],
        'meta': {'name': '...'}
    },
    {
        'content': contexts[1],
        'meta': {'name': '...'}
    },
    {
        'content': contexts[2],
        'meta': {'name': '...'}
    }
]
document_store.write_documents(docs)

# Initializing a Retriever (inaccurate but fast)
# Retrievers help to narrow down the scope for the Reader to smaller units of text
# where a given question could be answered.
retriever = TfidfRetriever(document_store=document_store)

# Initializing a Reader (accurate but slow)
# A Reader scans the texts returned by retrievers in detail and extracts the k best answers.
reader = DeepSparseReader(model_stub=model_stub)

# 4. Creating a Pipeline
# With a Haystack Pipeline you can stick together your building blocks to a search pipeline.
# Under the hood, Pipelines are Directed Acyclic Graphs (DAGs) that you can easily customize for your own use cases.
pipe = ExtractiveQAPipeline(reader, retriever)

prediction = pipe.run(
    query=query, params={"Retriever": {"top_k": 2}, "Reader": {"top_k": 1}}
)
print_answers(prediction, details="minimum")
