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
from haystack.pipelines import DocumentSearchPipeline
from haystack.utils import print_documents

from haystack.nodes import EmbeddingRetriever
#from readers import EmbeddingRetriever

# Pass the SparseZoo model
model_stub = "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned_3layers-aggressive_90"

query = "Famous artists"
contexts = [
    "Richard Paul Astley (born 6 February 1966) is an English singer, songwriter and "
    "radio personality, who has been active in music for several decades. He gained "
    "worldwide fame in the 1980s, having multiple hits including his signature song "
    "Never Gonna Give You Up, Together Forever and Whenever You Need Somebody, and "
    "returned to music full-time in the 2000s after a 6-year hiatus. Outside his "
    "music career, Astley has occasionally worked as a radio DJ and a podcaster",

    "An ANN is based on a collection of connected units or nodes called artificial "
    "neurons, which loosely model the neurons in a biological brain. Each connection, "
    "like the synapses in a biological brain, can transmit a signal to other neurons. "
    "An artificial neuron receives signals then processes them and can signal neurons "
    "connected to it. The signal at a connection is a real number, and the output of "
    "each neuron is computed by some non-linear function of the sum of its inputs.",

    "Pablo Ruiz Picasso (25 October 1881 – 8 April 1973) was a Spanish painter, "
    "sculptor, printmaker, ceramicist and theatre designer who spent most of his adult "
    "life in France. Regarded as one of the most influential painters of the 20th "
    "century, he is known for co-founding the Cubist movement, the invention of "
    "constructed sculpture, the co-invention of collage, and for the wide "
    "variety of styles that he helped develop and explore"
    ]

# Haystack finds answer to query within the documents stored in a DocumentStore
# e.g. ElasticsearchDocumentStore, SQLDocumentStore etc.
# Let's just use our memory as a store.
document_store = InMemoryDocumentStore(use_gpu=False)
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

embedding_retriever = EmbeddingRetriever(document_store=document_store,
                               embedding_model="deepset/sentence_bert", use_gpu=False)
document_store.update_embeddings(embedding_retriever, update_existing_embeddings=False)

pipe = DocumentSearchPipeline(embedding_retriever)
ret = pipe.run(query=query, params={"Retriever": {"top_k": 2}})
print_documents(ret, max_text_len=200)
