"""
Minimal example from: https://github.com/deepset-ai/haystack/blob/master/tutorials/Tutorial3_Basic_QA_Pipeline_without_Elasticsearch.ipynb
https://github.com/deepset-ai/haystack/issues/248#issuecomment-661977237
https://haystack.deepset.ai/components/reader

"""
import GPUtil
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import TfidfRetriever, TransformersReader
from haystack.pipelines import ExtractiveQAPipeline
from haystack.utils import fetch_archive_from_http, convert_files_to_dicts, clean_wiki_text, print_answers

GPU_available = GPUtil.getAvailable()


def fetch_docs():
    doc_dir = "data/tutorial3"
    s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt3.zip"
    fetch_archive_from_http(url=s3_url, output_dir=doc_dir)
    docs = convert_files_to_dicts(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)
    return docs


def run():
    # 1. Document Preprocessing and Saving in DocumentStore

    # Haystack finds answer to query within the documents stored in a DocumentStore
    # e.g. ElasticsearchDocumentStore, SQLDocumentStore etc.
    # for the sake of this demo, let's just use our memory as a store.
    document_store = InMemoryDocumentStore()
    docs = fetch_docs()
    document_store.write_documents(docs)

    # 2. Initializing a Retriever (inaccurate but fast)

    # Retrievers help to narrow down the scope for the Reader to smaller units of text
    # where a given question could be answered.
    retriever = TfidfRetriever(document_store=document_store)

    # 3. Initializing a Reader (accurate but slow)

    # A Reader scans the texts returned by retrievers in detail and extracts the k best answers.
    # Haystack supports TransformersReader and also propitiatory FARMReader
    reader = TransformersReader(model_name_or_path="distilbert-base-uncased-distilled-squad",
                                tokenizer="distilbert-base-uncased", use_gpu=GPU_available)

    # 4. Creating a Pipeline
    # With a Haystack Pipeline you can stick together your building blocks to a search pipeline.
    # Under the hood, Pipelines are Directed Acyclic Graphs (DAGs) that you can easily customize for your own use cases.

    pipe = ExtractiveQAPipeline(reader, retriever)

    prediction = pipe.run(
        query="How did Ned Stark die?", params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 2}}
    )
    print_answers(prediction, details="minimum")


if __name__ == "__main__":
    run()
