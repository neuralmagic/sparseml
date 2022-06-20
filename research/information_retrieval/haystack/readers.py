"""
The Reader, also known as Open-Domain QA systems in Machine Learning speak,
is the core component that enables Haystack to find the answers that you need.

https://haystack.deepset.ai/components/reader
"""
from typing import List, Optional, Union, Dict

import logging
import torch
import numpy

from deepsparse.transformers import pipeline

from haystack.nodes.reader.base import BaseReader
from haystack.nodes.retriever.base import BaseRetriever
from haystack.document_stores import BaseDocumentStore
from haystack.nodes.retriever._embedding_encoder import _EMBEDDING_ENCODERS, _BaseEmbeddingEncoder
from haystack.schema import Document, Answer
from haystack.modeling.utils import initialize_device_settings

logger = logging.getLogger(__name__)

class EmbeddingRetriever(BaseRetriever):
    def __init__(
        self,
        document_store: BaseDocumentStore,
        embedding_model: str,
        model_version: Optional[str] = None,
        use_gpu: bool = True,
        batch_size: int = 32,
        max_seq_len: int = 512,
        model_format: str = "farm",
        pooling_strategy: str = "reduce_mean",
        emb_extraction_layer: int = -1,
        top_k: int = 10,
        progress_bar: bool = True,
        devices: Optional[List[Union[str, torch.device]]] = None,
        use_auth_token: Optional[Union[str, bool]] = None,
        scale_score: bool = True,
    ):
        """
        :param document_store: An instance of DocumentStore from which to retrieve documents.
        :param embedding_model: Local path or name of model in Hugging Face's model hub such as ``'sentence-transformers/all-MiniLM-L6-v2'``
        :param model_version: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
        :param use_gpu: Whether to use all available GPUs or the CPU. Falls back on CPU if no GPU is available.
        :param batch_size: Number of documents to encode at once.
        :param max_seq_len: Longest length of each document sequence. Maximum number of tokens for the document text. Longer ones will be cut down.
        :param model_format: Name of framework that was used for saving the model. Options:

                             - ``'farm'``
                             - ``'transformers'``
                             - ``'sentence_transformers'``
        :param pooling_strategy: Strategy for combining the embeddings from the model (for farm / transformers models only).
                                 Options:

                                 - ``'cls_token'`` (sentence vector)
                                 - ``'reduce_mean'`` (sentence vector)
                                 - ``'reduce_max'`` (sentence vector)
                                 - ``'per_token'`` (individual token vectors)
        :param emb_extraction_layer: Number of layer from which the embeddings shall be extracted (for farm / transformers models only).
                                     Default: -1 (very last layer).
        :param top_k: How many documents to return per query.
        :param progress_bar: If true displays progress bar during embedding.
        :param devices: List of GPU (or CPU) devices, to limit inference to certain GPUs and not use all available ones
                        These strings will be converted into pytorch devices, so use the string notation described here:
                        https://pytorch.org/docs/stable/tensor_attributes.html?highlight=torch%20device#torch.torch.device
                        (e.g. ["cuda:0"]). Note: As multi-GPU training is currently not implemented for EmbeddingRetriever,
                        training will only use the first device provided in this list.
        :param use_auth_token:  API token used to download private models from Huggingface. If this parameter is set to `True`,
                                the local token will be used, which must be previously created via `transformer-cli login`.
                                Additional information can be found here https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained
        :param scale_score: Whether to scale the similarity score to the unit interval (range of [0,1]).
                            If true (default) similarity scores (e.g. cosine or dot_product) which naturally have a different value range will be scaled to a range of [0,1], where 1 means extremely relevant.
                            Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.
        """
        super().__init__()

        if devices is not None:
            self.devices = [torch.device(device) for device in devices]
        else:
            self.devices, _ = initialize_device_settings(use_cuda=use_gpu, multi_gpu=True)

        if batch_size < len(self.devices):
            logger.warning("Batch size is less than the number of devices. All gpus will not be utilized.")

        self.document_store = document_store
        self.embedding_model = embedding_model
        self.model_format = model_format
        self.model_version = model_version
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.pooling_strategy = pooling_strategy
        self.emb_extraction_layer = emb_extraction_layer
        self.top_k = top_k
        self.progress_bar = progress_bar
        self.use_auth_token = use_auth_token
        self.scale_score = scale_score

        logger.info(f"Init retriever using embeddings of model {embedding_model}")

        if model_format not in _EMBEDDING_ENCODERS.keys():
            raise ValueError(f"Unknown retriever embedding model format {model_format}")

        if self.embedding_model.startswith("sentence-transformers") and self.model_format != "sentence_transformers":
            logger.warning(
                f"You seem to be using a Sentence Transformer embedding model but 'model_format' is set to '{self.model_format}'."
                f" You may need to set 'model_format='sentence_transformers' to ensure correct loading of model."
            )

        self.embedding_encoder = _EMBEDDING_ENCODERS[model_format](self)

    def retrieve(
        self,
        query: str,
        filters: dict = None,
        top_k: Optional[int] = None,
        index: str = None,
        headers: Optional[Dict[str, str]] = None,
        scale_score: bool = None,
    ) -> List[Document]:
        """
        Scan through documents in DocumentStore and return a small number documents
        that are most relevant to the query.

        :param query: The query
        :param filters: A dictionary where the keys specify a metadata field and the value is a list of accepted values for that field
        :param top_k: How many documents to return per query.
        :param index: The name of the index in the DocumentStore from which to retrieve documents
        :param scale_score: Whether to scale the similarity score to the unit interval (range of [0,1]).
                                           If true similarity scores (e.g. cosine or dot_product) which naturally have a different value range will be scaled to a range of [0,1], where 1 means extremely relevant.
                                           Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.
        """
        if top_k is None:
            top_k = self.top_k
        if index is None:
            index = self.document_store.index
        if scale_score is None:
            scale_score = self.scale_score
        query_emb = self.embed_queries(texts=[query])
        documents = self.document_store.query_by_embedding(
            query_emb=query_emb[0], filters=filters, top_k=top_k, index=index, headers=headers, scale_score=scale_score
        )
        return documents

    def embed_queries(self, texts: List[str]) -> List[numpy.ndarray]:
        """
        Create embeddings for a list of queries.

        :param texts: Queries to embed
        :return: Embeddings, one per input queries
        """
        # for backward compatibility: cast pure str input
        if isinstance(texts, str):
            texts = [texts]
        assert isinstance(texts, list), "Expecting a list of texts, i.e. create_embeddings(texts=['text1',...])"
        return self.embedding_encoder.embed_queries(texts)

    def embed_documents(self, docs: List[Document]) -> List[numpy.ndarray]:
        """
        Create embeddings for a list of documents.

        :param docs: List of documents to embed
        :return: Embeddings, one per input document
        """
        docs = self._linearize_tables(docs)
        return self.embedding_encoder.embed_documents(docs)

    def _linearize_tables(self, docs: List[Document]) -> List[Document]:
        """
        Turns table documents into text documents by representing the table in csv format.
        This allows us to use text embedding models for table retrieval.

        :param docs: List of documents to linearize. If the document is not a table, it is returned as is.
        :return: List of documents with linearized tables or original documents if they are not tables.
        """
        linearized_docs = []
        for doc in docs:
            if doc.content_type == "table":
                doc = deepcopy(doc)
                if isinstance(doc.content, pd.DataFrame):
                    doc.content = doc.content.to_csv(index=False)
                else:
                    raise HaystackError("Documents of type 'table' need to have a pd.DataFrame as content field")
            linearized_docs.append(doc)
        return linearized_docs


class _DeepSparseEmbeddingEncoder(_BaseEmbeddingEncoder):
    def __init__(self, retriever: "EmbeddingRetriever"):
        # TODO: Check imports work
        try:
            from sentence_transformers import SentenceTransformer
        except (ImportError, ModuleNotFoundError) as ie:
            from haystack.utils.import_utils import _optional_component_not_installed

            _optional_component_not_installed(__name__, "sentence", ie)


        # Create model
        self.embedding_model = SentenceTransformer(
            retriever.embedding_model_stub, device=str(retriever.devices[0]), use_auth_token=retriever.use_auth_token
        )
        #self.embedding_model = pipeline.create(
        #    retriever.embedding_model_stub, device=str(retriever.devices[0]), use_auth_token=retriever.use_auth_token
        #)

        self.batch_size = retriever.batch_size
        self.embedding_model.max_seq_length = retriever.max_seq_len
        self.show_progress_bar = retriever.progress_bar
        document_store = retriever.document_store
        if document_store.similarity != "cosine":
            logger.warning(
                f"You are using a Sentence Transformer with the {document_store.similarity} function. "
                f"We recommend using cosine instead. "
                f"This can be set when initializing the DocumentStore"
            )

    def embed(self, texts: Union[List[List[str]], List[str], str]) -> List[numpy.ndarray]:
        # texts can be a list of strings or a list of [title, text]
        # get back list of numpy embedding vectors
        print(f"embed {texts}")
        emb = self.embedding_model.encode(texts, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar)
        emb = [r for r in emb]
        return emb

    def embed_queries(self, texts: List[str]) -> List[numpy.ndarray]:
        return self.embed(texts)

    def embed_documents(self, docs: List[Document]) -> List[numpy.ndarray]:
        passages = [[d.meta["name"] if d.meta and "name" in d.meta else "", d.content] for d in docs]  # type: ignore
        return self.embed(passages)


class DeepSparseEmbeddingRetriever(EmbeddingRetriever):
    def __init__(
        self,
        document_store: BaseDocumentStore,
        embedding_model_stub: str,
        model_version: Optional[str] = None,
        use_gpu: bool = True,
        batch_size: int = 32,
        max_seq_len: int = 512,
        model_format: str = "farm",
        pooling_strategy: str = "reduce_mean",
        emb_extraction_layer: int = -1,
        top_k: int = 10,
        progress_bar: bool = True,
        devices: Optional[List[Union[str, torch.device]]] = None,
        use_auth_token: Optional[Union[str, bool]] = None,
        scale_score: bool = True,
    ):
        super(BaseRetriever).__init__()

        if devices is not None:
            self.devices = [torch.device(device) for device in devices]
        else:
            self.devices, _ = initialize_device_settings(use_cuda=use_gpu, multi_gpu=True)

        if batch_size < len(self.devices):
            logger.warning("Batch size is less than the number of devices. All gpus will not be utilized.")

        self.document_store = document_store
        self.embedding_model_stub = embedding_model_stub
        self.model_format = model_format
        self.model_version = model_version
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.pooling_strategy = pooling_strategy
        self.emb_extraction_layer = emb_extraction_layer
        self.top_k = top_k
        self.progress_bar = progress_bar
        self.use_auth_token = use_auth_token
        self.scale_score = scale_score

        logger.info(f"Init retriever using embeddings of model stub {embedding_model_stub}")

        # TODO: Throw value error if unknown model
        # TODO: Give warning if using wrong kind of model

        self.embedding_encoder = _DeepSparseEmbeddingEncoder(self)
