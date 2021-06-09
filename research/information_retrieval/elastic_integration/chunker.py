import re
import logging
from typing import List
from transformers import DPRReaderTokenizer

class DocumentChunker:
    def __init__(self, tokenizer, max_tokens = 512, max_query_tokens=30, document_chunks=5):
        self.tokenizer = tokenizer #DPRReaderTokenizer.from_pretrained('facebook/dpr-reader-single-nq-base')
        self.max_tokens = max_tokens
        self.document_chunks = document_chunks
        self.max_query_tokens = max_query_tokens
        self.max_doc_tokens = self.max_tokens - self.max_query_tokens -2 # [SEP] and [CLS]
        
    def chunk_doc(self, document ):
        #This is where logic to split documents goes. We keep the first chunk, last chunk, and a classifed best chunk
        chunks = []
        return chunks
