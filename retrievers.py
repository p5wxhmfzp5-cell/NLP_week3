from rank_bm25 import BM25Okapi
import numpy as np

class BM25Retriever:
    def __init__(self, corpus):
        self.tokenized = [doc.split() for doc in corpus]
        self.bm25 = BM25Okapi(self.tokenized)
        self.corpus = corpus

    def retrieve(self, query, k):
        scores = self.bm25.get_scores(query.split())
        top_k = np.argsort(scores)[::-1][:k]
        return [self.corpus[i] for i in top_k]


class DenseRetriever:
    def __init__(self, encoder, index, corpus):
        self.encoder = encoder
        self.index = index
        self.corpus = corpus

    def retrieve(self, query, k):
        vec = self.encoder.encode([query])
        scores, idx = self.index.search(vec, k)
        return [self.corpus[i] for i in idx[0]]