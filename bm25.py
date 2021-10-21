""" Implementation of OKapi BM25 with sklearn's TfidfVectorizer
Distributed as CC-0 (https://creativecommons.org/publicdomain/zero/1.0/)
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse

class BM25(object):
    def __init__(self, tokenizer, b=0.75, k1=1.6):
        self.vectorizer = TfidfVectorizer(
            tokenizer=tokenizer,
            ngram_range=(1, 2),
            max_features=50000,
        )
        self.b = b
        self.k1 = k1

    def fit(self, X):
        """ Fit IDF to documents X """
        self.vectorizer.fit(X)
        self.y = super(TfidfVectorizer, self.vectorizer).transform(X)
        self.avdl = self.y.sum(1).mean()

    def transform(self, q, X):
        """ Calculate BM25 between query q and documents X """
        b, k1, avdl = self.b, self.k1, self.avdl

        # apply CountVectorizer
        X = self.y
        len_X = X.sum(1).A1
        q, = super(TfidfVectorizer, self.vectorizer).transform([q])
        assert sparse.isspmatrix_csr(q)

        # convert to csc for better column slicing
        X = X.tocsc()[:, q.indices]
        denom = X + (k1 * (1 - b + b * len_X / avdl))[:, None]
        # idf(t) = log [ n / df(t) ] + 1 in sklearn, so it need to be coneverted
        # to idf(t) = log [ n / df(t) ] with minus 1
        idf = self.vectorizer._tfidf.idf_[None, q.indices] - 1.
        numer = X.multiply(np.broadcast_to(idf, X.shape)) * (k1 + 1)                                                          
        return (numer / denom).sum(1).A1
