import pickle
import os
import pandas as pd
from elasticsearch import Elasticsearch, helpers
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer
from tqdm import tqdm
from dense_model import BertEncoder
import torch
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler
import numpy as np


class Retrieval:
    def __init__(
        self,
        tokenizer,
        data_path="../data/",
        context_path="/opt/ml/data/preprocess_wiki.json",
    ):
        self.tokenizer = tokenizer
        self.wiki_dataset = pd.read_json(context_path, orient="index")
        if (
            os.path.isfile("wiki_context_id_pair.bin")
            and os.path.isfile("wiki_id_context_pair.bin")
            and os.path.isfile("id_title_pair.bin")
        ):
            with open("wiki_context_id_pair.bin", "rb") as f:
                self.wiki_context_id_dict = pickle.load(f)
            with open("wiki_id_context_pair.bin", "rb") as f:
                self.wiki_id_context_dict = pickle.load(f)
            with open("id_title_pair.bin", "rb") as f:
                self.wiki_id_title_dict = pickle.load(f)
        else:
            wiki_text = self.wiki_dataset["text"]
            wiki_id = self.wiki_dataset["document_id"]
            wiki_title = self.wiki_dataset["title"]

            self.wiki_context_id_dict = {k: v for k, v in zip(wiki_text, wiki_id)}
            self.wiki_id_context_dict = {k: v for k, v in zip(wiki_id, wiki_text)}
            self.wiki_id_title_dict = {k: v for k, v in zip(wiki_id, wiki_title)}

            with open("wiki_context_id_pair.bin", "wb") as file:
                pickle.dump(self.wiki_context_id_dict, file)
            with open("wiki_id_context_pair.bin", "wb") as file:
                pickle.dump(self.wiki_id_context_dict, file)
            with open("id_title_pair.bin", "wb") as file:
                pickle.dump(self.wiki_id_title_dict, file)

        self.wiki_corpus = list(self.wiki_context_id_dict.keys())


class SparseRetrieval(Retrieval):
    def __init__(
        self,
        tokenizer,
        data_path="../data/",
        context_path="/opt/ml/data/preprocess_wiki.json",
    ):
        super().__init__(tokenizer, data_path=data_path, context_path=context_path)
        self.tokenized_corpus = [
            self.tokenizer.tokenize(context) for context in self.wiki_corpus
        ]
        if os.path.isfile("BM25Okapi.bin"):
            with open("BM25Okapi.bin", "rb") as f:
                self.bm25 = pickle.load(f)
        else:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
            with open("BM25Okapi.bin", "wb") as f:
                pickle.dump(self.bm25, f)

        self.es = Elasticsearch()
        self.index_name, self.index_setting = self.__get_index_settings()
        if self.es.indices.exists(self.index_name):
            self.es.indices.delete(index=self.index_name)
        self.es.indices.create(index=self.index_name, body=self.index_setting)
        helpers.bulk(self.es, self.__get_doc(self.index_name))

        ####
        # bug? 항상 첫 서치는 시간초과가 발생해서 init에서 한번 처리
        try:
            self.es.search(index=self.index_name, q="test", size=10)
        except:
            self.es.search(index=self.index_name, q="test", size=10)
        ####

    def get_topk_doc_id_and_score(self, query, top_k):

        try:
            res = self.es.search(index=self.index_name, q=query, size=top_k)
            hits = res["hits"]["hits"]
            top_k_list = []
            top_k_score = []

            for hit in hits:
                wiki_id = hit["_id"]
                ctx = hit["_source"]["content"]
                score = hit["_score"]

                top_k_list.append(self.wiki_context_id_dict[ctx])
                top_k_score.append(score)

        except Exception as e:
            top_k_list = []
            tokenized_query = self.tokenizer.tokenize(query)
            top_n_text = self.bm25.get_top_n(tokenized_query, self.wiki_corpus, n=top_k)

            for ctx in top_n_text:
                top_k_list.append(self.wiki_context_id_dict[ctx])
            top_k_score = self.bm25.get_scores(tokenized_query).tolist()
            top_k_score.sort(reverse=True)
            top_k_score = top_k_score[:top_k]

        return top_k_list, top_k_score

    def get_topk_doc_id_and_score_for_querys(self, querys, top_k):
        query_ids = {}
        query_scores = {}
        for i in tqdm(range(len(querys))):
            query = querys[i]
            top_k_ids, top_k_scores = self.get_topk_doc_id_and_score(query, top_k)
            query_ids[query] = top_k_ids
            query_scores[query] = top_k_scores
        return query_ids, query_scores

    def __get_index_settings(self):
        INDEX_NAME = "wiki_index"
        INDEX_SETTINGS = {
            "settings": {
                "index": {
                    "analysis": {
                        "analyzer": {
                            "korean": {
                                "type": "custom",
                                "tokenizer": "nori_tokenizer",
                                "filter": ["shingle"],
                            }
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "content": {
                        "type": "text",
                        "analyzer": "korean",
                        "search_analyzer": "korean",
                    },
                    "title": {
                        "type": "text",
                        "analyzer": "korean",
                        "search_analyzer": "korean",
                    },
                }
            },
        }
        return INDEX_NAME, INDEX_SETTINGS

    def __get_doc(self, index_name):
        doc = [
            {
                "_index": index_name,
                "_id": self.wiki_dataset.iloc[i]["document_id"],
                "title": self.wiki_dataset.iloc[i]["title"],
                "content": self.wiki_dataset.iloc[i]["text"],
            }
            for i in range(self.wiki_dataset.shape[0])
        ]
        return doc


class DenseRetrieval(Retrieval):
    def __init__(
        self,
        tokenizer,
        q_encoder_path,
        p_encoder_path,
        data_path="../data/",
        context_path="/opt/ml/data/preprocess_wiki.json",
    ):
        super().__init__(tokenizer, data_path=data_path, context_path=context_path)

        q_encoder = BertEncoder.from_pretrained("q_encoder_path")
        p_encoder = BertEncoder.from_pretrained("p_encoder_path")

        if os.path.isfile("dense_embedding.bin"):
            with open("dense_embedding.bin", "rb") as f:
                self.p_embs = pickle.load(f)
        else:
            self.p_embs = self.get_wiki_dense_embedding(p_encoder)
            with open("dense_embedding.bin", "wb") as f:
                pickle.dump(self.p_embs, f)

    def get_wiki_dense_embedding(self, p_encoder):
        eval_batch_size = 32
        wiki_title_corpus = []
        for i in range(len(self.wiki_corpus)):
            wiki_title_corpus.append(
                self.wiki_id_title_dict[self.wiki_context_id_dict[self.wiki_corpus[i]]]
            )

        if torch.cuda.is_available():
            p_encoder.cuda()

        p_seqs = self.tokenizer(
            wiki_title_corpus,
            self.wiki_corpus,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        dataset = TensorDataset(
            p_seqs["input_ids"], p_seqs["attention_mask"], p_seqs["token_type_ids"]
        )
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=eval_batch_size)

        p_embs = []

        with torch.no_grad():

            epoch_iterator = tqdm(dataloader, desc="Iteration", position=0, leave=True)
            p_encoder.eval()
            for _, batch in enumerate(epoch_iterator):
                batch = tuple(t.cuda() for t in batch)
                p_inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                }
                outputs = p_encoder(**p_inputs).to("cpu").numpy()
                p_embs.extend(outputs)
        torch.cuda.empty_cache()
        p_embs = np.array(p_embs)

        return p_embs

    def to_cuda(batch):
        return tuple(t.cuda() for t in batch)
