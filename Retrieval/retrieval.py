import pickle
import os
import pandas as pd
from elasticsearch import Elasticsearch, helpers
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer
from tqdm import tqdm
from .dense_model import BertEncoder
import torch
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler
import numpy as np


class Retrieval:
    def __init__(
        self,
        tokenizer,
        data_path="/opt/ml/mrc-level2-nlp-08/Retrieval/",
        caching_path="caching/",
        context_path="/opt/ml/data/preprocess_wiki.json",
    ):
        '''
        Retrieval의 최상위 클래스
        Sparse, Dense, Hybrid 모두 이 클래스를 상속받아서 사용합니다.        
        '''
        self.tokenizer = tokenizer
        self.wiki_dataset = pd.read_json(context_path, orient="index")

        caching_context_id_path = data_path + caching_path + "wiki_context_id_pair.bin"
        caching_id_context_path = data_path + caching_path + "wiki_id_context_pair.bin"
        caching_id_title_path = data_path + caching_path + "id_title_pair.bin"

        if (
            os.path.isfile(caching_context_id_path)
            and os.path.isfile(caching_id_context_path)
            and os.path.isfile(caching_id_title_path)
        ):
            with open(caching_context_id_path, "rb") as f:
                self.wiki_context_id_dict = pickle.load(f)
            with open(caching_id_context_path, "rb") as f:
                self.wiki_id_context_dict = pickle.load(f)
            with open(caching_id_title_path, "rb") as f:
                self.wiki_id_title_dict = pickle.load(f)
        else:
            wiki_text = self.wiki_dataset["text"]
            wiki_id = self.wiki_dataset["document_id"]
            wiki_title = self.wiki_dataset["title"]

            self.wiki_context_id_dict = {k: v for k, v in zip(wiki_text, wiki_id)}
            self.wiki_id_context_dict = {k: v for k, v in zip(wiki_id, wiki_text)}
            self.wiki_id_title_dict = {k: v for k, v in zip(wiki_id, wiki_title)}

            with open(caching_context_id_path, "wb") as file:
                pickle.dump(self.wiki_context_id_dict, file)
            with open(caching_id_context_path, "wb") as file:
                pickle.dump(self.wiki_id_context_dict, file)
            with open(caching_id_title_path, "wb") as file:
                pickle.dump(self.wiki_id_title_dict, file)

        self.wiki_corpus = list(self.wiki_context_id_dict.keys())

    def get_topk_doc_id_and_score(self, query, top_k):
        '''
        query를 입력받아 wiki context에서 점수가 높은 top_k개의 context의 id와 해당 score를 
        점수가 높은 순으로 output으로 내보냅니다.
        '''
        pass

    def get_topk_doc_id_and_score_for_querys(self, querys, top_k):
        '''
        query들을 list type으로 받아서 점수가 높은 top_k개의 context의 id와 해당 socre를 점수가 높은 순으로 
        output으로 내보냅니다. get_topk_doc_id_and_score와의 차이점은 query를 리스트로 받기 때문에
        최종 output은 query - ids, query - socres의 dict type입니다.
        '''
        pass


class SparseRetrieval(Retrieval):
    def __init__(
        self,
        tokenizer,
        data_path="/opt/ml/mrc-level2-nlp-08/Retrieval/",
        caching_path="caching/",
        context_path="/opt/ml/data/preprocess_wiki.json",
    ):
        super().__init__(
            tokenizer,
            data_path=data_path,
            caching_path=caching_path,
            context_path=context_path,
        )
        self.tokenized_corpus = [
            self.tokenizer.tokenize(context) for context in self.wiki_corpus
        ]
        caching_bm25_path = data_path + caching_path + "BM25Okapi.bin"
        if os.path.isfile(caching_bm25_path):
            with open(caching_bm25_path, "rb") as f:
                self.bm25 = pickle.load(f)
        else:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
            with open(caching_bm25_path, "wb") as f:
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
        data_path="/opt/ml/mrc-level2-nlp-08/Retrieval/",
        caching_path="caching/",
        context_path="/opt/ml/data/preprocess_wiki.json",
    ):
        super().__init__(
            tokenizer,
            data_path=data_path,
            caching_path=caching_path,
            context_path=context_path,
        )

        self.q_encoder = BertEncoder.from_pretrained(data_path + q_encoder_path)
        self.p_encoder = BertEncoder.from_pretrained(data_path + p_encoder_path)
        if torch.cuda.is_available():
            self.p_encoder.cuda()
            self.q_encoder.cuda()

        dense_embedding_path = data_path + caching_path + "dense_embedding.bin"

        if os.path.isfile(dense_embedding_path):
            with open(dense_embedding_path, "rb") as f:
                self.p_embs = pickle.load(f)
        else:
            self.p_embs = self.get_wiki_dense_embedding(self.p_encoder)
            with open(dense_embedding_path, "wb") as f:
                pickle.dump(self.p_embs, f)

    def get_wiki_dense_embedding(self, p_encoder):
        eval_batch_size = 32
        wiki_title_corpus = []
        for i in range(len(self.wiki_corpus)):
            wiki_title_corpus.append(
                self.wiki_id_title_dict[self.wiki_context_id_dict[self.wiki_corpus[i]]]
            )

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

    def get_topk_doc_id_and_score(self, query, top_k):
        with torch.no_grad():
            self.q_encoder.eval()

            q_seqs_val = self.tokenizer(
                [query], padding="max_length", truncation=True, return_tensors="pt"
            ).to("cuda")
            q_emb = self.q_encoder(**q_seqs_val).to("cpu")  # (num_query, emb_dim)

            p_embs = torch.Tensor(self.p_embs).squeeze()  # (num_passage, emb_dim)
            dot_prod_scores = torch.matmul(q_emb, torch.transpose(p_embs, 0, 1))

            rank = (
                torch.argsort(dot_prod_scores, dim=1, descending=True)
                .squeeze()
                .to("cpu")
                .numpy()
                .tolist()
            )
            scores = []
            for r in rank[:top_k]:
                scores.append(dot_prod_scores[0][r].item())

        return rank[:top_k], scores

    def get_topk_doc_id_and_score_for_querys(self, querys, top_k):
        q_seqs = self.tokenizer(
            querys,
            max_length=64,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        dataset = TensorDataset(
            q_seqs["input_ids"], q_seqs["attention_mask"], q_seqs["token_type_ids"]
        )
        query_sampler = SequentialSampler(dataset)
        query_dataloader = DataLoader(dataset, sampler=query_sampler, batch_size=32)
        q_embs = []
        with torch.no_grad():

            epoch_iterator = tqdm(
                query_dataloader, desc="Iteration", position=0, leave=True
            )
            self.q_encoder.eval()

            for _, batch in enumerate(epoch_iterator):
                batch = tuple(t.cuda() for t in batch)

                q_inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                }

                outputs = self.q_encoder(**q_inputs).to("cpu").numpy()
                q_embs.extend(outputs)
        q_embs = np.array(q_embs)
        if torch.cuda.is_available():
            p_embs_cuda = torch.Tensor(self.p_embs).to("cuda")
            q_embs_cuda = torch.Tensor(q_embs).to("cuda")
        dot_prod_scores = torch.matmul(q_embs_cuda, torch.transpose(p_embs_cuda, 0, 1))
        rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()

        query_ids = {}
        query_scores = {}
        idx = 0
        for i in tqdm(range(len(querys))):
            p_list = []
            scores = []
            q = querys[i]
            for j in range(top_k):
                p_list.append(self.wiki_context_id_dict[self.wiki_corpus[rank[idx][j]]])
                scores.append(dot_prod_scores[idx][rank[idx][j]].item())
            query_ids[q] = p_list
            query_scores[q] = scores
            idx += 1

        return query_ids, query_scores

    def to_cuda(batch):
        return tuple(t.cuda() for t in batch)


class HybridRetrieval(Retrieval):
    def __init__(
        self,
        tokenizer,
        q_encoder_path,
        p_encoder_path,
        data_path="/opt/ml/mrc-level2-nlp-08/Retrieval/",
        caching_path="caching/",
        context_path="/opt/ml/data/preprocess_wiki.json",
    ):
        super().__init__(
            tokenizer,
            data_path=data_path,
            caching_path=caching_path,
            context_path=context_path,
        )

        self.sparse_retrieval = SparseRetrieval(tokenizer=tokenizer)
        self.dense_retrieval = DenseRetrieval(
            tokenizer=tokenizer,
            p_encoder_path=p_encoder_path,
            q_encoder_path=q_encoder_path,
        )
        self.q_encoder = self.dense_retrieval.q_encoder
        self.p_embs = self.dense_retrieval.p_embs
        if torch.cuda.is_available():
            self.p_embs = torch.Tensor(self.p_embs).to("cuda")

    def get_topk_doc_id_and_score(self, query, top_k):
        es_id, es_score = self.sparse_retrieval.get_topk_doc_id_and_score(
            query=query, top_k=top_k
        )
        return self.__rerank(query, es_id, es_score)

    def get_topk_doc_id_and_score_for_querys(self, querys, top_k):
        hybrid_ids = {}
        hybrid_scores = {}

        for i in tqdm(range(len(querys))):
            query = querys[i]
            doc_ids, scores = self.get_topk_doc_id_and_score(query, top_k)
            hybrid_ids[query] = doc_ids
            hybrid_scores[query] = scores

        return hybrid_ids, hybrid_scores

    def __rerank(self, query, es_id, es_score):
        p_embs = self.p_embs
        with torch.no_grad():
            self.q_encoder.eval()
            q_seqs_val = self.tokenizer(
                query, padding="max_length", truncation=True, return_tensors="pt"
            ).to("cuda")
            q_emb = self.q_encoder(**q_seqs_val).to("cuda")
        dot_prod_scores = torch.matmul(q_emb, torch.transpose(p_embs, 0, 1))
        rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()
        rank = rank.cpu().numpy().tolist()

        es_id_score = {k: v for k, v in zip(es_id, es_score)}

        hybrid_id_score = dict()

        for i in range(len(rank)):
            dense_id = self.wiki_context_id_dict[self.wiki_corpus[i]]
            if dense_id in es_id_score:
                lin_score = dot_prod_scores[0][i].item() + es_id_score[dense_id]
                hybrid_id_score[dense_id] = lin_score

        hybrid_id_score = list(hybrid_id_score.items())
        hybrid_id_score.sort(key=lambda x: x[1], reverse=True)
        hybrid_ids = list(map(lambda x: x[0], hybrid_id_score))
        hybrid_scores = list(map(lambda x: x[1], hybrid_id_score))

        return hybrid_ids, hybrid_scores
