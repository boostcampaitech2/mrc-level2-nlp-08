import os
import random
import datasets
import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd
import pickle
import re


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class InBatchNegativeRandomDatasetNoTitle(Dataset):
    '''
    dense retrieval 모델을 학습시킬 데이터 셋
    '''
    def __init__(
        self,
        data_path: str,
        bm25_path: str,
        max_context_seq_length: int,
        max_question_seq_length: int,
        neg_num,
        tokenizer,
    ):
        preprocess_data = self.preprocess_pos_neg(
            data_path,
            bm25_path,
            max_context_seq_length,
            max_question_seq_length,
            neg_num,
            tokenizer,
        )
        '''
        data_path
        -> query - context로 이루어진 dataset의 경로
        
        bm25_path
        -> query에 대해서 bm25로 찾아낸 유사도가 높은 context 데이터
        retrieval/SparseRetrieval의 get_topk_doc_id_and_score_for_querys 메서드로 
        해당 bin 파일을 만들 수 있습니다.

        max_context_seq_length
        -> context의 max_length

        max_question_seq_length
        -> question의 max_lenth

        neg_num
        -> retrieval 학습에 사용할 Inbatch negative 외에 hard negative sample의 갯수

        '''

        self.p_input_ids = preprocess_data[0]
        self.p_attension_mask = preprocess_data[1]
        self.p_token_type_ids = preprocess_data[2]

        self.np_input_ids = preprocess_data[3]
        self.np_attension_mask = preprocess_data[4]
        self.np_token_type_ids = preprocess_data[5]

        self.q_input_ids = preprocess_data[6]
        self.q_attension_mask = preprocess_data[7]
        self.q_token_type_ids = preprocess_data[8]

    def __len__(self):
        return self.p_input_ids.size()[0]

    def __getitem__(self, index):
        return (
            self.p_input_ids[index],
            self.p_attension_mask[index],
            self.p_token_type_ids[index],
            self.np_input_ids[index],
            self.np_attension_mask[index],
            self.np_token_type_ids[index],
            self.q_input_ids[index],
            self.q_attension_mask[index],
            self.q_token_type_ids[index],
        )

    def preprocess_pos_neg(
        self,
        data_path: str,
        bm25_path: str,
        max_context_seq_length: int,
        max_question_seq_length: int,
        num_neg,
        tokenizer,
    ):
        data_path = "/opt/ml/mrc-level2-nlp-08/Retrieval/"
        caching_path = "caching/"
        caching_context_id_path = data_path + caching_path + "wiki_context_id_pair.bin"
        caching_id_context_path = data_path + caching_path + "wiki_id_context_pair.bin"
        caching_id_title_path = data_path + caching_path + "id_title_pair.bin"

        # doc_id - context dict
        with open(caching_id_context_path, "rb") as f:
            wiki_id_context = pickle.load(f)
        # doc_id - title dict
        with open(caching_id_title_path, "rb") as f:
            wiki_id_title = pickle.load(f)
        # context - doc_id dict
        with open(caching_context_id_path, "rb") as f:
            wiki_context_id = pickle.load(f)

        # question - doc_id_list
        with open(bm25_path, "rb") as file:  # query - bm25_doc_id
            elastic_question_ids = pickle.load(file)

        dataset = load_from_disk(data_path).to_pandas()

        pos_ctx = dataset["context"].to_list()
        pos_title = dataset["title"].to_list()
        questions = dataset["question"].to_list()

        # print(len(pos_ctx))

        neg_ctx = []
        neg_title = []
        for i in tqdm(range(len(pos_ctx))):
            q = questions[i]  # i 번째 question
            ground_truth = pos_ctx[i]  # 정답 문장
            ground_truth_title = pos_title[i]  # 정답 타이틀
            cnt = num_neg  # 추가할 negative context 갯수
            answer = dataset["answers"][i]["text"][0]  # 정답
            idx = 0

            while cnt != 0:
                neg_ctx_sample = wiki_id_context[elastic_question_ids[q][idx]]
                if (ground_truth != neg_ctx_sample) and (not answer in neg_ctx_sample):
                    # 비슷한 context를 추가하되 정답을 포함하지 않는 문장을 추가한다.
                    neg_ctx.append(wiki_id_context[int(elastic_question_ids[q][idx])])
                    neg_title.append(wiki_id_title[int(elastic_question_ids[q][idx])])
                    cnt -= 1
                idx += 1
                if idx == len(elastic_question_ids[q]):
                    # 예외처리 ex) 정답이 전부 포함되서 추가할 문장이 없을 경우
                    idx_step = 1
                    while cnt != 0:
                        temp_neg = pos_ctx[i - idx_step]
                        temp_neg_title = pos_title[i - idx_step]
                        # 이전에 추가된 ground truth context를 negative sample로 생성
                        neg_ctx.append(temp_neg)
                        neg_title.append(temp_neg_title)
                        idx_step += 1
                        cnt -= 1

        print(f"pos_context cnt: {len(pos_ctx)}")
        print(f"neg_context cnt: {len(neg_ctx)}")

        print(f"pos * num_neg == neg? : {len(pos_ctx) * num_neg == len(neg_ctx)}")

        q_seqs = tokenizer(
            questions,
            max_length=max_question_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        p_seqs = tokenizer(
            pos_ctx,
            max_length=max_context_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        np_seqs = tokenizer(
            neg_ctx,
            max_length=max_context_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        max_len = np_seqs["input_ids"].size(-1)
        np_seqs["input_ids"] = np_seqs["input_ids"].view(-1, num_neg, max_len)
        np_seqs["attention_mask"] = np_seqs["attention_mask"].view(-1, num_neg, max_len)
        np_seqs["token_type_ids"] = np_seqs["token_type_ids"].view(-1, num_neg, max_len)

        return (
            p_seqs["input_ids"],
            p_seqs["attention_mask"],
            p_seqs["token_type_ids"],
            np_seqs["input_ids"],
            np_seqs["attention_mask"],
            np_seqs["token_type_ids"],
            q_seqs["input_ids"],
            q_seqs["attention_mask"],
            q_seqs["token_type_ids"],
        )

    def preprocess_text(self, text):
        text = re.sub(r"\n", " ", text)
        text = re.sub(r"\\n", " ", text)
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"#", " ", text)
        return text
