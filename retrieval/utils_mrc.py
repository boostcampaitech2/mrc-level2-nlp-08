import os
import random
import datasets
import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import (
    Dataset,
    DataLoader,
    RandomSampler,
    TensorDataset,
    SequentialSampler,
)
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


def get_tensor_for_dense(
    data_path: str, max_context_seq_length: int, max_question_seq_length: int, tokenizer
) -> TensorDataset:
    """
    question과 context를 tokenize 한 수 TensorDataset으로 concat하여 return합니다.
    """
    dataset = load_from_disk(data_path)
    q_seqs = tokenizer(
        dataset["question"],
        max_length=max_question_seq_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    p_seqs = tokenizer(
        dataset["title"],
        dataset["context"],
        max_length=max_context_seq_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    tensor_dataset = TensorDataset(
        p_seqs["input_ids"],
        p_seqs["attention_mask"],
        p_seqs["token_type_ids"],
        q_seqs["input_ids"],
        q_seqs["attention_mask"],
        q_seqs["token_type_ids"],
    )

    return tensor_dataset


def get_tensor_for_dense_temp(
    data_path: str, max_context_seq_length: int, max_question_seq_length: int, tokenizer
) -> TensorDataset:
    """
    validation 전처리 -> 삭제 예정
    """
    dataset = load_from_disk(data_path)
    ctx = []

    # print(dataset["context"][0])
    for i in range(len(dataset)):
        ctx.append(preprocess(dataset["context"][i]))

    q_seqs = tokenizer(
        dataset["question"],
        max_length=max_question_seq_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    p_seqs = tokenizer(
        ctx,
        max_length=max_context_seq_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    tensor_dataset = TensorDataset(
        p_seqs["input_ids"],
        p_seqs["attention_mask"],
        p_seqs["token_type_ids"],
        q_seqs["input_ids"],
        q_seqs["attention_mask"],
        q_seqs["token_type_ids"],
    )

    return tensor_dataset


def get_tensor_for_dense_negative(
    data_path: str,
    bm25_path: str,
    max_context_seq_length: int,
    max_question_seq_length: int,
    tokenizer,
) -> TensorDataset:

    dataset = load_from_disk(data_path).to_pandas()
    dataset["context"] = dataset["context"].apply(preprocess)
    ctx = dataset["context"].to_list()
    q_list = dataset["question"].to_list()
    # 시간이 많이 걸림 -> 미리 처리해둘것
    print(ctx[0:3])
    with open(bm25_path, "rb") as file:
        elastic_pair = pickle.load(file)

    neg_ctx = []
    for i in tqdm(range(len(dataset))):
        query = dataset["question"][i]
        ground_truth = ctx[i]
        answer = dataset["answers"][i]["text"][0]
        cnt = 1
        idx = 0

        while cnt != 0:
            if ground_truth != elastic_pair[query][idx] and not (
                answer in elastic_pair[query][idx]
            ):
                # 비슷한 context를 추가하되 정답을 포함하지 않는 문장을 추가한다.
                neg_ctx.append(elastic_pair[query][idx])
                cnt -= 1
            idx += 1
            if idx == 100:  # index를 넘어가면 break
                break
    print("ctx")
    print(len(ctx))
    print("neg_ctx")
    print(len(neg_ctx))

    print(ctx[0])
    print()
    print(neg_ctx[0])

    q_seqs = tokenizer(
        q_list,
        max_length=max_question_seq_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    p_seqs = tokenizer(
        ctx,
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

    tensor_dataset = TensorDataset(
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

    return tensor_dataset


def preprocess(text):
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\\n", " ", text)  # remove newline character
    text = re.sub(r"\s+", " ", text)  # remove continuous spaces
    text = re.sub(r"#", " ", text)

    return text


class InBatchNegativeRandomDataset(Dataset):
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
        # wiki_path:str,
        bm25_path: str,
        # context_id_pair_path : str,
        max_context_seq_length: int,
        max_question_seq_length: int,
        num_neg,
        tokenizer,
    ):
        # 전처리된 위키
        # wiki_dataset = pd.read_json(
        #     "/opt/ml/data/preprocess_wiki.json", orient="index"
        # )
        # 중복 제거
        # wiki_dataset = wiki_dataset.drop_duplicates(
        #     ["text", "title"], ignore_index=True
        # )

        # doc_id - context dict
        with open("/opt/ml/data/wiki_id_context_pair.bin", "rb") as f:
            wiki_id_context = pickle.load(f)
        with open("/opt/ml/data/wiki_id_title_pair.bin", "rb") as f:
            wiki_id_title = pickle.load(f)
        # context - doc_id
        with open("/opt/ml/data/wiki_context_id_pair.bin", "rb") as f:
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
            pos_title,
            pos_ctx,
            max_length=max_context_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        np_seqs = tokenizer(
            neg_title,
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


class InBatchNegativeRandomDatasetNoTitle(Dataset):
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
        # wiki_path:str,
        bm25_path: str,
        # context_id_pair_path : str,
        max_context_seq_length: int,
        max_question_seq_length: int,
        num_neg,
        tokenizer,
    ):
        # 전처리된 위키
        # wiki_dataset = pd.read_json(
        #     "/opt/ml/data/preprocess_wiki.json", orient="index"
        # )
        # 중복 제거
        # wiki_dataset = wiki_dataset.drop_duplicates(
        #     ["text", "title"], ignore_index=True
        # )

        # doc_id - context dict
        with open("/opt/ml/data/wiki_id_context_pair.bin", "rb") as f:
            wiki_id_context = pickle.load(f)
        with open("/opt/ml/data/wiki_id_title_pair.bin", "rb") as f:
            wiki_id_title = pickle.load(f)
        # context - doc_id
        with open("/opt/ml/data/wiki_context_id_pair.bin", "rb") as f:
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
            # pos_title,
            pos_ctx,
            max_length=max_context_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        np_seqs = tokenizer(
            # neg_title,
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
