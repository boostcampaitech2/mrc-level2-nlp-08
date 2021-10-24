import os
import random
import datasets
import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader, RandomSampler, TensorDataset, SequentialSampler
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
    # ctx = []
    # print(dataset["context"][0])
    # for i in tqdm(range(len(dataset))):
    #     ctx.append(preprocess(dataset["context"][i]))
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
