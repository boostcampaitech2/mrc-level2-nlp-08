import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import gc

gc.enable()
import math
import json
import time
import random
import multiprocessing
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from sklearn import model_selection
from string import punctuation

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler

import datasets
from datasets import load_from_disk

import transformers
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    logging,
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    AutoModelForQuestionAnswering,
)


def fix_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Config:
    # model
    model_type = "roberta"
    model_name_or_path = "klue/roberta-large"
    config_name = "klue/roberta-large"
    fp16 = True
    fp16_opt_level = "O1"
    gradient_accumulation_steps = 2

    # tokenizer
    tokenizer_name = "klue/roberta-large"
    max_seq_length = 384
    doc_stride = 128

    # train
    epochs = 5
    train_batch_size = 16
    eval_batch_size = 128

    # optimzer
    optimizer_type = "AdamW"
    learning_rate = 1.809598615643362e-05
    weight_decay = 0.19132033828553255
    epsilon = 1e-8
    max_grad_norm = 1.0

    # scheduler
    decay_name = "linear-warmup"
    warmup_ratio = 0.1

    # logging
    logging_steps = 10

    # evaluate
    output_dir = "output"
    seed = 107


def make_model(args):
    config = AutoConfig.from_pretrained(args.config_name)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    model = AutoModelForQuestionAnswering.from_pretrained(
        args.model_name_or_path, config=config
    )
    return config, tokenizer, model


def prepare_test_features(args, example, tokenizer):
    example["question"] = example["question"].lstrip()

    tokenized_example = tokenizer(
        example["question"],
        example["context"],
        truncation="only_second",
        max_length=args.max_seq_length,
        stride=args.doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        return_token_type_ids=False,
        padding="max_length",
    )

    features = []
    for i in range(len(tokenized_example["input_ids"])):
        feature = {}
        feature["example_id"] = example["id"]
        feature["context"] = example["context"]
        feature["question"] = example["question"]
        feature["input_ids"] = tokenized_example["input_ids"][i]
        feature["attention_mask"] = tokenized_example["attention_mask"][i]
        feature["offset_mapping"] = tokenized_example["offset_mapping"][i]
        feature["sequence_ids"] = [
            0 if i is None else i for i in tokenized_example.sequence_ids(i)
        ]
        features.append(feature)
    return features


import collections


def postprocess_qa_predictions(
    examples, features, raw_predictions, n_best_size=20, max_answer_length=30
):
    all_start_logits, all_end_logits = raw_predictions

    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    predictions = collections.OrderedDict()

    print(
        f"Post-processing {len(examples)} example predictions split into {len(features)} features."
    )

    for example_index, example in examples.iterrows():
        feature_indices = features_per_example[example_index]

        min_null_score = None
        valid_answers = []

        context = example["context"]
        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]

            sequence_ids = features[feature_index]["sequence_ids"]
            context_index = 1

            features[feature_index]["offset_mapping"] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(features[feature_index]["offset_mapping"])
            ]
            offset_mapping = features[feature_index]["offset_mapping"]
            cls_index = features[feature_index]["input_ids"].index(
                tokenizer.cls_token_id
            )
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score

            start_indexes = np.argsort(start_logits)[
                -1 : -n_best_size - 1 : -1
            ].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": context[start_char:end_char],
                        }
                    )

        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[
                0
            ]
        else:
            best_answer = {"text": "", "score": 0.0}

        predictions[example["id"]] = best_answer["text"]

    return predictions


test = pd.DataFrame(load_from_disk("../../hb_k_20_v3"))

# base_model_path = '../input/chaii-qa-5-fold-xlmroberta-torch-fit'

tokenizer = AutoTokenizer.from_pretrained(Config().tokenizer_name)

test_features = []
for i, row in test.iterrows():
    test_features += prepare_test_features(Config(), row, tokenizer)

args = Config()
test_dataset = test
test_dataloader = DataLoader(
    test_dataset,
    batch_size=args.eval_batch_size,
    sampler=SequentialSampler(test_dataset),
    pin_memory=True,
    drop_last=False,
)


def get_predictions(checkpoint_path):
    config, tokenizer, model = make_model(Config())
    model.cuda()
    model.load_state_dict(torch.load(checkpoint_path))

    start_logits = []
    end_logits = []
    for batch in test_dataloader:
        with torch.no_grad():
            outputs_start, outputs_end = model(
                batch["input_ids"].cuda(), batch["attention_mask"].cuda()
            )
            start_logits.append(outputs_start.cpu().numpy().tolist())
            end_logits.append(outputs_end.cpu().numpy().tolist())
            del outputs_start, outputs_end
    del model, tokenizer, config
    gc.collect()
    return np.vstack(start_logits), np.vstack(end_logits)


start_logits1, end_logits1 = get_predictions(
    "../kfold/0_fold_train/checkpoint-568/pytorch_model.bin"
)
start_logits2, end_logits2 = get_predictions(
    "../kfold/1_fold_train/checkpoint-564/pytorch_model.bin"
)
start_logits3, end_logits3 = get_predictions(
    "../kfold/2_fold_train/checkpoint-564/pytorch_model.bin"
)
start_logits4, end_logits4 = get_predictions(
    "../kfold/3_fold_train/checkpoint-566/pytorch_model.bin"
)
start_logits5, end_logits5 = get_predictions(
    "../kfold/4_fold_train/checkpoint-564/pytorch_model.bin"
)


start_logits = (
    start_logits1 + start_logits2 + start_logits3 + start_logits4 + start_logits5
) / 5
end_logits = (end_logits1 + end_logits2 + end_logits3 + end_logits4 + end_logits5) / 5


fin_preds = postprocess_qa_predictions(test, test_features, (start_logits, end_logits))

submission = []
for p1, p2 in fin_preds.items():
    p2 = " ".join(p2.split())
    p2 = p2.strip(punctuation)
    submission.append((p1, p2))

sample = pd.DataFrame(submission, columns=["id", "PredictionString"])

test_data = pd.merge(left=test, right=sample, on="id")

bad_starts = [".", ",", "(", ")", "-", "–", ",", ";"]
bad_endings = ["...", "-", "(", ")", "–", ",", ";"]

cleaned_preds = []
for pred, context in test_data[["PredictionString", "context"]].to_numpy():
    if pred == "":
        cleaned_preds.append(pred)
        continue
    while any([pred.startswith(y) for y in bad_starts]):
        pred = pred[1:]
    while any([pred.endswith(y) for y in bad_endings]):
        if pred.endswith("..."):
            pred = pred[:-3]
        else:
            pred = pred[:-1]
    if pred.endswith("..."):
        pred = pred[:-3]

    cleaned_preds.append(pred)

test_data["PredictionString"] = cleaned_preds
# test_data[["id", "PredictionString"]].to_csv("submission.csv", index=False)

answers = {}
for i in range(len(test_data)):
    row = test_data.iloc[i]

    answers[row["id"]] = row["PredictionString"]

with open("fold_logits_prediction.json", "w", encoding="utf-8") as w:
    w.write(json.dumps(answers, indent=4, ensure_ascii=False) + "\n")
