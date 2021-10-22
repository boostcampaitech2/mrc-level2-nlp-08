from koeda import AEDA
import pprint
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
from datasets import load_from_disk, load_dataset, Features, Value, Sequence, DatasetDict, Dataset
import os
import re


SPACE_TOKEN = "\u241F"


def replace_space(text: str) -> str:
    return text.replace(" ", SPACE_TOKEN)


def revert_space(text: list) -> str:
    clean = " ".join("".join(text).replace(SPACE_TOKEN, " ").split()).strip()
    return clean


class myAEDA(AEDA):
    def _aeda(self, data: str, p: float) -> str:
        if p is None:
            p = self.ratio

        split_words = self.morpheme_analyzer.morphs(replace_space(data))
        words = self.morpheme_analyzer.morphs(data)

        new_words = []
        q = random.randint(1, int(p * len(words) + 1))
        qs_list = [index for index in range(len(split_words)) if split_words[index] != SPACE_TOKEN]
        if len(qs_list) < q:
            q = len(qs_list)
        qs = random.sample(qs_list, q)

        for j, word in enumerate(split_words):
            if j in qs:
                new_words.append(SPACE_TOKEN)
                new_words.append(self.punctuations[random.randint(0, len(self.punctuations) - 1)])
                new_words.append(SPACE_TOKEN)
                new_words.append(word)
            else:
                new_words.append(word)

        augmented_sentences = revert_space(new_words)

        return augmented_sentences


def preprocess(text):
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\\n", " ", text)  # remove newline character
    text = re.sub(r"\s+", " ", text)  # remove continuous spaces
    text = re.sub(r"#", " ", text)

    return text


def preprocess_dataset(datasets):
    new_dataset = []
    for i, data in enumerate(datasets):
        context = data["context"]
        start_idx = data["answers"]["answer_start"][0]

        before_start = context[:start_idx]
        after_start = context[start_idx:]

        processed_before = preprocess(before_start)
        processed_after = preprocess(after_start)
        processed_context = processed_before + processed_after
        moved_idx = len(before_start) - len(processed_before)

        data["context"] = processed_context
        data["answers"]["answer_start"][0] = start_idx - moved_idx
        data["question"] = result[i]
        new_dataset.append(data)

    return new_dataset


train_dataset = load_from_disk("../data/train_dataset/train/")
# pd_train = pd.DataFrame(preprocess_dataset(train_dataset))

pp = pprint.PrettyPrinter()


# 물음표 제외
aeda = myAEDA(morpheme_analyzer="Mecab", punc_ratio=0.2, punctuations=[".", ",", "!", ";", ":"])

# features: ['__index_level_0__', 'answers', 'context', 'document_id', 'id', 'question', 'title'],
question = train_dataset["question"]

result = []
for _, t in enumerate(tqdm(question)):
    result.append(aeda(t, repetition=1))

# result = aeda(text, repetition=1)
print(pp.pprint(result[:5]))

new_train_data = pd.DataFrame(preprocess_dataset(train_dataset))

train_features = Features(
    {
        "answers": Sequence(
            feature={"text": Value(dtype="string", id=None), "answer_start": Value(dtype="int32", id=None)},
            length=-1,
            id=None,
        ),
        "context": Value(dtype="string", id=None),
        "id": Value(dtype="string", id=None),
        "question": Value(dtype="string", id=None),
    }
)

Dataset.from_pandas(new_train_data, features=train_features).save_to_disk(
    "/opt/ml/data/new_train_dataset/train"
)
