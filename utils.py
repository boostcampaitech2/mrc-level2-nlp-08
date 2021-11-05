import torch
import numpy as np
import random
import os

import argparse
import pickle

# for reproducibility
def seed_everything(seed: int = 2021):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if device == "cuda:0":
        torch.cuda.manual_seed(seed)  # type: ignore
        torch.backends.cudnn.deterministic = True  # type: ignore
        torch.backends.cudnn.benchmark = True

def append_neg_p(num_neg: int, wiki: list, train_context: list, answer: list):
    with open("pickle/elastic_train_neg", "rb") as file:
        elastic_train_neg = pickle.load(file)

    p_with_neg = []
    
    for idx, c in enumerate(train_context):
        ans_text = answer[idx]['text'][0]
        neg_idx = 0
        counter = 0
        p_with_neg.append(c)

        while True:
            if ans_text not in wiki[elastic_train_neg[idx][neg_idx]]:
                counter += 1
                p_with_neg.append(wiki[elastic_train_neg[idx][neg_idx]])
            
            if counter == num_neg:
                break
            
            neg_idx += 1
    
    return p_with_neg

# old deprecated version
'''
def append_neg_p(num_neg: int, corpus: list, train_context: list):
    corpus = np.array(corpus)
    p_with_neg = []
    
    for c in train_context:
        while True:
            neg_idxs = np.random.randint(len(corpus), size=num_neg)

            if not c in corpus[neg_idxs]:
                p_neg = corpus[neg_idxs]

                p_with_neg.append(c)
                p_with_neg.extend(p_neg)
                break
    
    return p_with_neg
'''