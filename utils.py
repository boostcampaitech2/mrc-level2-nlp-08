import torch
import numpy as np
import random
import os

import argparse

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