# for arguments
import argparse
from arguments import get_args_parser

# torch
import torch
from torch.utils.data import TensorDataset

# settings
from utils import seed_everything

# hf
from datasets import load_from_disk
from transformers import AutoTokenizer

# negative sample
from utils import append_neg_p

# basics
import json
import pickle
import os
from tqdm import tqdm

from model import BertEncoder
from train import train


def main(args):
    # model for use
    model_checkpoint = "bert-base-multilingual-cased"

    # loading train and validation data
    train_data = load_from_disk(args.train_data)
    validation_data = load_from_disk(args.val_data)
    test_data = load_from_disk(args.val_data)
    
    # test
    #import numpy as np
    #sample_idx = np.random.choice(range(len(train_data)), 20)
    #train_data = train_data[sample_idx]

    # load wiki data and remove duplicates
    with open('../data/wikipedia_documents.json', "r", encoding="utf-8") as f:
        wiki = json.load(f)
    #corpus = list(dict.fromkeys([v["text"] for v in wiki.values()]))
    wiki = [v['text'] for v in wiki.values()]

    # append negative passages
    p_with_neg = append_neg_p(args.num_neg, wiki, train_data['context'], train_data['answers'])

    # tokenize
    ## train_data
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    train_q_seqs = tokenizer(train_data['question'], padding="max_length", truncation=True, return_tensors='pt')
    train_p_seqs = tokenizer(p_with_neg, padding="max_length", truncation=True, return_tensors='pt')
    ## validation_data
    val_q_seqs = tokenizer(validation_data['question'], padding="max_length", truncation=True, return_tensors='pt').to('cuda')
    test_q_seqs = tokenizer(test_data['question'], padding="max_length", truncation=True, return_tensors='pt').to('cuda') 
    if os.path.isfile('pickle/wiki_token'):
        with open("pickle/wiki_token", "rb") as file:
            val_p_seqs = pickle.load(file)
    else:
        val_p_seqs = []
        for p in tqdm(wiki):
            val_p_seqs.append(tokenizer(p, padding="max_length", truncation=True, return_tensors='pt'))
        with open("pickle/wiki_token", "wb") as file:
            pickle.dump(val_p_seqs, file)

    max_len = train_p_seqs['input_ids'].size(-1)
    train_p_seqs['input_ids'] = train_p_seqs['input_ids'].view(-1, args.num_neg+1, max_len)
    train_p_seqs['attention_mask'] = train_p_seqs['attention_mask'].view(-1, args.num_neg+1, max_len)
    train_p_seqs['token_type_ids'] = train_p_seqs['token_type_ids'].view(-1, args.num_neg+1, max_len)

    # create train dataset
    train_dataset = TensorDataset(train_p_seqs['input_ids'], train_p_seqs['attention_mask'], train_p_seqs['token_type_ids'], 
                        train_q_seqs['input_ids'], train_q_seqs['attention_mask'], train_q_seqs['token_type_ids'])

    # load model
    p_encoder = BertEncoder.from_pretrained(model_checkpoint).cuda()
    q_encoder = BertEncoder.from_pretrained(model_checkpoint).cuda()
    model_dict = torch.load("dense_encoder/encoder.pth")
    p_encoder.load_state_dict(model_dict['p_encoder'])
    q_encoder.load_state_dict(model_dict['q_encoder'])

    train(args, train_dataset, val_q_seqs, val_p_seqs, validation_data['context'], test_q_seqs, wiki, p_encoder, q_encoder)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dense Embedding', parents=[get_args_parser()])
    args = parser.parse_args()
    seed_everything(args.seed)
    main(args)