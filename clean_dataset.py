import os
import json
import re
import pandas as pd
from datasets import (load_from_disk, load_dataset, Features, 
                        Value, Sequence, DatasetDict, Dataset)

def preprocess(text):
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\\n', ' ', text) # remove newline character
    text = re.sub(r'\s+', ' ', text) # remove continuous spaces
    text = re.sub(r'#', ' ', text)

    return text


def preprocess_dataset(datasets):
    new_dataset = []
    for data in datasets:
        context = data['context']
        start_idx = data['answers']['answer_start'][0]

        before_start = context[:start_idx]
        after_start = context[start_idx:]

        processed_before = preprocess(before_start)
        processed_after = preprocess(after_start)
        processed_context = processed_before + processed_after
        moved_idx = len(before_start) - len(processed_before)

        data['context'] = processed_context
        data['answers']['answer_start'][0] = start_idx - moved_idx
        new_dataset.append(data)

    return new_dataset

def preprocess_wiki(dataset):
    new_wiki = {}
    for i in range(len(dataset)):
        key = str(i)
        context = dataset[key]['text']
        dataset[key]['text'] = preprocess(context)
        new_wiki[key] = dataset[key]

    return new_wiki

def create_processed_datasets(data_path='/opt/ml/data/'):
    train_features = Features({'answers': Sequence(feature={'text': Value(dtype='string', id=None), 'answer_start': Value(dtype='int32', id=None)}, length=-1, id=None),
                    'context': Value(dtype='string', id=None),
                    'id': Value(dtype='string', id=None),
                    'question': Value(dtype='string', id=None)})
    
    with open(os.path.join(data_path, 'wikipedia_documents.json'), "r", encoding="utf-8") as f:
            wiki = json.load(f)
    
    new_wiki = preprocess_wiki(wiki)
    with open('/opt/ml/data/preprocess_wiki.json', 'w', encoding='utf-8') as make_file:
            json.dump(new_wiki, make_file, indent="\t", ensure_ascii=False)


    train_dataset = load_from_disk(os.path.join(data_path, 'train_dataset/train'))
    validation_dataset = load_from_disk(os.path.join(data_path, 'train_dataset/validation'))

    new_train_data = pd.DataFrame(preprocess_dataset(train_dataset))
    new_validation_data = pd.DataFrame(preprocess_dataset(validation_dataset))

    Dataset.from_pandas(new_train_data, features=train_features).save_to_disk('/opt/ml/data/new_train_dataset/train')
    Dataset.from_pandas(new_validation_data, features=train_features).save_to_disk('/opt/ml/data/new_train_dataset/validation')

if __name__ == '__main__':
    create_processed_datasets()
