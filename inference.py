import os

from datasets import Features, load_from_disk, Value, DatasetDict, Dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    DataCollatorWithPadding,
    HfArgumentParser,
    Trainer,
    set_seed,
)

from arguments import SettingsArguments, Arguments
from process import preprocess_testset

from metric import postprocess
from utils import send_along
from retrieval import SparseRetrieval
import pandas as pd
import pickle


def inference(settings, args):
    args.config = AutoConfig.from_pretrained(settings.trained_model_path)
    args.tokenizer = AutoTokenizer.from_pretrained(settings.trained_model_path)
    model = AutoModelForQuestionAnswering.from_pretrained(settings.trained_model_path)
    data_collator = DataCollatorWithPadding(
        tokenizer=args.tokenizer,
        pad_to_multiple_of=args.pad_to_multiple_of if args.fp16 else None,
    )
    args.dataset = load_from_disk(settings.testset_path)

    ##
    # retriever = SparseRetrieval(
    #     tokenize_fn=args.tokenizer,
    #     data_path="../data",
    #     context_path="wikipedia_documents.json",
    # )
    # retriever.get_sparse_embedding()
    # df = retriever.retrieve(args.dataset["validation"], topk=args.top_k_retrieval)
    # f = Features(
    #     {
    #         "context": Value(dtype="string", id=None),
    #         "id": Value(dtype="string", id=None),
    #         "question": Value(dtype="string", id=None),
    #     }
    # )
    # args.dataset = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
    ##
    args.dataset = run_dense_retrival(args.dataset, args.top_k_retrieval)

    eval_dataset = args.dataset["validation"]
    column_names = eval_dataset.column_names
    eval_dataset = eval_dataset.map(
        send_along(preprocess_testset, sent_along=args),
        batched=True,
        num_proc=settings.num_proc,
        remove_columns=column_names,
        load_from_cache_file=settings.load_from_cache_file,
    )
    args.processed_eval_dataset = eval_dataset

    trainer = Trainer(
        model=model,
        args=args,
        tokenizer=args.tokenizer,
        data_collator=data_collator,
    )
    postprocess(args, trainer.predict(test_dataset=eval_dataset))


def run_dense_retrival(text_data, top_k):
    test_data = get_pickle("/opt/ml/data/hybrid_retrieval.bin")
    question_texts = text_data["validation"]["question"]

    total = []
    for i in range(len(text_data["validation"]["id"])):
        total.append(" ".join(test_data[question_texts[i]][:top_k]))
    df = pd.DataFrame(
        data={
            "id": text_data["validation"]["id"],
            "question": question_texts,
            "context": total,
        }
    )
    df.to_csv("dense.csv")

    f = Features(
        {
            "context": Value(dtype="string", id=None),
            "id": Value(dtype="string", id=None),
            "question": Value(dtype="string", id=None),
        }
    )
    datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})

    return datasets


def get_pickle(pickle_path):
    f = open(pickle_path, "rb")
    dataset = pickle.load(f)
    f.close()
    return dataset


if __name__ == "__main__":
    os.environ["WANDB_DISABLED"] = "true"
    parser = HfArgumentParser((SettingsArguments, Arguments))
    settings, args = parser.parse_args_into_dataclasses()
    set_seed(args.seed)

    inference(settings, args)
