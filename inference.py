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
from Retrieval.retrieval import DenseRetrieval, HybridRetrieval
import pandas as pd
import pickle


def inference(settings, args):
    args.config = AutoConfig.from_pretrained(settings.trained_model_path)
    args.tokenizer = AutoTokenizer.from_pretrained(settings.trained_model_path)
    model = AutoModelForQuestionAnswering.from_pretrained(
        settings.trained_model_path)
    data_collator = DataCollatorWithPadding(
        tokenizer=args.tokenizer,
        pad_to_multiple_of=args.pad_to_multiple_of if args.fp16 else None,
    )
    args.dataset = load_from_disk(settings.testset_path)

    eval_dataset = args.dataset["validation"]
    hybrid_retrieval = HybridRetrieval(
        args.tokenizer, "p_encoder/", "q_encoder/")
    top_k_passage_ids, _ = hybrid_retrieval.get_topk_doc_id_and_score_for_querys(
        eval_dataset.to_pandas()["question"].to_list(), args.top_k_retrieval
    )

    args.dataset = run_dense_retrival(
        args.dataset,
        top_k_ids_dict=top_k_passage_ids,
        wiki_id_context_dict=hybrid_retrieval.wiki_id_context_dict,
        top_k=args.top_k_retrieval,
    )
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


def run_dense_retrival(eval_datasets, top_k_ids_dict, wiki_id_context_dict):
    question_texts = eval_datasets["validation"]["question"]
    total = []
    for i in range(len(eval_datasets["validation"]["id"])):
        texts = []
        for j in range(len(top_k_ids_dict[question_texts[i]])):
            texts.append(
                wiki_id_context_dict[top_k_ids_dict[question_texts[i]][j]])
        total.append(" ".join(texts))

    df = pd.DataFrame(
        data={
            "id": eval_datasets["validation"]["id"],
            "question": question_texts,
            "context": total,
        }
    )

    f = Features(
        {
            "context": Value(dtype="string", id=None),
            "id": Value(dtype="string", id=None),
            "question": Value(dtype="string", id=None),
        }
    )
    datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})

    return datasets


if __name__ == "__main__":
    os.environ["WANDB_DISABLED"] = "false"
    parser = HfArgumentParser((SettingsArguments, Arguments))
    settings, args = parser.parse_args_into_dataclasses()
    set_seed(args.seed)

    inference(settings, args)
