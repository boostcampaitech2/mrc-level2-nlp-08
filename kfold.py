import os

from datasets import load_from_disk, set_caching_enabled, Dataset
from datasets.dataset_dict import DatasetDict
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    DataCollatorWithPadding,
    HfArgumentParser,
    Trainer,
    set_seed,
    EarlyStoppingCallback,
)
import datasets
import wandb
import pandas as pd

from arguments import SettingsArguments, Arguments
from process import preprocess
from metric import compute_metrics
from utils import send_along
from models.lstm_roberta import LSTMRobertaForQuestionAnswering
from models.frozen_head import CustomModel
from models.double_roberta import DoubleRoberta
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold


def train(settings, args):
    args.config = AutoConfig.from_pretrained(settings.pretrained_model_name_or_path)
    args.tokenizer = AutoTokenizer.from_pretrained(
        settings.pretrained_model_name_or_path
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=args.tokenizer,
        pad_to_multiple_of=args.pad_to_multiple_of if args.fp16 else None,
    )

    train_dataset = load_from_disk(
        "../train_with_origin_gt_add_top_k_passage/not_include_answer_passage_train_es_top_4"
    )
    # valid_dataset = load_from_disk(
    #     "../valid_with_origin_gt_add_top_k_passage/not_include_answer_passage_valid_es_top_4"
    # )
    # concat_dataset = datasets.Dataset.from_pandas(
    #     pd.concat([pd.DataFrame(train_dataset), pd.DataFrame(valid_dataset)])
    # )
    concat_dataset = train_dataset

    kfold = KFold(n_splits=5)

    split_datasets = []
    for i, (td, vd) in enumerate(kfold.split(concat_dataset)):
        split_dataset = DatasetDict()
        split_dataset["train"] = Dataset.from_dict(concat_dataset[td])
        split_dataset["validation"] = Dataset.from_dict(concat_dataset[vd])
        split_datasets.append(split_dataset)

    for i, fold in enumerate(split_datasets):
        if i in [0, 1, 2]:
            continue
        model = AutoModelForQuestionAnswering.from_pretrained(
            settings.pretrained_model_name_or_path, config=args.config
        )
        # wandb.init(
        #     project="MRC_kfold", entity="chungye-mountain-sherpa", name=f"{i}_fold"
        # )

        args.dataset = fold
        args.output_dir = f"kfold_only_train/{i}_fold_train"

        train_dataset = args.dataset["train"]
        column_names = train_dataset.column_names
        train_dataset = train_dataset.map(
            send_along(preprocess, sent_along=args),
            batched=True,
            num_proc=settings.num_proc,
            remove_columns=column_names,
            load_from_cache_file=settings.load_from_cache_file,
        )

        eval_dataset = args.dataset["validation"]
        column_names = eval_dataset.column_names
        eval_dataset = eval_dataset.map(
            send_along(preprocess, sent_along=args),
            batched=True,
            num_proc=settings.num_proc,
            remove_columns=column_names,
            load_from_cache_file=settings.load_from_cache_file,
        )
        args.processed_eval_dataset = eval_dataset

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=args.tokenizer,
            data_collator=data_collator,
            compute_metrics=send_along(compute_metrics, sent_along=args),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )
        trainer.train()
        # model.save_pretrained(f"{args.output_dir}_{i}_fold")


if __name__ == "__main__":
    os.environ["WANDB_DISABLED"] = "true"

    parser = HfArgumentParser((SettingsArguments, Arguments))
    settings, args = parser.parse_args_into_dataclasses()
    set_seed(args.seed)
    set_caching_enabled(False)

    train(settings, args)
