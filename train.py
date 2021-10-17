import logging
import os
import sys

from typing import List, Callable, NoReturn, NewType, Any
import dataclasses
from datasets import load_metric, load_from_disk, Dataset, DatasetDict

from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer

from transformers import (
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from tokenizers import Tokenizer
from tokenizers.models import WordPiece

from utils_qa import find_last_checkpoint, postprocess_qa_predictions, check_no_error
from trainer_qa import QuestionAnsweringTrainer
from retrieval import SparseRetrieval

from arguments import ModelArguments, DataTrainingArguments, MyTrainingArguments

import wandb
import torch
import torch.nn as nn

from preprocess import make_datasets

logger = logging.getLogger(__name__)


def main(args):
    # logging 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    train(args)


def compute_metrics(p: EvalPrediction):
    metric = load_metric("squad")
    return metric.compute(predictions=p.predictions, references=p.label_ids)


def train(args):
    model_args, data_args, training_args = args
    set_seed(training_args.seed)

    logger.info("Training/evaluation parameters %s", training_args)

    datasets = load_from_disk(data_args.dataset_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name is not None else model_args.model_name_or_path,
        use_fast=True,
    )

    check_no_error(datasets, tokenizer)

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)
    train_dataset, eval_dataset = make_datasets(datasets, tokenizer, training_args, data_args)

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name is not None else model_args.model_name_or_path,
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
    )

    last_checkpoint = find_last_checkpoint(training_args)

    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        data_args=data_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        eval_examples=datasets["validation"] if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    if last_checkpoint is not None:
        checkpoint = last_checkpoint
    elif os.path.isdir(model_args.model_name_or_path):
        checkpoint = model_args.model_name_or_path
    else:
        checkpoint = None
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    output_train_file = os.path.join(training_args.output_dir, "train_results.txt")

    with open(output_train_file, "w") as writer:
        logger.info("***** Train results *****")
        for key, value in sorted(train_result.metrics.items()):
            logger.info(f"  {key} = {value}")
            writer.write(f"{key} = {value}\n")

    # State 저장
    trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

    # Evaluation
    logger.info("*** Evaluate ***")
    metrics = trainer.evaluate()

    metrics["eval_samples"] = len(eval_dataset)

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


if __name__ == "__main__":

    os.environ["WANDB_DISABLED"] = "true"
    # wandb.init(
    #     project="MRC_baseline",
    #     entity="chungye-mountain-sherpa",
    #     name="default settings",
    #     group=model_args.model_name_or_path
    # )

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, MyTrainingArguments))

    args = parser.parse_args_into_dataclasses()

    main(args)
