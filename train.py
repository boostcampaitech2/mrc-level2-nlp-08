import os

from datasets import load_from_disk
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
from process import preprocess

from metric import compute_metrics
from utils import send_along
from model import LSTMRobertaForQuestionAnswering


def train(settings, args):
    args.config = AutoConfig.from_pretrained(settings.pretrained_model_name_or_path)
    args.tokenizer = AutoTokenizer.from_pretrained(settings.pretrained_model_name_or_path)
    # model = AutoModelForQuestionAnswering.from_pretrained(settings.pretrained_model_name_or_path)
    model = LSTMRobertaForQuestionAnswering(
        model_name_or_path=settings.pretrained_model_name_or_path,
        config=args.config,
    )
    data_collator = DataCollatorWithPadding(
        tokenizer=args.tokenizer, pad_to_multiple_of=args.pad_to_multiple_of if args.fp16 else None
    )
    args.dataset = load_from_disk(settings.trainset_path)

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
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model()
    trainer.evaluate()


if __name__ == "__main__":
    os.environ["WANDB_DISABLED"] = "true"
    parser = HfArgumentParser((SettingsArguments, Arguments))
    settings, args = parser.parse_args_into_dataclasses()
    set_seed(args.seed)

    train(settings, args)
