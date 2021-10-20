import os

from datasets import load_from_disk
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    DataCollatorWithPadding,
    HfArgumentParser,
    Trainer,
    set_seed,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
)

from arguments import SettingsArguments, Arguments, Seq2SeqArguments
from process import preprocess, preprocess_g

from metric import compute_metrics, compute_metrics_g
from utils import send_along


def train(settings, args):
    args.config = AutoConfig.from_pretrained(settings.pretrained_model_name_or_path)
    args.tokenizer = AutoTokenizer.from_pretrained(settings.pretrained_model_name_or_path)
    model = AutoModelForQuestionAnswering.from_pretrained(
        settings.pretrained_model_name_or_path, config=args.config
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


def train_g(settings, args):
    args.config = AutoConfig.from_pretrained(settings.pretrained_model_name_or_path)
    args.tokenizer = AutoTokenizer.from_pretrained(settings.pretrained_model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        settings.pretrained_model_name_or_path, config=args.config
    )
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=args.tokenizer, pad_to_multiple_of=args.pad_to_multiple_of if args.fp16 else None
    )
    args.dataset = load_from_disk(settings.trainset_path)

    train_dataset = args.dataset["train"]
    column_names = train_dataset.column_names
    train_dataset = train_dataset.map(
        send_along(preprocess_g, sent_along=args),
        batched=True,
        num_proc=settings.num_proc,
        remove_columns=column_names,
        load_from_cache_file=settings.load_from_cache_file,
    )

    eval_dataset = args.dataset["validation"]
    eval_dataset = eval_dataset.map(
        send_along(preprocess_g, sent_along=args),
        batched=True,
        num_proc=settings.num_proc,
        remove_columns=column_names,
        load_from_cache_file=settings.load_from_cache_file,
    )
    args.processed_eval_dataset = eval_dataset

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=args.tokenizer,
        data_collator=data_collator,
        compute_metrics=send_along(compute_metrics_g, sent_along=args),
    )
    trainer.train()
    trainer.save_model()
    trainer.evaluate(
        max_length=args.max_answer_length,
        num_beams=args.num_beams,
        metric_key_prefix="eval"
    )


if __name__ == "__main__":
    os.environ["WANDB_DISABLED"] = "true"
    parser = HfArgumentParser((SettingsArguments, Seq2SeqArguments))
    settings, seqargs = parser.parse_args_into_dataclasses()
    if settings.extractive:
        pass
    else:
        set_seed(seqargs.seed)
        print(seqargs)
        train_g(settings, seqargs)