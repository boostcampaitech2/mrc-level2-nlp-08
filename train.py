import os

from datasets import load_from_disk, set_caching_enabled
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    HfArgumentParser,
    set_seed,
    DataCollatorForSeq2Seq,
    EncoderDecoderModel,
    EncoderDecoderConfig,
)
from myseqtrainer import MySeq2SeqTrainer
from arguments import SettingsArguments, Seq2SeqArguments
from process import preprocess_g, preprocess_e

from metric import compute_metrics_g
from transformers.utils.dummy_pt_objects import AutoModel
import wandb
from utils import send_along


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
    trainer = MySeq2SeqTrainer(
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

def train_e(settings, args):
    args.encoder_config = AutoConfig.from_pretrained("klue/bert-base")
    args.decoder_config = AutoConfig.from_pretrained("klue/bert-base", is_decoder=True, add_cross_attention=True, decoder_start_token_id=2)
    args.tokenizer = AutoTokenizer.from_pretrained("klue/bert-base", padding=True)
    args.config = EncoderDecoderConfig.from_encoder_decoder_configs(args.encoder_config, args.decoder_config)
    model = EncoderDecoderModel(args.config)

    #print(model)
    # data_collator = DataCollatorForSeq2Seq(
    #     tokenizer=args.tokenizer, pad_to_multiple_of=args.pad_to_multiple_of if args.fp16 else None
    # )
    args.dataset = load_from_disk(settings.trainset_path)
    
    #train_dataset = args.dataset["train"]
    column_names = train_dataset.column_names
    train_dataset = train_dataset.map(
        send_along(preprocess_e, sent_along=args),
        batched=True,
        num_proc=settings.num_proc,
        remove_columns=column_names,
        load_from_cache_file=settings.load_from_cache_file,
    )

    eval_dataset = args.dataset["validation"]
    eval_dataset = eval_dataset.map(
        send_along(preprocess_e, sent_along=args),
        batched=True,
        num_proc=settings.num_proc,
        remove_columns=column_names,
        load_from_cache_file=settings.load_from_cache_file,
    )
    args.processed_eval_dataset = eval_dataset
    trainer = MySeq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=args.tokenizer,
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
    # os.environ["WANDB_DISABLED"] = "true"
    parser = HfArgumentParser((SettingsArguments, Seq2SeqArguments))
    settings, args = parser.parse_args_into_dataclasses()
    wandb.login()
    wandb.init(
        project="generative_qa",
        entity="chungye-mountain-sherpa",
        name='len256',
        group='kobart with concat',
    )

    set_seed(args.seed)
    print(args)
    train_g(settings, args)
