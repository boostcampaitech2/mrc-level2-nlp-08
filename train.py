import os

import datasets
from datasets import load_from_disk, set_caching_enabled
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    DataCollatorWithPadding,
    HfArgumentParser,
    Trainer,
    set_seed,
    BigBirdForQuestionAnswering,
    BigBirdConfig,
)
from sentence_transformers import SentenceTransformer
import wandb
import pandas as pd
from arguments import SettingsArguments, Arguments
from process import preprocess
from metric import compute_metrics
from utils import send_along
from models.lstm_roberta import LSTMRobertaForQuestionAnswering
from models.cnn_head import Conv1DRobertaForQuestionAnswering
from models.frozen_head import CustomModel
from models.combine_electra import RobertaElectra
from models.custom import MyModel


def train(settings, args):
    args.config = AutoConfig.from_pretrained(settings.pretrained_model_name_or_path)
    args.tokenizer = AutoTokenizer.from_pretrained(settings.pretrained_model_name_or_path)
    model = AutoModelForQuestionAnswering.from_pretrained(
        settings.pretrained_model_name_or_path,
        config=args.config,
    )
    # model = FiD(settings.pretrained_model_name_or_path, config=args.config)
    # model = LSTMRobertaForQuestionAnswering(settings.pretrained_model_name_or_path, config=args.config)
    # model = Conv1DRobertaForQuestionAnswering(settings.pretrained_model_name_or_path, config=args.config)
    # model = CustomModel(settings.pretrained_model_name_or_path, config=args.config)
    # model = RobertaElectra(settings.pretrained_model_name_or_path, config=args.config)
    # model = MyModel(settings.pretrained_model_name_or_path, config=args.config, tokenizer=args.tokenizer)

    data_collator = DataCollatorWithPadding(tokenizer=args.tokenizer, pad_to_multiple_of=8)
    # data_collator = DataCollatorWithPadding(
    #     tokenizer=args.tokenizer, pad_to_multiple_of=args.pad_to_multiple_of if args.fp16 else None
    # )
    args.dataset = load_from_disk(settings.trainset_path)
    # args.dataset = load_from_disk("../split_dataset")
    # train_dataset = args.dataset["train"]

    train_dataset = load_from_disk(
        # "../qg_title",
        # "../sentence_shuffle",
        "../train_with_origin_gt_add_top_k_passage/not_include_answer_passage_train_es_top_4"
        # "../shuffle_sentence_concat"
        # "../not_include_answer_passage_train_es_top_7"
        # "../qg_title_concat_4"
    )
    # t0 = load_from_disk("../passage_shuffle/concat_train_gt_0")
    # t1 = load_from_disk("../passage_shuffle/concat_train_gt_1")
    # t2 = load_from_disk("../passage_shuffle/concat_train_gt_2")
    # t3 = load_from_disk("../passage_shuffle/concat_train_gt_3")
    # t4 = load_from_disk("../passage_shuffle/concat_train_gt_4")
    # t5 = load_from_disk("../passage_random_shuffle")
    # sh_s = load_from_disk("/home/sentence_shuffle")
    # sh_s = load_from_disk("/home/qg_title")
    # sh_t = load_from_disk("/home/sentence_shuffle")
    # train_dataset = datasets.Dataset.from_pandas(pd.concat([pd.DataFrame(train_dataset), pd.DataFrame(sh_s)]))
    # train_dataset = datasets.Dataset.from_pandas(
    #     pd.concat([pd.DataFrame(train_dataset), pd.DataFrame(sh_s), pd.DataFrame(sh_t)])
    # )
    # train_dataset = datasets.concatenate_datasets([t0, t1, t2, t3, t4, t5])
    # train_dataset = load_from_disk("../train_es_top_20")  # 15542
    # train_dataset = load_from_disk(
    #     "../train_no_include_answer_concats/not_include_answer_passage_train_es_top_15"
    # )

    # args.sroberta = SentenceTransformer("Huffon/sentence-klue-roberta-base")

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
    )
    trainer.train()
    trainer.save_model()
    # print(trainer.evaluate())


if __name__ == "__main__":
    os.environ["WANDB_DISABLED"] = "true"

    # wandb.init(
    #     project="MRC_aeda",
    #     entity="chungye-mountain-sherpa",
    #     # name="topk_5_with_lstm_layers",
    #     name="sota_frozen_weight_train_only_head_shuffle_sentence_concat",
    #     group="shuffle_sentence",
    # )

    parser = HfArgumentParser((SettingsArguments, Arguments))
    settings, args = parser.parse_args_into_dataclasses()
    set_seed(args.seed)
    set_caching_enabled(False)

    train(settings, args)
