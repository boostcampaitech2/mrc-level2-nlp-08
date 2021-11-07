import os

from datasets import Features, load_from_disk, Value, DatasetDict, Dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    set_seed,
)
from myseqtrainer import MySeq2SeqTrainer

from arguments import SettingsArguments, Seq2SeqArguments
from process import preprocess_testset_g

from metric import postprocess_g_test
from utils import send_along
from retrieval import SparseRetrieval


def inference(settings, args):
    args.config = AutoConfig.from_pretrained(settings.trained_model_path)
    args.tokenizer = AutoTokenizer.from_pretrained(settings.trained_model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(settings.trained_model_path)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=args.tokenizer, pad_to_multiple_of=args.pad_to_multiple_of if args.fp16 else None
    )
    args.dataset = load_from_disk(settings.testset_path)
    
    # retriever = SparseRetrieval(
    #     tokenize_fn=args.tokenizer, data_path="../../data", context_path="preprocess_wiki.json"
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

    eval_dataset = args.dataset["validation"]
    column_names = eval_dataset.column_names
    eval_dataset = eval_dataset.map(
        send_along(preprocess_testset_g, sent_along=args),
        batched=True,
        num_proc=settings.num_proc,
        remove_columns=column_names,
        load_from_cache_file=settings.load_from_cache_file,
    )
    args.processed_eval_dataset = eval_dataset

    trainer = MySeq2SeqTrainer(
        model=model,
        args=args,
        tokenizer=args.tokenizer,
        data_collator=data_collator,
    )
    postprocess_g_test(args, trainer.predict(test_dataset=eval_dataset, ignore_keys=["labels"]))


if __name__ == "__main__":
    os.environ["WANDB_DISABLED"] = "true"
    parser = HfArgumentParser((SettingsArguments, Seq2SeqArguments))
    settings, args = parser.parse_args_into_dataclasses()
    set_seed(args.seed)

    inference(settings, args)
