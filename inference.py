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


def inference(settings, args):
    args.config = AutoConfig.from_pretrained(settings.trained_model_path)
    args.tokenizer = AutoTokenizer.from_pretrained(settings.trained_model_path)
    model = AutoModelForQuestionAnswering.from_pretrained(settings.trained_model_path)
    data_collator = DataCollatorWithPadding(
        tokenizer=args.tokenizer, pad_to_multiple_of=args.pad_to_multiple_of if args.fp16 else None
    )
    args.dataset = load_from_disk(settings.testset_path)

    ##
    retriever = SparseRetrieval(
        tokenize_fn=args.tokenizer, data_path="../data", context_path="wikipedia_documents.json"
    )
    retriever.get_sparse_embedding()
    df = retriever.retrieve(args.dataset["validation"], topk=args.top_k_retrieval)
    f = Features(
        {
            "context": Value(dtype="string", id=None),
            "id": Value(dtype="string", id=None),
            "question": Value(dtype="string", id=None),
        }
    )
    args.dataset = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
    ##

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


if __name__ == "__main__":
    os.environ["WANDB_DISABLED"] = "true"
    parser = HfArgumentParser((SettingsArguments, Arguments))
    settings, args = parser.parse_args_into_dataclasses()
    set_seed(args.seed)

    inference(settings, args)
