<<<<<<< HEAD
from dataclasses import dataclass, field
from importlib.metadata import metadata
from typing import Optional

from transformers import TrainingArguments
from transformers.trainer_utils import IntervalStrategy


@dataclass
class SettingsArguments:
    pretrained_model_name_or_path: str = field(default="klue/bert-base")
    trainset_path: str = field(default="../data/train_dataset")
    testset_path: str = field(default="../data/test_dataset")
    load_from_cache_file: bool = field(default=True)
    num_proc: Optional[int] = field(default=None)


@dataclass
class Arguments(TrainingArguments):
    output_dir: str = field(
        default="output",
        metadata={
            "help": "The output directory where the model predictions and checkpoints will be written."
        },
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory."
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )

    do_train: bool = field(default=True, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=True, metadata={"help": "Whether to run eval on the dev set."})
    evaluation_strategy: IntervalStrategy = field(
        default="epoch",
        metadata={"help": "The evaluation strategy to use."},
    )

    per_device_train_batch_size: int = field(
        default=16, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=16, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )

    num_train_epochs: float = field(
        default=0.1, metadata={"help": "Total number of training epochs to perform."}
    )

    save_strategy: IntervalStrategy = field(
        default="epoch",
        metadata={"help": "The checkpoint save strategy to use."},
    )
    save_total_limit: Optional[int] = field(
        default=1,
        metadata={
            "help": (
                "Limit the total amount of checkpoints."
                "Deletes the older checkpoints in the output_dir. Unlimited checkpoints if 'None'"
            )
        },
    )

    fp16: bool = field(
        default=True,
        metadata={"help": "Whether to use 16-bit (mixed) precision instead of 32-bit"},
    )
    pad_to_multiple_of: int = field(default=8, metadata={"help": "Pad to multiple of set number"})

    max_length: Optional[int] = field(default=384)
    stride: int = field(
        default=128,
        metadata={"help": "The stride to use when handling overflow."},
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an answer tokens that can be generated."
            "This is needed because the start and end predictions are not conditioned on one another."
        },
    )
    num_max_prediction: int = field(default=20)
    eval_retrieval: bool = field(
        default=True,
        metadata={"help": "Whether to run passage retrieval using sparse embedding."},
    )
    top_k_retrieval: int = field(
        default=1,
        metadata={"help": "Define how many top-k passages to retrieve based on similarity."},
    )
    use_faiss: bool = field(default=False, metadata={"help": "Whether to build with faiss"})
    num_clusters: int = field(default=64, metadata={"help": "Define how many clusters to use for faiss."})
=======
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="klue/roberta-large",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default="../data/train_dataset",
        metadata={"help": "The name of the dataset to use."},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
            "be faster on GPU but will be slower on TPU)."
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={
            "help": "When splitting up a long document into chunks, how much stride to take between chunks."
        },
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        },
    )
    eval_retrieval: bool = field(
        default=True,
        metadata={"help": "Whether to run passage retrieval using sparse embedding."},
    )
    num_clusters: int = field(
        default=64, metadata={"help": "Define how many clusters to use for faiss."}
    )
    top_k_retrieval: int = field(
        default=1,
        metadata={
            "help": "Define how many top-k passages to retrieve based on similarity."
        },
    )
    use_faiss: bool = field(
        default=False, metadata={"help": "Whether to build with faiss"}
    )
>>>>>>> 65df02b... feat: baseline
