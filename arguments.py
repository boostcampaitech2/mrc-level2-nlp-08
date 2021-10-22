from dataclasses import dataclass, field
from typing import Optional, Tuple

from transformers import TrainingArguments
from transformers.trainer_utils import IntervalStrategy


@dataclass
class SettingsArguments:
    pretrained_model_name_or_path: str = field(default="klue/roberta-large")
    trained_model_path: str = field(default="./outputs")
    trainset_path: str = field(default="../data/new_train_dataset")
    testset_path: str = field(default="../data/test_dataset")
    load_from_cache_file: bool = field(default=False)
    num_proc: Optional[int] = field(default=None)


@dataclass
class Arguments(TrainingArguments):
    per_device_train_batch_size: int = field(
        default=16,
        metadata={"help": "Batch size per GPU/TPU core/CPU for training."},
    )
    per_device_eval_batch_size: int = field(
        default=16,
        metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."},
    )
    gradient_accumulation_steps: int = field(
        default=8,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )
    learning_rate: float = field(
        default=6.819759978366989e-06,
        # default=3e-5,
        metadata={"help": "The initial learning rate for AdamW."},
    )
    weight_decay: float = field(
        default=0.17537006645417813,
        metadata={"help": "Weight decay for AdamW if we apply some."},
    )
    num_train_epochs: float = field(
        default=10.0, metadata={"help": "Total number of training epochs to perform."}
    )
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
    seed: int = field(
        default=21, metadata={"help": "Random seed that will be set at the beginning of training."}
    )
    do_train: bool = field(default=True, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=True, metadata={"help": "Whether to run eval on the dev set."})
    do_predict: bool = field(default=False, metadata={"help": "Whether to run predictions on the test set."})

    evaluation_strategy: IntervalStrategy = field(
        default="epoch",
        metadata={"help": "The evaluation strategy to use."},
    )

    logging_strategy: IntervalStrategy = field(
        default="epoch",
        metadata={"help": "The logging strategy to use."},
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

    label_names: Optional[Tuple[str]] = field(
        default=("start_positions", "end_positions"),
        metadata={"help": "The list of keys in your dictionary of inputs that correspond to the labels."},
    )
    load_best_model_at_end: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Whether or not to load the best model found during training at the end of training."
        },
    )
    metric_for_best_model: Optional[str] = field(
        default="f1", metadata={"help": "The metric to use to compare two different models."}
    )

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
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "The path to a folder with a valid checkpoint for your model."},
    )
    num_max_prediction: int = field(default=20)
    eval_retrieval: bool = field(
        default=True,
        metadata={"help": "Whether to run passage retrieval using sparse embedding."},
    )
    top_k_retrieval: int = field(
        default=20,
        metadata={"help": "Define how many top-k passages to retrieve based on similarity."},
    )
    use_faiss: bool = field(default=False, metadata={"help": "Whether to build with faiss"})
    num_clusters: int = field(default=5, metadata={"help": "Define how many clusters to use for faiss."})
