from transformers import Seq2SeqTrainingArguments
from dataclasses import dataclass, field

from typing import Optional, List


@dataclass
class TrainingArguments(Seq2SeqTrainingArguments):
    do_test: Optional[bool] = field(
        default=False, metadata={"help": "If set, evaluates the test performance."}
    )

    per_device_test_batch_size: Optional[int] = field(
        default=8, metadata={"help": "Specify test batch size."}
    )

    model_name_or_path: Optional[str] = field(
        default="t5-base",
        metadata={"help": "Specify model to use while train/test/eval."},
    )

    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": "Specify wandb project."},
    )

    train_dataset_names: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Specify train dataset name or names."},
    )

    origin_prompt_name: Optional[str] = field(
        default=None,
        metadata={"help": "Specify origin prompt name or names."},
    )
