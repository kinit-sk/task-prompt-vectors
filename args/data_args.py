from dataclasses import dataclass, field

from typing import Optional, List


@dataclass
class DataTrainingArguments:
    dataset_names: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "The name of the test dataset to use (via the datasets library)."
        },
    )

    max_source_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )

    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )

    data_tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Specify tokenizer to use while train/test/eval."},
    )

    split_validation_test: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If set, for the datasets which do not"
            "have the test set, we use validation set as their"
            "test set and make a validation set from either"
            "splitting the validation set into half (for smaller"
            "than 10K samples datasets), or by using 1K examples"
            "from training set as validation set (for larger"
            " datasets)."
        },
    )

    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )

    max_valid_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )

    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
            "value if set."
        },
    )

    pad_to_max_length: Optional[bool] = field(
        default=False, metadata={"help": "Pad labels to the size of the input leght"}
    )
