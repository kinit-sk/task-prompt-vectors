from collections import OrderedDict
from typing import List, Dict, Any, Optional


class AbstractTaskType:
    formater: function = NotImplemented


class Seq2SeqLM(AbstractTaskType):
    def formater(
        self,
        task_name: str,
        inputs: List[str],
        labels: List[str],
        add_prefix: bool,
        prefix: Optional[str] = None,
        extra_fields: Dict[str, Any] = {},
    ):
        input_prefix = task_name if prefix is None else prefix
        inputs = [input_prefix] + inputs if add_prefix else inputs
        return {
            "source": " ".join(inputs),
            "target": " ".join(labels),
            "task": task_name,
            "extra_fields": extra_fields,
        }


TYPE_MAPPING: OrderedDict[str, AbstractTaskType] = OrderedDict(
    [("seq_2_seq_lm", Seq2SeqLM)]
)


class AutoType:
    @classmethod
    def get(self, task: str):
        if task in TYPE_MAPPING:
            return TYPE_MAPPING[task]()

        raise ValueError(
            f"Unrecognized task {task} for AutoType.\n"
            f"Task name should be one of {TYPE_MAPPING.keys()}."
        )
