from collections import OrderedDict
from typing import List, Dict, Any, Optional


class AbstractTaskType:
    formater = NotImplemented


class Seq2SeqLM(AbstractTaskType):
    def formater(
        self,
        task_name: str,
        inputs: List[str],
        labels: List[str],
        add_prefix: bool,
        prefix: Optional[str] = None,
    ):
        input_prefix = task_name if prefix is None else prefix
        inputs = [input_prefix] + inputs if add_prefix else inputs
        return {
            "source": " ".join(inputs),
            "target": " ".join(labels),
        }


class CausalLM(AbstractTaskType):
    def formater(
        self,
        task_name: str,
        inputs: List[str],
        labels: List[str],
        add_prefix: bool,
        prefix: Optional[str] = None,
        instruct=False,
        generation=False,
    ):
        if instruct:
            if generation:
                return {
                    "content": "\n".join(inputs) + "\n",
                    "target": " ".join(labels),
                    "role": "user",
                }

            return {
                "content": "\n".join(inputs) + "\nlabel: ",
                "target": " ".join(labels),
                "role": "user",
            }
        else:
            input_prefix = task_name if prefix is None else prefix
            inputs = [input_prefix] + inputs if add_prefix else inputs
            return {
                "source": f"{' '.join(inputs)} label: ",
                "target": " ".join(labels),
            }


TYPE_MAPPING: OrderedDict[str, AbstractTaskType] = OrderedDict(
    [
        ("SEQ_2_SEQ_LM", Seq2SeqLM),
        ("CAUSAL_LM", CausalLM),
    ]
)


class AutoType:
    @classmethod
    def get(self, task: str, eos_token: str = ""):
        if task in TYPE_MAPPING:
            return TYPE_MAPPING[task]()

        raise ValueError(
            f"Unrecognized task {task} for AutoType.\n"
            f"Task name should be one of {TYPE_MAPPING.keys()}."
        )
