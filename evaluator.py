from typing import List, Dict
from arithmetics import TaskPrompt, PromptArithmeticsModel
from args import TrainingArguments
from tasks import AutoTask

from transformers import Seq2SeqTrainer, default_data_collator, PreTrainedTokenizer

from datasets import Dataset

import functools

class ArithmeticsEvaluator:
    task_prompts: List[TaskPrompt] = None

    def __init__(
        self, task_prompts: List[TaskPrompt], pa_model: PromptArithmeticsModel, datasets: Dict[str, Dataset], training_args: TrainingArguments, tokenizer: PreTrainedTokenizer
    ):
        self.task_prompts = task_prompts
        self.pa_model = pa_model
        self.datasets = datasets
        self.training_args = training_args
        self.tokenizer = tokenizer

    def run(self):
        for tp in self.task_prompts:
            print(f"Evaluating task origin {tp.task_name}")
            self.pa_model.set_task(tp)

            print(tp.tasks, tp.task_name)

            for t in tp.tasks:
                trainer = Seq2SeqTrainer(
                    model=self.pa_model,
                    tokenizer=self.tokenizer,
                    train_dataset=self.datasets[t],
                    eval_dataset=self.datasets[t],
                    args=self.training_args,
                    data_collator=default_data_collator,
                    compute_metrics=functools.partial(AutoTask.get(t).compute_metrics, tokenizer=self.tokenizer),
                )

                print(self.datasets[t][0])
                trainer.evaluate(eval_dataset=self.datasets[t], metric_key_prefix="test")