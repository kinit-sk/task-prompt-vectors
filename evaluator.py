from typing import List, Dict
from arithmetics import TaskPrompt, PromptArithmeticsModel
from args import TrainingArguments
from tasks import AutoTask

from transformers import (
    Seq2SeqTrainer,
    default_data_collator,
    PreTrainedTokenizer,
    AutoTokenizer,
)

from datasets import Dataset

import functools

import wandb


def compute_metrics(eval_preds):

    tokenizer = AutoTokenizer.from_pretrained(
        "t5-base", model_max_length=512, use_fast=True
    )
    preds, labels = eval_preds
    # print(tokenizer.pad_token_id)

    preds[preds == -100] = tokenizer.pad_token_id
    labels[labels == -100] = tokenizer.pad_token_id

    # print(preds, labels)
    preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # print(preds, labels)

    correct = 0
    total = 0
    for pred, true in zip(preds, labels):
        if pred.strip() == true.strip():
            correct += 1
        total += 1
    accuracy = correct / total
    return {"accuracy": accuracy}


class ArithmeticsEvaluator:
    task_prompts: List[TaskPrompt] = None

    def __init__(
        self,
        task_prompts: List[TaskPrompt],
        pa_model: PromptArithmeticsModel,
        datasets: Dict[str, Dataset],
        training_args: TrainingArguments,
        tokenizer: PreTrainedTokenizer,
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

            print(
                "current PT weights:",
                self.pa_model.peft_model.prompt_encoder.default.embedding.weight,
            )

            for t in ["mnli", "qnli"]:
                print(t)
                trainer = Seq2SeqTrainer(
                    model=self.pa_model.peft_model,
                    tokenizer=self.tokenizer,
                    args=self.training_args,
                    data_collator=default_data_collator,
                    compute_metrics=compute_metrics,
                )

                print(
                    trainer.evaluate(
                        eval_dataset=self.datasets[t], metric_key_prefix="test"
                    )
                )

                if wandb.run:
                    wandb.finish()
