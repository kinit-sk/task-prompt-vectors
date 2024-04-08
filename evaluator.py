from typing import List, Dict
from arithmetics import TaskPrompt, PromptArithmeticsModel
from args import TrainingArguments

from transformers import (
    Seq2SeqTrainer,
    default_data_collator,
    PreTrainedTokenizer,
    AutoTokenizer,
)

from datasets import Dataset

import wandb
import torch

from peft import PeftModel

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
        model: PeftModel,
        datasets: Dict[str, Dataset],
        training_args: TrainingArguments,
        tokenizer: PreTrainedTokenizer,
    ):
        self.task_prompts = task_prompts
        self.model = model
        self.datasets = datasets
        self.training_args = training_args
        self.tokenizer = tokenizer

    def set_task(self, task_prompt: TaskPrompt):
        self.model.prompt_encoder.default.embedding.weight = task_prompt.apply()

    def run(self):
        for tp in self.task_prompts:
            print(f"Evaluating task origin {tp.task_name}")
            self.set_task(tp)

            print(
                "current PT weights:",
                self.model.peft_model.prompt_encoder.default.embedding.weight,
            )

            for dataset_name in self.datasets:
                print(dataset_name)
                trainer = Seq2SeqTrainer(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    args=self.training_args,
                    data_collator=default_data_collator,
                    compute_metrics=compute_metrics,
                )

                print(
                    trainer.evaluate(
                        eval_dataset=self.datasets[dataset_name], metric_key_prefix="test"
                    )
                )

                if wandb.run:
                    wandb.finish()
