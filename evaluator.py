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

import pandas as pd
import numpy as np

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
        test_datasets: Dict[str, Dataset],
        eval_datasets: Dict[str, Dataset],
        training_args: TrainingArguments,
        tokenizer: PreTrainedTokenizer,
        origin_weights: torch.Tensor,
    ):
        self.task_prompts = task_prompts
        self.model = model
        self.test_datasets = test_datasets
        self.eval_datasets = eval_datasets
        self.training_args = training_args
        self.tokenizer = tokenizer
        self.origin_weights = origin_weights
        self.results = []
        self.orig_run_name = self.training_args.run_name
        self.scaling_coefs = np.arange(0.0, 1.05, 0.05)

    def set_task(self, task_prompt: TaskPrompt, coef: float = 1):
        self.model.prompt_encoder.default.embedding.weight = task_prompt.apply(self.origin_weights, coef)

    def run(self):
        for tp in self.task_prompts:
            self.training_args.run_name = f"{self.orig_run_name}{tp.task_name.replace(' ', '')}"

            print(f"Evaluating task origin {tp.task_name}")

            print(f"Evaluating on scaling coefs: {self.scaling_coefs}")

            best_coef = 1
            best_acc = 0
            if len(tp.tasks) > 1:
                for coef in self.scaling_coefs:
                    eval_acc = []
                    for dataset_name in tp.tasks:
                        print(dataset_name, coef)
                        
                        self.set_task(tp, coef=coef)
                        print(f"Current PT weights: {self.model.prompt_encoder.default.embedding.weight}")
                        
                        trainer = Seq2SeqTrainer(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        args=self.training_args,
                        data_collator=default_data_collator,
                        compute_metrics=compute_metrics,
                        )

                        eval_res = trainer.evaluate(eval_dataset=self.eval_datasets[dataset_name], metric_key_prefix=f"eval_{dataset_name}")
                        print(eval_res)

                        eval_acc.append(eval_res[f"eval_{dataset_name}_accuracy"])
                    
                    mean_acc = torch.tensor(eval_acc).mean()
                    if mean_acc > best_acc:
                        print(f"New best mean acc: {mean_acc} with coef: {coef}")
                        best_acc = mean_acc
                        best_coef = coef
                    
            print(f"Testing with best coef: {best_coef}")
            for dataset_name in tp.tasks:
                print(dataset_name, best_coef)
                
                self.set_task(tp, coef=best_coef)
                print(f"Current PT weights: {self.model.prompt_encoder.default.embedding.weight}")
                
                trainer = Seq2SeqTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                args=self.training_args,
                data_collator=default_data_collator,
                compute_metrics=compute_metrics,
                )

                test_res = trainer.evaluate(eval_dataset=self.test_datasets[dataset_name], metric_key_prefix=f"test_{dataset_name}")
                print(test_res)

                self.results.append({"tasks":  " ".join(tp.tasks), f"{dataset_name}_accuracy": test_res[f"test_{dataset_name}_accuracy"]})

            if wandb.run:
                wandb.finish()

        df_results = pd.DataFrame.from_dict(self.results)
        df_results = df_results.groupby(["tasks"], as_index=False).first()

        return df_results