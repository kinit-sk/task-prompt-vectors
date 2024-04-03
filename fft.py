from evaluator import ArithmeticsEvaluator
from args import TrainingArguments, DataTrainingArguments, ArgumentParser
from tasks import Preprocessor

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    GenerationConfig,
    default_data_collator,
)

import wandb
import os


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


parser = ArgumentParser((TrainingArguments, DataTrainingArguments))

training_args, data_args = parser.parse_toml_file("configs/fft_qnli_mnli.toml")
print(training_args.device)
os.environ["WANDB_PROJECT"] = training_args.wandb_project

tokenizer = AutoTokenizer.from_pretrained(
    data_args.data_tokenizer_name_or_path, model_max_length=512, use_fast=True
)

model = AutoModelForSeq2SeqLM.from_pretrained(training_args.model_name_or_path)
model.resize_token_embeddings(len(tokenizer))

preprocessor = Preprocessor(["mnli", "qnli"], data_args, training_args)

train_dataset, valid_datasets, test_datasets = preprocessor.get_data()


trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_datasets,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,
)
trainer.train()

for td in test_datasets:
    trainer.evaluate(eval_dataset=test_datasets[td], metric_key_prefix="test")

if wandb.run is not None:
    wandb.finish()

model.save_pretrained("./mnli_qnli")
