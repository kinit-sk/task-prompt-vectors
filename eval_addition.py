from evaluator import ArithmeticsEvaluator
from args import TrainingArguments, DataTrainingArguments, ArgumentParser
from arithmetics import PromptArithmeticsConfig, get_pa_model, TaskPrompt
from tasks import Preprocessor

import torch

parser = ArgumentParser(
    (TrainingArguments, DataTrainingArguments, PromptArithmeticsConfig)
)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

training_args, data_args, pa_config = parser.parse_toml_file("configs/addition.toml")
print(training_args.device)

tokenizer = AutoTokenizer.from_pretrained(data_args.tokenizer_name_or_path, model_max_length=512, use_fast=True)

model = AutoModelForSeq2SeqLM.from_pretrained(training_args.model_name_or_path)
model.resize_token_embeddings(len(tokenizer))

pa_model = get_pa_model(model=model, pa_config=pa_config)

origin_prompt = torch.load(pa_config.origin_prompt)["prompt_embeddings"]
qnli_prompt = torch.load("soft_prompts/qnli.bin")["prompt_embeddings"]
mnli_prompt = torch.load("soft_prompts/mnli.bin")["prompt_embeddings"]

qnli = TaskPrompt("qnli", task_weights=qnli_prompt, origin_weigts=origin_prompt)
mnli = TaskPrompt("mnli", task_weights=mnli_prompt, origin_weigts=origin_prompt)

print(pa_model.peft_model.prompt_encoder.default.embedding.weight)

preprocessor = Preprocessor(["mnli", "qnli"], data_args, training_args)

_, _, test_datasets = preprocessor.get_data()

print(test_datasets)

evaluator = ArithmeticsEvaluator(task_prompts=[mnli, qnli], pa_model=pa_model, datasets=test_datasets, training_args=training_args, tokenizer=tokenizer)
evaluator.run()