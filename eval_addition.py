from evaluator import ArithmeticsEvaluator
from args import TrainingArguments, DataTrainingArguments, ArgumentParser
from arithmetics import PromptArithmeticsConfig, get_pa_model, TaskPrompt
from tasks import Preprocessor

import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# import transformers
# transformers.logging.set_verbosity_debug()


parser = ArgumentParser(
    (TrainingArguments, DataTrainingArguments, PromptArithmeticsConfig)
)

training_args, data_args, pa_config = parser.parse_toml_file("configs/addition.toml")
print(training_args.device)

tokenizer = AutoTokenizer.from_pretrained(data_args.tokenizer_name_or_path, model_max_length=512, use_fast=True)

model = AutoModelForSeq2SeqLM.from_pretrained(training_args.model_name_or_path)
model.resize_token_embeddings(len(tokenizer))

pa_model = get_pa_model(model=model, pa_config=pa_config)

origin_prompt = torch.load(pa_config.origin_prompt)["prompt_embeddings"]
qnli_prompt = torch.load("soft_prompts/qnli.bin")["prompt_embeddings"]
mnli_prompt = torch.load("soft_prompts/mnli.bin")["prompt_embeddings"]

print("mnli_prompt:", mnli_prompt)
print("qnli_prompt:", qnli_prompt)
print("origin_prompt:", origin_prompt)

qnli = TaskPrompt("qnli", task_weights=qnli_prompt, origin_weigts=origin_prompt)
mnli = TaskPrompt("mnli", task_weights=mnli_prompt, origin_weigts=origin_prompt)

print(pa_model.peft_model.prompt_encoder.default.embedding.weight)

preprocessor = Preprocessor(["mnli", "qnli"], data_args, training_args)

_, _, test_datasets = preprocessor.get_data()

# from torch.utils.data import DataLoader
# from transformers import default_data_collator

# test_dataloaders = {
#     name: DataLoader(
#         test_datasets[name],
#         collate_fn=default_data_collator,
#         batch_size=32,
#         pin_memory=True,
#     )
#     for name in test_datasets
# }

# for i,b in enumerate(test_dataloaders["qnli"]):
#     print(b)
#     break

# mnli_qnli = mnli + qnli
# mnli_not_qnli = mnli - qnli
# qnli_not_mnli = qnli - mnli
# task_prompts = [mnli_qnli, mnli_not_qnli, qnli_not_mnli]

zeros = TaskPrompt("zeros", prompt=torch.zeros(size=origin_prompt.shape))
noise = TaskPrompt("noise", prompt=(100)*torch.randn(size=origin_prompt.shape))
task_prompts = [mnli + noise, qnli + noise]

# task_prompts = [mnli, qnli]

evaluator = ArithmeticsEvaluator(task_prompts=task_prompts, pa_model=pa_model, datasets=test_datasets, training_args=training_args, tokenizer=tokenizer)
evaluator.run()