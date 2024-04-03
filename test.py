from arithmetics import TaskPrompt, get_pa_model, PromptArithmeticsConfig
import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def sparsity(t):
    elements = t.numel()
    zero = elements - torch.count_nonzero(t.round())
    return 100 * zero / elements


model_name_or_path = "t5-base"
tokenizer_name_or_path = "t5-base"

origin_prompt = torch.load("soft_prompts/origin.bin")["prompt_embeddings"]
qnli_prompt = torch.load("soft_prompts/qnli.bin")["prompt_embeddings"]
mnli_prompt = torch.load("soft_prompts/mnli.bin")["prompt_embeddings"]
# print(origin_prompt, qnli_prompt, mnli_prompt)

qnli_diff = qnli_prompt - origin_prompt
mnli_diff = mnli_prompt - origin_prompt

qnli = TaskPrompt("qnli", task_weights=qnli_prompt, origin_weigts=origin_prompt)
mnli = TaskPrompt("mnli", task_weights=mnli_prompt, origin_weigts=origin_prompt)

assert (qnli_diff.numpy() == qnli.prompt.numpy()).all()
assert (mnli_diff.numpy() == mnli.prompt.numpy()).all()

qnli_mnli = qnli + mnli
assert ((mnli_diff + qnli_diff).numpy() == qnli_mnli.prompt.numpy()).all()

qnli_not_mnli = qnli - mnli
assert ((qnli_diff - mnli_diff).numpy() == qnli_not_mnli.prompt.numpy()).all()

print("Basic PA tests passed :)")

print(sparsity(mnli_diff), sparsity(qnli_diff))

tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_name_or_path, model_max_length=512, use_fast=True
)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
model.resize_token_embeddings(len(tokenizer))

pa_config = PromptArithmeticsConfig(
    num_virtual_tokens=50, origin_prompt="soft_prompts/origin.bin"
)
pa_model = get_pa_model(model=model, pa_config=pa_config)

print(pa_model.peft_model.prompt_encoder.default.embedding.weight)
