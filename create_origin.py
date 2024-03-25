import torch
import wandb
import numpy as np
from peft import PromptTuningConfig, get_peft_model, PeftType, TaskType
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datetime import datetime

from safetensors import safe_open

seed = 1024
np.random.seed(seed=seed)

num_virtual_tokens = 50
model_name_or_path = "t5-base"
tokenizer_name_or_path = "t5-base"
origin_prompt_save = "./saves/origin"
name = f"origin_{datetime.now().strftime('%m%d%Y%H%M%S')}"

run = wandb.init(
    project="arithmetics",
    name=name,
    config={
        "num_virtual_tokens": num_virtual_tokens,
        "model_name_or_path": model_name_or_path,
        "tokenizer_name_or_path": tokenizer_name_or_path,
        "origin_prompt_save": origin_prompt_save,
        "prompt_init": "vocab",
        "seed": seed,
    },
)

tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_name_or_path, model_max_length=512, use_fast=True
)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
model.resize_token_embeddings(len(tokenizer))

peft_config = PromptTuningConfig(
    peft_type=PeftType.PROMPT_TUNING,
    task_type=TaskType.SEQ_2_SEQ_LM,
    num_virtual_tokens=num_virtual_tokens,
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

indices = np.random.permutation(range(5000))[:200]

word_embedding_weights = (
    model.word_embeddings(torch.LongTensor(indices)).detach().clone()
)
word_embedding_weights = word_embedding_weights.to(torch.float32)

model.prompt_encoder.default.embedding.weight = torch.nn.Parameter(
    word_embedding_weights
)
model.save_pretrained(origin_prompt_save)

origin_emb_weights = safe_open(f"./{origin_prompt_save}/adapter_model.safetensors", framework="pt", device="cpu").get_slice("prompt_embeddings")[:, :]
print(origin_emb_weights)
torch.save({"prompt_embeddings": origin_emb_weights}, f"./{origin_prompt_save}/origin.bin")

artifact = wandb.Artifact(name=name, type="weights")
artifact.add_dir(local_path=origin_prompt_save)
run.log_artifact(artifact)
run.finish()