import torch
import wandb
import numpy as np
from peft import PromptTuningConfig, get_peft_model, PeftType, TaskType
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer
from datetime import datetime

import argparse

from safetensors import safe_open

seed = 1024
np.random.seed(seed=seed)

n_origins = 10

argparse_parser = argparse.ArgumentParser(
    prog="Generate rarndomly intialized soft-prompts",
    description="Generate rarndomly intialized soft-prompts that can be later used to initialize prompt-tuning.",
)

argparse_parser.add_argument(
    "model_name_or_path", help="Model name to generate soft-prompts for."
)
args = argparse_parser.parse_args()

model_name_or_path = args.model_name_or_path
tokenizer_name_or_path = args.model_name_or_path

tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_name_or_path, model_max_length=512, use_fast=True
)

if "llama" in model_name_or_path.lower():
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16).to("cuda")
else:
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)

for i in range(n_origins):
    num_virtual_tokens = 50

    origin_prompt_save = (
        f"./saves/origin_{i}_{model_name_or_path.lower().split('/')[-1]}"
    )
    name = f"origin_{datetime.now().strftime('%m%d%Y%H%M%S')}_{i}_{model_name_or_path.lower().split('/')[-1]}"

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

    peft_config = PromptTuningConfig(
        peft_type=PeftType.PROMPT_TUNING,
        task_type=TaskType.SEQ_2_SEQ_LM,
        num_virtual_tokens=num_virtual_tokens,
    )
    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters()

    indices = np.random.permutation(range(5000))[:200]

    word_embedding_weights = (
        peft_model.word_embeddings(torch.LongTensor(indices).to("cuda")).detach().clone()
    )
    word_embedding_weights = word_embedding_weights.to(torch.bfloat16)

    peft_model.prompt_encoder.default.embedding.weight = torch.nn.Parameter(
        word_embedding_weights
    )
    peft_model.save_pretrained(origin_prompt_save)

    origin_emb_weights = safe_open(
        f"./{origin_prompt_save}/adapter_model.safetensors",
        framework="pt",
        device="cpu",
    ).get_slice("prompt_embeddings")[:, :]
    print(origin_emb_weights)
    torch.save(
        {"prompt_embeddings": origin_emb_weights},
        f"./{origin_prompt_save}/origin_{i}_{model_name_or_path.lower().split('/')[-1]}.bin",
    )

    artifact = wandb.Artifact(name=name, type="weights")
    artifact.add_dir(local_path=origin_prompt_save)
    run.log_artifact(artifact)
    wandb.log(data={})
    run.finish()
