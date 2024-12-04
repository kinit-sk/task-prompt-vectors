import argparse
import os
import torch
import wandb
import numpy as np

from args import DataTrainingArguments, ArgumentParser
from arithmetics import PromptArithmeticsConfig
from tasks import AutoTask

from datetime import datetime
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    AutoModelForSeq2SeqLM,
)
from peft import get_peft_model, PromptTuningConfig
from trl import SFTTrainer, SFTConfig, ModelConfig

from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

from metrics.utils import binary_reverse

from itertools import combinations


configs = [
    "configs/prompt_tuning/single-task/prompt_tuning.toml",
    "configs/prompt_tuning/single-task/llama31_8b_instruct.toml",
]


def generate_model_name(components):
    # Map for abbreviating components
    abbreviations = {
        "prompt tuning": "pt",
        "origin_0_meta": "origin0",
        "origin_1_meta": "origin1",
        "origin_2_meta": "origin2",
        "origin_0": "origin0",
        "origin_1": "origin1",
        "origin_2": "origin2",
        "origin_3": "origin3",
        "origin_4": "origin4",
        "origin_5": "origin5",
        "origin_6": "origin6",
        "origin_7": "origin7",
        "origin_8": "origin8",
        "origin_9": "origin9",
        "llama-3.1-8b-instruct": "llama3.1-8b",
        "t5-base": "t5-base",
        "dbpedia_text": "dbpedia",
        "mnli_text": "mnli",
        "qnli_text": "qnli",
        "sst2_text": "sst2",
        "trec_coarse_text": "trec",
        "yelp_polarity_text": "yelp",
        "dbpedia_text_instruct": "dbpedia",
        "mnli_text_instruct": "mnli",
        "qnli_text_instruct": "qnli",
        "sst2_text_instruct": "sst2",
        "trec_coarse_text_instruct": "trec",
        "yelp_polarity_text_instruct": "yelp",
    }

    # Abbreviate and combine components
    short_name_parts = []
    for component in components:
        for key, abbrev in abbreviations.items():
            if key in component.lower():
                if abbrev not in short_name_parts:
                    short_name_parts.append(abbrev)

    # Join the parts with hyphens
    return "-".join(short_name_parts)


for c in configs:
    parser = ArgumentParser(
        (SFTConfig, ModelConfig, DataTrainingArguments, PromptArithmeticsConfig)
    )

    training_args, model_args, data_args, peft_config = parser.parse_toml_file(c)

    print(model_args.model_name_or_path)

    if peft_config.task_type == "CAUSAL_LM":
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=torch.bfloat16,
        ).to("cuda")
        model.active_adapters = [
            "default"
        ]  # fix because llama has some active adapters for some reason
        model = get_peft_model(model, peft_config=peft_config)

        tokenizer = AutoTokenizer.from_pretrained(
            data_args.data_tokenizer_name_or_path,
            trust_remote_code=True,
            padding_side="right",
        )
        tokenizer.add_special_tokens({"pad_token": "<|reserved_special_token_0|>"})
        model.config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    elif peft_config.task_type == "SEQ_2_SEQ_LM":

        model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name_or_path).to(
            "cuda"
        )

        model.generation_config.max_new_tokens = 16
        model.generation_config.max_length = 256

        model = get_peft_model(model, peft_config=peft_config)

        tokenizer = AutoTokenizer.from_pretrained(
            data_args.data_tokenizer_name_or_path, model_max_length=512, use_fast=True
        )

        model.config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    # config_dict = model.peft_config["default"].__dict__
    # del config_dict["origin_prompts"]
    # del config_dict["init_prompts"]
    # del config_dict["encoder_hidden_size"]

    # print(config_dict)

    # model.peft_config["default"] = PromptTuningConfig(**config_dict)

    soft_prompt_names = [
        "_".join(comb) for comb in combinations(sorted(data_args.dataset_names), 2)
    ] + data_args.dataset_names

    for o in peft_config.origin_prompts:
        soft_prompt_names.append(o)

        for n in soft_prompt_names:
            soft_prompt = torch.load(f"soft_prompts/{o}/{n}.bin")

            if isinstance(soft_prompt, dict):
                soft_prompt = soft_prompt["prompt_embeddings"]

            model.prompt_encoder.default.embedding.weight = torch.nn.Parameter(
                soft_prompt
            )

            model_name = generate_model_name(
                [model_args.model_name_or_path, "prompt tuning", o, n]
            )

            model.push_to_hub(model_name)

        soft_prompt_names.remove(o)
