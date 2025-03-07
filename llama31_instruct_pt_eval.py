import argparse
import os
import torch
import numpy as np

from args import DataTrainingArguments, ArgumentParser
from arithmetics import PromptArithmeticsConfig
from tasks import AutoTask

from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, EvalPrediction
from peft import get_peft_model
from trl import SFTConfig, ModelConfig

from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

from metrics.utils import binary_reverse

from utils import get_task_prompt_from_safetensor

import pandas as pd

# prompts_to_load = {
#     "qnli_text_instruct": [
#         "prompt_tuning_09262024190021_qnli_text_instruct_origin_0_meta-llama-3.1-8b-instruct_best",
#         "prompt_tuning_09282024205601_qnli_text_instruct_origin_1_meta-llama-3.1-8b-instruct_best",
#     ],
#     "sst2_text_instruct": [
#         "prompt_tuning_09282024094012_sst2_text_instruct_origin_0_meta-llama-3.1-8b-instruct_best",
#         "prompt_tuning_09282024094012_sst2_text_instruct_origin_1_meta-llama-3.1-8b-instruct_best",
#         "prompt_tuning_09282024094012_sst2_text_instruct_origin_2_meta-llama-3.1-8b-instruct_best",
#     ],
#     "trec_coarse_text_instruct": [
#         "prompt_tuning_09282024094154_trec_coarse_text_instruct_origin_0_meta-llama-3.1-8b-instruct_best",
#         "prompt_tuning_09282024094154_trec_coarse_text_instruct_origin_1_meta-llama-3.1-8b-instruct_best",
#         "prompt_tuning_09282024094154_trec_coarse_text_instruct_origin_2_meta-llama-3.1-8b-instruct_best",
#     ],
# }

# prompts_to_load = {
#     "qqp_text_instruct": [
#         "prompt_tuning_02182025154215_qqp_text_instruct_origin_0_meta-llama-3.1-8b-instruct_best",
#         "prompt_tuning_02182025195555_qqp_text_instruct_origin_1_meta-llama-3.1-8b-instruct_best",
#         "prompt_tuning_02182025195329_qqp_text_instruct_origin_2_meta-llama-3.1-8b-instruct_best",
#     ],
#     "stsb_text_instruct": [
#         "prompt_tuning_02202025132651_stsb_text_instruct_origin_0_meta-llama-3.1-8b-instruct_best",
#         "prompt_tuning_02202025132651_stsb_text_instruct_origin_1_meta-llama-3.1-8b-instruct_best",
#         "prompt_tuning_02202025132651_stsb_text_instruct_origin_2_meta-llama-3.1-8b-instruct_best",
#     ]

# }

prompts_to_load = {
    "stsb_text_instruct": [
        "prompt_tuning_03042025154255_stsb_text_instruct_origin_0_deepseek-llm-7b-chat_best",
        "prompt_tuning_03042025154255_stsb_text_instruct_origin_1_deepseek-llm-7b-chat_best",
        "prompt_tuning_03042025154255_stsb_text_instruct_origin_2_deepseek-llm-7b-chat_best",
    ]
}

# prompts_to_load = {
#     "rte_text_instruct": [
#         "prompt_tuning_02272025162514_rte_text_instruct_origin_1_deepseek-r1-distill-llama-8b_best",
#         "prompt_tuning_02272025162514_rte_text_instruct_origin_2_deepseek-r1-distill-llama-8b_best",
#     ],
#     "stsb_text_instruct" : [
#         "prompt_tuning_02272025162407_stsb_text_instruct_origin_0_deepseek-r1-distill-llama-8b_best",
#         "prompt_tuning_02272025162407_stsb_text_instruct_origin_1_deepseek-r1-distill-llama-8b_best",
#         "prompt_tuning_02272025162407_stsb_text_instruct_origin_2_deepseek-r1-distill-llama-8b_best",
#     ],
#     "mrpc_text_instruct" : [
#         "prompt_tuning_02272025163540_mrpc_text_instruct_origin_0_deepseek-r1-distill-llama-8b_best",
#         "prompt_tuning_02272025163540_mrpc_text_instruct_origin_1_deepseek-r1-distill-llama-8b_best",
#         "prompt_tuning_02272025163540_mrpc_text_instruct_origin_2_deepseek-r1-distill-llama-8b_best",
#     ],
#     "math_text_instruct": [
#         "prompt_tuning_02272025164541_math_instruct_origin_0_deepseek-r1-distill-llama-8b_best",
#         "prompt_tuning_02272025164541_math_instruct_origin_1_deepseek-r1-distill-llama-8b_best",
#         "prompt_tuning_02272025164541_math_instruct_origin_2_deepseek-r1-distill-llama-8b_best",
#     ],
# }

# prompts_to_load = {
#     "rte_text_instruct": [
#         "prompt_tuning_02182025153915_rte_text_instruct_origin_0_meta-llama-3.1-8b-instruct_best",
#         "prompt_tuning_02182025195335_rte_text_instruct_origin_1_meta-llama-3.1-8b-instruct_best",
#         "prompt_tuning_02182025195335_rte_text_instruct_origin_2_meta-llama-3.1-8b-instruct_best",
#     ],
#     "mrpc_text_instruct": [
#         "prompt_tuning_02182025154213_mrpc_text_instruct_origin_0_meta-llama-3.1-8b-instruct_best",
#         "prompt_tuning_02182025195351_mrpc_text_instruct_origin_1_meta-llama-3.1-8b-instruct_best",
#         "prompt_tuning_02182025195351_mrpc_text_instruct_origin_2_meta-llama-3.1-8b-instruct_best",
#     ],
#     "cola_text_instruct": [
#         "prompt_tuning_02182025154220_cola_text_instruct_origin_0_meta-llama-3.1-8b-instruct_best",
#         "prompt_tuning_02182025195340_cola_text_instruct_origin_1_meta-llama-3.1-8b-instruct_best",
#         "prompt_tuning_02182025195340_cola_text_instruct_origin_2_meta-llama-3.1-8b-instruct_best",
#     ],
# }


def replace_map(examples, str1, str2):
    # print(examples["text"].replace(str1, str2))
    return {"text": examples["text"].replace(str1, str2)}


def apply_test_template(examples):
    return {
        "text": tokenizer.apply_chat_template(
            [examples], tokenize=False, add_generation_prompt=True
        )
    }


def apply_template(examples):
    return {
        "text": tokenizer.apply_chat_template(
            [examples, {"role": "assistant", "content": examples["target"]}],
            tokenize=False,
            add_generation_prompt=False,
        )
    }


def predict(test_dataset, model, tokenizer, labels_list):
    y_pred = []
    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=16,
        do_sample=False,
        top_p=None,
        temperature=None,
        use_cache=False,
        device="cuda",
    )

    for x_test in tqdm(test_dataset["text"]):

        # print(x_test)
        result = pipe(x_test)
        # print(result)

        # print(model.config.name_or_path)

        if "deepseek-llm" in model.config.name_or_path.lower():
            answer = result[0]["generated_text"].split("Assistant:")[-1].strip()
        else:
            answer = (
                result[0]["generated_text"]
                .split("label:<|eot_id|><|start_header_id|>assistant<|end_header_id|>")[
                    -1
                ]
                .strip()
            )

        for label in labels_list:
            if label.lower() == answer.lower():
                y_pred.append(label)
                break
        else:
            y_pred.append("none")
            print(result)

        # print(answer)

    return y_pred


def evaluate(y_pred, y_true, compute_metrics, prefix="eval"):
    metrics = compute_metrics(EvalPrediction(y_pred, y_true))

    return {f"{prefix}/{k}": v for k, v in metrics.items()}


timestamp = datetime.now().strftime("%m%d%Y%H%M%S")

argparse_parser = argparse.ArgumentParser(
    prog="Run prompt tuning",
    description="Run prompt tuning to train soft-prompts.",
)

argparse_parser.add_argument("filename", help="Filename of a config to run.")
argparse_parser.add_argument(
    "--print_data", action="store_true", help="Print parsed data and exit."
)
args = argparse_parser.parse_args()

parser = ArgumentParser(
    (SFTConfig, ModelConfig, DataTrainingArguments, PromptArithmeticsConfig)
)

training_args, model_args, data_args, peft_config = parser.parse_toml_file(
    args.filename
)

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

full_test_results = {"zero_shot": {}, "prompt_tuning": {}}

for dataset_name in prompts_to_load:
    print(f"task: {dataset_name}")

    # print("Eval zero-shot performance")
    test_dataset = AutoTask.get(dataset_name).get(
        split="test",
        task_type=peft_config.task_type,
        add_prefix=False,
        n_obs=data_args.max_test_samples,
        split_validation_test=data_args.split_validation_test,
    )

    compute_metrics = AutoTask.get(dataset_name).get_compute_metrics(
        tokenizer, postprocess=False
    )

    chat_test_dataset = test_dataset.map(apply_test_template)

    if "deepseek-llm" in model_args.model_name_or_path.lower():
        chat_test_dataset = chat_test_dataset.map(
            replace_map, fn_kwargs={"str1": "label:", "str2": ""}
        )

    # test_results = evaluate(
    #     predict(
    #         chat_test_dataset,
    #         model.base_model,
    #         tokenizer,
    #         AutoTask.get(dataset_name).labels_list,
    #     ),
    #     test_dataset["target"],
    #     compute_metrics,
    #     prefix="test",
    # )

    # print(test_results)

    # full_test_results["zero_shot"][dataset_name] = test_results
    full_test_results["prompt_tuning"][dataset_name] = {}

    print("Eval prompt tuning performance")
    for promt_to_load in prompts_to_load[dataset_name]:
        model.prompt_encoder.default.embedding.weight = get_task_prompt_from_safetensor(
            f"saves/{promt_to_load}"
        )

        print("current PT weights:", model.prompt_encoder.default.embedding.weight)

        test_results = evaluate(
            predict(
                chat_test_dataset,
                model,
                tokenizer,
                AutoTask.get(dataset_name).labels_list,
            ),
            test_dataset["target"],
            compute_metrics,
            prefix="test",
        )

        print(test_results)

        full_test_results["prompt_tuning"][dataset_name][promt_to_load] = test_results


df = pd.DataFrame.from_dict(full_test_results)

df.to_csv(f"{timestamp}_llama31_test_results.csv")
