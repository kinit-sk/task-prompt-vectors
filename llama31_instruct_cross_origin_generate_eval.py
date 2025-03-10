import argparse
import os
import torch
import numpy as np

from args import DataTrainingArguments, ArgumentParser
from arithmetics import PromptArithmeticsConfig
from tasks import AutoTask

from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import get_peft_model
from trl import SFTConfig, ModelConfig

from tqdm import tqdm
from utils import get_task_prompt_vectors_from_prompts

import pandas as pd

import evaluate

import re

# origin_prompts = [
#     "origin_0_meta-llama-3.1-8b-instruct",
#     "origin_1_meta-llama-3.1-8b-instruct",
#     "origin_2_meta-llama-3.1-8b-instruct",
# ]
origin_prompts = [
    "origin_0_deepseek-llm-7b-chat",
    "origin_1_deepseek-llm-7b-chat",
    "origin_2_deepseek-llm-7b-chat",
]
# dataset_names = ["qnli_text_instruct", "sst2_text_instruct", "trec_coarse_text_instruct"]
# dataset_names = ["math_instruct"]
dataset_names = ["squad_v2_instruct"]


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


def escape_inner_single_quotes(s):

    pattern = r"\[(.*?)\]"
    matches = re.findall(pattern, s)

    if len(matches) == 0:
        return "{'prediction_text': [], 'answer_start': []}"

    answers = re.findall(pattern, s)[0].split(",")

    escaped_answers = []

    # print(answers)
    for answer in answers:
        escaped_answers.append('"' + answer[1:-1].replace('"', "\\'") + '"')

    if len(matches) == 1:
        return (
            "{'prediction_text': ["
            + ", ".join(escaped_answers)
            + "], 'answer_start': []}"
        )

    return (
        "{'prediction_text': ["
        + ", ".join(escaped_answers)
        + "], 'answer_start': ["
        + re.findall(pattern, s)[1]
        + "]}"
    )


def tryeval(string, default):
    try:
        return eval(string)
    except Exception:
        print("eval exception:", string)
        return eval(default)


def predict_generative(test_dataset, model, tokenizer):

    y_pred = []
    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=False,
        top_p=None,
        temperature=None,
        use_cache=False,
        device="cuda",
    )

    for x_test in tqdm(test_dataset["text"]):

        # print(x_test)

        result = pipe(x_test)
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

        y_pred.append(answer)
        # print("result:", result)
        # print("answer:", answer)

    # print(len(y_pred))
    return y_pred


def evaluate_generative(y_pred, y_true, prefix="eval", squadv2=False, ids=None):
    rouge_metric = evaluate.load("rouge")
    bleu_metric = evaluate.load("bleu")

    scores_to_return = {}

    # print("y_pred:", y_pred)
    # print("y_true:", y_true)

    rouge_score = rouge_metric.compute(predictions=y_pred, references=y_true)
    bleu_score = bleu_metric.compute(predictions=y_pred, references=y_true)

    if squadv2:
        squad_v2_metric = evaluate.load("squad_v2")

        y_pred_processed = list(
            map(
                lambda x: tryeval(
                    escape_inner_single_quotes(x.replace("text", "prediction_text")),
                    "{'prediction_text': [], 'answer_start': []}",
                ),
                y_pred,
            )
        )
        y_true_proccesed = list(map(lambda x: {"answers": eval(x)}, y_true))
        for i, _id in enumerate(ids):
            y_pred_processed[i]["id"] = _id
            y_pred_processed[i]["no_answer_probability"] = 0.0
            if y_pred_processed[i]["prediction_text"]:
                y_pred_processed[i]["prediction_text"] = y_pred_processed[i][
                    "prediction_text"
                ][0]
            else:
                y_pred_processed[i]["prediction_text"] = ""
            del y_pred_processed[i]["answer_start"]

            y_true_proccesed[i]["id"] = _id

        # print("y_pred_processed:", y_pred_processed)
        # print("y_true_proccesed:", y_true_proccesed)
        squad_v2_score = squad_v2_metric.compute(
            predictions=y_pred_processed, references=y_true_proccesed
        )
        scores_to_return.update(
            {
                f"{prefix}/squad_exatct_match": squad_v2_score["exact"],
                f"{prefix}/squad_f1": squad_v2_score["f1"],
            }
        )

    scores_to_return.update(
        {
            f"{prefix}/bleu": bleu_score["bleu"],
            f"{prefix}/rougeL": rouge_score["rougeL"],
        }
    )

    return scores_to_return


task_prompt_vectors = get_task_prompt_vectors_from_prompts(
    origin_prompts=origin_prompts, dataset_names=dataset_names
)

timestamp = datetime.now().strftime("%m%d%Y%H%M%S")

argparse_parser = argparse.ArgumentParser(
    prog="Run prompt tuning",
    description="Run prompt tuning to train soft-prompts.",
)

argparse_parser.add_argument("filename", help="Filename of a config to run.")
argparse_parser.add_argument("--parse_data", help="Parse data into table.")
args = argparse_parser.parse_args()

if args.parse_data:
    print("Parsing data")

    data_dict = pd.read_csv(args.parse_data, index_col=0).to_dict()

    print(data_dict)

    results = {}
    pt_results = {}

    for dataset_name in data_dict["prompt_tuning"]:
        acc, f1, bleu, rougeL = [], [], [], []

        pt_acc, pt_f1 = [], []
        for origin in eval(data_dict["prompt_tuning"][dataset_name]):
            o1, o2 = "_".join(origin.split("_")[:2]), "_".join(origin.split("_")[3:5])

            if o1 != o2:
                print(
                    dataset_name,
                    o1,
                    o2,
                    eval(data_dict["prompt_tuning"][dataset_name])[origin],
                )

                if "squad" in dataset_name:
                    acc.append(
                        eval(data_dict["prompt_tuning"][dataset_name])[origin][
                            "eval/squad_exatct_match"
                        ]
                    )
                    f1.append(
                        eval(data_dict["prompt_tuning"][dataset_name])[origin]["eval/squad_f1"]
                    )

                bleu.append(
                    eval(data_dict["prompt_tuning"][dataset_name])[origin]["eval/bleu"]
                )

                rougeL.append(
                    eval(data_dict["prompt_tuning"][dataset_name])[origin][
                        "eval/rougeL"
                    ]
                )
            else:
                pt_acc.append(
                    eval(data_dict["prompt_tuning"][dataset_name])[origin][
                        "eval/squad_exatct_match"
                    ]
                )
                pt_f1.append(
                    eval(data_dict["prompt_tuning"][dataset_name])[origin][
                    "eval/squad_f1"
                    ]
                )

        print(acc, f1)
        results[dataset_name] = {
            # "accuracy": {
            #     "mean": np.round(np.array(acc).mean(), 1),
            #     "std": np.round(np.array(acc).std(), 1),
            # },
            # "f1": {
            #     "mean": np.round(np.array(f1).mean(), 1),
            #     "std": np.round(np.array(f1).std(), 1),
            # },
            "bleu": {
                "mean": np.round(np.array(bleu).mean() * 100, 1),
                "std": np.round(np.array(bleu).std() * 100, 1),
            },
            "rougeL": {
                "mean": np.round(np.array(rougeL).mean() * 100, 1),
                "std": np.round(np.array(rougeL).std() * 100, 1),
            },
        }
        # pt_results[dataset_name] = {
        #     "accuracy": {
        #         "mean": np.round(np.array(pt_acc).mean() * 100, 1),
        #         "std": np.round(np.array(pt_acc).std() * 100, 1),
        #     },
        #     "f1": {
        #         "mean": np.round(np.array(pt_f1).mean() * 100, 1),
        #         "std": np.round(np.array(pt_f1).std() * 100, 1),
        #     },
        # }

    pd.DataFrame.from_dict(results).to_csv("cross_origin_results.csv")
    print(results)
    # print(pd.DataFrame.from_dict(pt_results).to_csv("prompt_tuning_results.csv"))
    exit()


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

for dataset_name in dataset_names:
    print(f"task: {dataset_name}")

    test_dataset = AutoTask.get(dataset_name).get(
        split="test",
        task_type=peft_config.task_type,
        add_prefix=False,
        n_obs=data_args.max_test_samples,
        split_validation_test=data_args.split_validation_test,
    )

    chat_test_dataset = test_dataset.map(apply_test_template)

    if "deepseek-llm" in model_args.model_name_or_path.lower():
        chat_test_dataset = chat_test_dataset.map(
            replace_map, fn_kwargs={"str1": "label:", "str2": ""}
        )

    # print("Eval zero-shot performance")
    # test_results = evaluate_generative(
    #     predict_generative(
    #         chat_test_dataset,
    #         model.base_model,
    #         tokenizer,
    #     ),
    #     test_dataset["target"],
    #     squadv2="squad" in dataset_name,
    #     ids=test_dataset["id"] if "squad" in dataset_name else None,
    # )
    # print(test_results)

    # full_test_results["zero_shot"][dataset_name] = test_results
    full_test_results["prompt_tuning"][dataset_name] = {}

print("Eval prompt tuning performance")
for o1 in task_prompt_vectors:
    origin_weights = torch.load(f"soft_prompts/{o1}/{o1}.bin")["prompt_embeddings"].to(
        training_args.device
    )

    for o2 in task_prompt_vectors:
        if o1 == o2:
            continue

        training_args.run_name = f"addition_{timestamp}_{o1}_{o2}"
        print(f"origin: {o1} task prompt vectors: {o2}")

        for tp in task_prompt_vectors[o2]:
            for dataset_name in tp.tasks:
                print(f"task: {dataset_name}")

                test_dataset = AutoTask.get(dataset_name).get(
                    split="test",
                    task_type=peft_config.task_type,
                    add_prefix=False,
                    n_obs=data_args.max_test_samples,
                    split_validation_test=data_args.split_validation_test,
                )

                chat_test_dataset = test_dataset.map(apply_test_template)

                model.prompt_encoder.default.embedding.weight = tp.apply(
                    origin_weights=origin_weights
                )
                print(
                    "current PT weights:", model.prompt_encoder.default.embedding.weight
                )

                test_results = evaluate_generative(
                    predict_generative(
                        chat_test_dataset,
                        model,
                        tokenizer,
                    ),
                    test_dataset["target"],
                    squadv2="squad" in dataset_name,
                    ids=test_dataset["id"] if "squad" in dataset_name else None,
                )

                print(test_results)

                full_test_results["prompt_tuning"][dataset_name][
                    f"{o1}_{o2}_{dataset_name}"
                ] = test_results


df = pd.DataFrame.from_dict(full_test_results)

df.to_csv(f"{timestamp}_llama31_test_results.csv")
