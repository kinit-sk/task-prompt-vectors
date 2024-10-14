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
from sklearn.metrics import accuracy_score, f1_score

from metrics.utils import binary_reverse

from utils import get_task_prompt_vectors_from_prompts

import pandas as pd


origin_prompts = ["origin_0_meta-llama-3.1-8b-instruct", "origin_1_meta-llama-3.1-8b-instruct", "origin_2_meta-llama-3.1-8b-instruct"]
# dataset_names = ["qnli_text_instruct", "sst2_text_instruct", "trec_coarse_text_instruct"]
dataset_names = ["mnli_text_instruct"]


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

        result = pipe(x_test)
        answer = (
            result[0]["generated_text"]
            .split("label:<|eot_id|><|start_header_id|>assistant<|end_header_id|>")[-1]
            .strip()
        )

        for label in labels_list:
            if label.lower() == answer.lower():
                y_pred.append(label)
                break
        else:
            y_pred.append("none")
            # print(answer)

    return y_pred


def evaluate(y_pred, y_true, mapping, prefix="eval"):
    def map_func(x):
        return mapping.get(x, -1)

    # print(y_pred)
    y_pred_mapped = np.vectorize(map_func)(y_pred)
    y_true_mapped = np.vectorize(map_func)(y_true)

    unique_labels = list(set(y_true_mapped))

    accuracy = accuracy_score(y_pred=y_pred_mapped, y_true=y_true_mapped)

    if len(unique_labels) > 2:
        f1 = f1_score(
            y_pred=y_pred_mapped,
            y_true=y_true_mapped,
            labels=unique_labels,
            average="macro",
        )
    else:
        invalid_idx_mask = y_pred_mapped == -1
        y_pred_mapped[invalid_idx_mask] = binary_reverse(
            y_true_mapped[invalid_idx_mask], unique_labels
        )

        f1 = f1_score(
            y_pred=y_pred_mapped,
            y_true=y_true_mapped,
            labels=unique_labels,
            pos_label=unique_labels[1],
        )

    return {f"{prefix}/accuracy": accuracy, f"{prefix}/f1": f1}


task_prompt_vectors = get_task_prompt_vectors_from_prompts(origin_prompts=origin_prompts, dataset_names=dataset_names)

timestamp = datetime.now().strftime("%m%d%Y%H%M%S")

argparse_parser = argparse.ArgumentParser(
    prog="Run prompt tuning",
    description="Run prompt tuning to train soft-prompts.",
)

argparse_parser.add_argument("filename", help="Filename of a config to run.")
argparse_parser.add_argument(
    "--parse_data", help="Parse data into table."
)
args = argparse_parser.parse_args()

if args.parse_data:
    print("Parsing data")

    data_dict = pd.read_csv(args.parse_data, index_col=0).to_dict()

    results = {}
    pt_results = {}

    for dataset_name in data_dict["prompt_tuning"]:
        acc, f1 = [], []

        pt_acc, pt_f1 = [], []
        for origin in eval(data_dict["prompt_tuning"][dataset_name]):
            o1, o2 = "_".join(origin.split("_")[:2]), "_".join(origin.split("_")[3:5])

            if o1 != o2:
                print(dataset_name, o1, o2, eval(data_dict["prompt_tuning"][dataset_name])[origin])

                acc.append(eval(data_dict["prompt_tuning"][dataset_name])[origin]["test/accuracy"])
                f1.append(eval(data_dict["prompt_tuning"][dataset_name])[origin]["test/f1"])
            else:
                pt_acc.append(eval(data_dict["prompt_tuning"][dataset_name])[origin]["test/accuracy"])
                pt_f1.append(eval(data_dict["prompt_tuning"][dataset_name])[origin]["test/f1"])

        # print(acc, f1)
        results[dataset_name] = {"accuracy": {"mean": np.round(np.array(acc).mean()*100, 1), "std": np.round(np.array(acc).std()*100, 1)}, "f1": {"mean": np.round(np.array(f1).mean()*100, 1), "std": np.round(np.array(f1).std()*100, 1)}}
        pt_results[dataset_name] = {"accuracy": {"mean": np.round(np.array(pt_acc).mean()*100, 1), "std": np.round(np.array(pt_acc).std()*100, 1)}, "f1": {"mean": np.round(np.array(pt_f1).mean()*100, 1), "std": np.round(np.array(pt_f1).std()*100, 1)}}

    print(pd.DataFrame.from_dict(results).to_csv("cross_origin_results.csv"))
    print(pd.DataFrame.from_dict(pt_results).to_csv("prompt_tuning_results.csv"))
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

    print("Eval zero-shot performance")
    test_results = evaluate(
        predict(
            chat_test_dataset,
            model.base_model,
            tokenizer,
            AutoTask.get(dataset_name).labels_list,
        ),
        test_dataset["target"],
        {label: id_ for id_, label in AutoTask.get(dataset_name).id2label.items()},
        prefix="test",
    )

    print(test_results)

    full_test_results["zero_shot"][dataset_name] = test_results
    full_test_results["prompt_tuning"][dataset_name] = {}

print("Eval prompt tuning performance")
for o1 in task_prompt_vectors:
    origin_weights = torch.load(f"soft_prompts/{o1}/{o1}.bin")[
            "prompt_embeddings"
        ].to(training_args.device)

    for o2 in task_prompt_vectors:
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

                model.prompt_encoder.default.embedding.weight = tp.apply(origin_weights=origin_weights)
                print("current PT weights:", model.prompt_encoder.default.embedding.weight)     

                test_results = evaluate(
                    predict(
                        chat_test_dataset,
                        model,
                        tokenizer,
                        AutoTask.get(dataset_name).labels_list,
                    ),
                    test_dataset["target"],
                    {label: id_ for id_, label in AutoTask.get(dataset_name).id2label.items()},
                    prefix="test",
                )

                print(test_results)

                full_test_results["prompt_tuning"][dataset_name][f"{o1}_{o2}_{dataset_name}"] = test_results


df = pd.DataFrame.from_dict(full_test_results)

df.to_csv(f"{timestamp}_llama31_test_results.csv")