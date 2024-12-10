import argparse
import torch

from args import DataTrainingArguments, ArgumentParser
from arithmetics import PromptArithmeticsConfig
from tasks import AutoTask

from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import get_peft_model
from trl import SFTConfig, ModelConfig

from tqdm import tqdm

from utils import get_task_prompt_from_safetensor

import evaluate

import pandas as pd

from transformers.pipelines.pt_utils import KeyDataset

prompts_to_load = {
    "squad_v2_instruct": [
        "prompt_tuning_12042024144342_squad_v2_instruct_origin_0_meta-llama-3.1-8b-instruct_best",
        "prompt_tuning_12042024144534_squad_v2_instruct_origin_1_meta-llama-3.1-8b-instruct_best",
        "prompt_tuning_12042024144807_squad_v2_instruct_origin_2_meta-llama-3.1-8b-instruct_best",
    ],
    # "hotpot_qa_instruct": [
    #     "prompt_tuning_12042024144827_hotpot_qa_instruct_origin_0_meta-llama-3.1-8b-instruct_best",
    #     "prompt_tuning_12042024145638_hotpot_qa_instruct_origin_1_meta-llama-3.1-8b-instruct_best",
    #     "prompt_tuning_12042024150532_hotpot_qa_instruct_origin_2_meta-llama-3.1-8b-instruct_best",
    # ],
    # "math_instruct": [
    #     "prompt_tuning_12042024094358_math_instruct_origin_0_meta-llama-3.1-8b-instruct_best",
    #     "prompt_tuning_12042024094358_math_instruct_origin_1_meta-llama-3.1-8b-instruct_best",
    #     "prompt_tuning_12042024094358_math_instruct_origin_2_meta-llama-3.1-8b-instruct_best",
    # ],
}


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
        answer = (
            result[0]["generated_text"]
            .split("<|eot_id|><|start_header_id|>assistant<|end_header_id|>")[-1]
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

        y_pred_processed = list(map(lambda x: tryeval(x.replace("text", "prediction_text"), "{'prediction_text': [], 'answer_start': []}"), y_pred))
        y_true_proccesed = list(map(lambda x: {"answers": eval(x)}, y_true))
        for i, _id in enumerate(ids):
            y_pred_processed[i]["id"] = _id
            y_pred_processed[i]["no_answer_probability"] = 0.
            if y_pred_processed[i]["prediction_text"]:
                y_pred_processed[i]["prediction_text"] = y_pred_processed[i]["prediction_text"][0]
            else:
                y_pred_processed[i]["prediction_text"] = ""
            del y_pred_processed[i]["answer_start"]
            
            y_true_proccesed[i]["id"] = _id

        # print("y_pred_processed:", y_pred_processed)
        # print("y_true_proccesed:", y_true_proccesed)
        squad_v2_score = squad_v2_metric.compute(predictions=y_pred_processed, references=y_true_proccesed)
        scores_to_return.update({f"{prefix}/squad_exatct_match": squad_v2_score["exact"],f"{prefix}/squad_f1": squad_v2_score["f1"]})

    scores_to_return.update({f"{prefix}/bleu": bleu_score["bleu"], f"{prefix}/rougeL": rouge_score["rougeL"]})

    return scores_to_return


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

    print("Eval zero-shot performance")
    test_dataset = AutoTask.get(dataset_name).get(
        split="test",
        task_type=peft_config.task_type,
        add_prefix=False,
        n_obs=data_args.max_test_samples,
        split_validation_test=data_args.split_validation_test,
    )

    chat_test_dataset = test_dataset.map(apply_test_template)

    if args.print_data:
        print("Test data")
        # print(test_dataset[0])
        print(chat_test_dataset["text"][0], test_dataset["target"][0], test_dataset["id"][0])

        exit(0)

    test_results = evaluate_generative(
        predict_generative(
            chat_test_dataset,
            model.base_model,
            tokenizer,
        ),
        test_dataset["target"],
        squadv2="squad" in dataset_name,
        ids=test_dataset["id"] if "squad" in dataset_name else None,
    )

    print(test_results)

    full_test_results["zero_shot"][dataset_name] = test_results
    full_test_results["prompt_tuning"][dataset_name] = {}

    print("Eval prompt tuning performance")
    for promt_to_load in prompts_to_load[dataset_name]:
        model.prompt_encoder.default.embedding.weight = get_task_prompt_from_safetensor(
            f"saves/{promt_to_load}"
        )

        print("current PT weights:", model.prompt_encoder.default.embedding.weight)

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

        full_test_results["prompt_tuning"][dataset_name][promt_to_load] = test_results


df = pd.DataFrame.from_dict(full_test_results)

df.to_csv(f"{timestamp}_llama31_test_results.csv")
