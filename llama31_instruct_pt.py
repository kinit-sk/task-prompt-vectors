import argparse
import os
import torch
import wandb
import numpy as np

from args import DataTrainingArguments, ArgumentParser
from arithmetics import PromptArithmeticsConfig
from tasks import AutoTask

from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, EvalPrediction
from peft import get_peft_model
from trl import SFTTrainer, SFTConfig, ModelConfig

from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

from metrics.utils import binary_reverse


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

def replace_map(examples, str1, str2):
    # print(examples["text"].replace(str1, str2))
    return {"text": examples["text"].replace(str1, str2)}


def predict(test_dataset, model, tokenizer, labels_list):
    y_pred = []
    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=2048,
        do_sample=False,
        top_p=None,
        temperature=None,
        use_cache=False,
        device="cuda",
    )

    for x_test in tqdm(test_dataset["text"]):

        result = pipe(x_test)
        print(result)
        # print(x_test)
        # print(model.config.name_or_path)
        
        if "deepseek" in model.config.name_or_path:
            answer = (
                result[0]["generated_text"]
                .split("</think>")[-1].strip()
            )
        else:
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
            # print(x_test)
         
        # print(answer)

    return y_pred


def predict_generative(test_dataset, model, tokenizer):
    y_pred = []
    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        do_sample=False,
        top_p=None,
        temperature=None,
        device="cuda",
    )

    for x_test in tqdm(test_dataset["text"]):

        print(x_test)

        result = pipe(x_test)
        answer = (
            result[0]["generated_text"]
            .split("<|eot_id|><|start_header_id|>assistant<|end_header_id|>")[-1]
            .strip()
        )

        if "answer:" in answer:
            answer = answer.split("answer:")[-1]
        else:
            answer = None

        y_pred.append(answer)
        print("result:", result)
        print("answer:", answer)

    return y_pred


def evaluate(y_pred, y_true, compute_metrics, prefix="eval"):
    metrics = compute_metrics(EvalPrediction(y_pred, y_true))
    
    return {f"{prefix}/{k}": v for k,v in metrics.items()}


def evaluate_generative(y_pred, y_true, prefix="eval"):

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    accuracy = np.size(y_pred == y_true) / y_true.size

    return {f"{prefix}/accuracy": accuracy}


timestamp = datetime.now().strftime("%m%d%Y%H%M%S")

argparse_parser = argparse.ArgumentParser(
    prog="Run prompt tuning",
    description="Run prompt tuning to train soft-prompts.",
)

argparse_parser.add_argument("filename", help="Filename of a config to run.")
argparse_parser.add_argument(
    "--print_data", action="store_true", help="Print parsed data and exit."
)
argparse_parser.add_argument(
    "--pretrain_eval", action="store_true", help="Do pre-training evaluation."
)
args = argparse_parser.parse_args()

parser = ArgumentParser(
    (SFTConfig, ModelConfig, DataTrainingArguments, PromptArithmeticsConfig)
)

training_args, model_args, data_args, peft_config = parser.parse_toml_file(
    args.filename
)

training_args.packing = False

os.environ["WANDB_PROJECT"] = "arithmetics"

output_dir = training_args.output_dir

for origin_prompt in peft_config.origin_prompts:
    training_args.origin_prompt_name = origin_prompt

    for dataset_name in data_args.dataset_names:

        training_args.output_dir = f"prompt_tuning_{timestamp}_{'_'.join(data_args.dataset_names)}_origin_{origin_prompt.split("_")[1]}_{model_args.model_name_or_path.split("/")[-1].lower()}"
        training_args.run_name = f"prompt_tuning_{timestamp}_{'_'.join(data_args.dataset_names)}_origin_{origin_prompt.split("_")[1]}_{model_args.model_name_or_path.split("/")[-1].lower()}"

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

        model.prompt_encoder.default.embedding.weight = torch.nn.Parameter(
            torch.load(f"saves/{origin_prompt}/{origin_prompt}.bin")[
                "prompt_embeddings"
            ].to("cuda")
        )

        print("current PT weights:", model.prompt_encoder.default.embedding.weight)

        model.print_trainable_parameters()

        print(f"task: {dataset_name}")

        train_dataset = AutoTask.get(dataset_name).get(
            split="train",
            task_type=peft_config.task_type,
            add_prefix=False,
            n_obs=data_args.max_train_samples,
            split_validation_test=data_args.split_validation_test,
        )
        valid_dataset = AutoTask.get(dataset_name).get(
            split="validation",
            task_type=peft_config.task_type,
            add_prefix=False,
            n_obs=data_args.max_valid_samples,
            split_validation_test=data_args.split_validation_test,
        )
        test_dataset = AutoTask.get(dataset_name).get(
            split="test",
            task_type=peft_config.task_type,
            add_prefix=False,
            n_obs=data_args.max_test_samples,
            split_validation_test=data_args.split_validation_test,
        )

        compute_metrics = AutoTask.get(dataset_name).get_compute_metrics(tokenizer, postprocess=False)
        chat_train_dataset = train_dataset.map(apply_template)
        chat_valid_dataset = valid_dataset.map(apply_template)
        chat_test_dataset = test_dataset.map(apply_test_template)

        if "deepseek" in model_args.model_name_or_path:
            chat_train_dataset = chat_train_dataset.map(replace_map, fn_kwargs={"str1": "<｜Assistant｜>", "str2": "<｜Assistant｜><think></think>"})
            chat_valid_dataset = chat_valid_dataset.map(replace_map, fn_kwargs={"str1": "<｜Assistant｜>", "str2": "<｜Assistant｜><think></think>"})

        if args.print_data:
            print("Train data")
            print(chat_train_dataset["text"][0])

            print("Valid data")
            print(chat_valid_dataset["text"][0])

            print("Test data")
            print(chat_test_dataset["text"][0])

            exit(0)

        if args.pretrain_eval:
            if "math" in dataset_name:
                l4_test_dataset = AutoTask.get("math_l4_instruct_eval_aimo").get(
                    split="test",
                    task_type=peft_config.task_type,
                    add_prefix=False,
                    n_obs=data_args.max_test_samples,
                    split_validation_test=data_args.split_validation_test,
                )

                l5_test_dataset = AutoTask.get("math_l5_instruct_eval_aimo").get(
                    split="test",
                    task_type=peft_config.task_type,
                    add_prefix=False,
                    n_obs=data_args.max_test_samples,
                    split_validation_test=data_args.split_validation_test,
                )

                chat_l4_test_dataset = l4_test_dataset.map(apply_test_template)
                chat_l5_test_dataset = l5_test_dataset.map(apply_test_template)

                pre_train_l4_results = evaluate_generative(
                    predict_generative(
                        chat_l4_test_dataset,
                        model.base_model,
                        tokenizer,
                    ),
                    test_dataset["target"],
                    prefix="test_l4",
                )

                pre_train_l5_results = evaluate_generative(
                    predict_generative(
                        chat_l5_test_dataset,
                        model.base_model,
                        tokenizer,
                    ),
                    test_dataset["target"],
                    prefix="test_l5",
                )
            else:
                pre_train_results = evaluate(
                    predict(
                        chat_test_dataset,
                        model.base_model,
                        tokenizer,
                        AutoTask.get(dataset_name).labels_list,
                    ),
                    test_dataset["target"],
                    compute_metrics,
                    prefix="test",
                )

            print(pre_train_results)
            exit()

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=chat_train_dataset,
            eval_dataset=chat_valid_dataset,
            tokenizer=tokenizer,
            # packing=False,
        )

        trainer.train()

        if (
            "math" in dataset_name
            or "squad" in dataset_name
            or "hotpot" in dataset_name
            or "stsb" in dataset_name
        ):
            pass
        else:
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

            for metric in test_results:
                wandb.define_metric(metric, step_metric="step")

            print(test_results)
            wandb.run.log(test_results)

        if isinstance(dataset_name, list):
            save_name = f"./saves/prompt_tuning_{timestamp}_{'_'.join(dataset_name)}_origin_{origin_prompt.split("_")[1]}_{model_args.model_name_or_path.split("/")[-1].lower()}_best"
        else:
            save_name = (
                f"./saves/prompt_tuning_{timestamp}_{dataset_name}_origin_{origin_prompt.split("_")[1]}_{model_args.model_name_or_path.split("/")[-1].lower()}_best"
            )

        model.save_pretrained(save_name)

        if wandb.run is not None:
            artifact = wandb.Artifact(name=training_args.run_name, type="weights")
            artifact.add_dir(local_path=save_name)
            wandb.run.log_artifact(artifact)
            wandb.log(data={})

            wandb.finish()
