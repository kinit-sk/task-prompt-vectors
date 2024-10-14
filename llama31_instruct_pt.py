import argparse
import os
import torch
import wandb
import numpy as np

from args import DataTrainingArguments, ArgumentParser
from arithmetics import PromptArithmeticsConfig
from tasks import AutoTask

from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
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

    print(y_pred)
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

os.environ["WANDB_PROJECT"] = "arithmetics"

output_dir = training_args.output_dir

for origin_prompt in peft_config.origin_prompts:
    training_args.origin_prompt_name = origin_prompt

    for dataset_name in data_args.dataset_names:

        training_args.output_dir = f"{output_dir}_{timestamp}_{'_'.join(data_args.dataset_names)}_{origin_prompt}"
        training_args.run_name = f"prompt_tuning_{timestamp}_{'_'.join(data_args.dataset_names)}_{origin_prompt}"

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

        chat_train_dataset = train_dataset.map(apply_template)
        chat_valid_dataset = valid_dataset.map(apply_template)
        chat_test_dataset = test_dataset.map(apply_test_template)

        if args.print_data:
            print("Train data")
            print(chat_train_dataset["text"][0])

            print("Valid data")
            print(chat_valid_dataset["text"][0])

            print("Test data")
            print(chat_test_dataset["text"][0])

            exit(0)

        # pre_train_results = evaluate(
        #     predict(
        #         chat_test_dataset,
        #         model,
        #         tokenizer,
        #         AutoTask.get(dataset_name).labels_list,
        #     ),
        #     test_dataset["target"],
        #     {label: id_ for id_, label in AutoTask.get(dataset_name).id2label.items()},
        #     prefix="test",
        # )

        # print(pre_train_results)

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=chat_train_dataset,
            eval_dataset=chat_valid_dataset,
            tokenizer=tokenizer,
            packing=False,
        )

        trainer.train()

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

        if isinstance(dataset_name, list):
            save_name = f"./saves/prompt_tuning_{timestamp}_{'_'.join(dataset_name)}_{origin_prompt}_best"
        else:
            save_name = (
                f"./saves/prompt_tuning_{timestamp}_{dataset_name}_{origin_prompt}_best"
            )

        for metric in test_results:
            wandb.define_metric(metric, step_metric="step")

        print(test_results)
        wandb.run.log(test_results)

        model.save_pretrained(save_name)

        if wandb.run is not None:
            artifact = wandb.Artifact(name=training_args.run_name, type="weights")
            artifact.add_dir(local_path=save_name)
            wandb.run.log_artifact(artifact)
            wandb.log(data={})

            wandb.finish()
