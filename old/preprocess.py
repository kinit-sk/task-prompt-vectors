from datasets import load_dataset

def get_mrpc(tokenizer, source_len, target_len):
    dataset = load_dataset("glue", "mrpc")

    def tokenizer_function(examples):
        inputs = examples["inputs"]
        targets = examples["targets"]

        model_inputs = tokenizer(inputs, max_length=source_len, padding="max_length", truncation=True, return_tensors="pt")
        labels = tokenizer(targets, max_length=target_len, padding="max_length", truncation=True, return_tensors="pt")


        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else 0) for l in label]
            for label in labels["input_ids"]
        ]
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs


    def preprocess_function(example):
        id2label = {0: "0", 1: "1"}

        inputs = " ".join([
            "sentence1:",
            example["sentence1"],
            "sentence2:",
            example["sentence2"],
        ])

        targets = id2label[example["label"]]


        return {"inputs": inputs, "targets": targets}



    processed_datasets = dataset.map(
        preprocess_function,
        remove_columns=dataset["train"].column_names,
        load_from_cache_file=False,
        desc="Running preprocessor on dataset",
    ).map(tokenizer_function,
        batched=True,
        load_from_cache_file=False,
        remove_columns=["inputs", "targets"],
        desc="Running tokenizer on dataset",
    )


    train_dataset = processed_datasets["train"].shuffle()
    eval_dataset = processed_datasets["validation"]
    test_dataset = processed_datasets["test"]

    return train_dataset, eval_dataset, test_dataset


def get_rte(tokenizer, source_len, target_len):
    dataset = load_dataset("glue", "rte")
    del dataset["test"]

    def tokenizer_function(examples):
        inputs = examples["inputs"]
        targets = examples["targets"]

        model_inputs = tokenizer(inputs, max_length=source_len, padding="max_length", truncation=True, return_tensors="pt")
        labels = tokenizer(targets, max_length=target_len, padding="max_length", truncation=True, return_tensors="pt")


        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label]
            for label in labels["input_ids"]
        ]
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs


    def preprocess_function(example):
        id2label = {0: "0", 1: "1"}

        inputs = " ".join([
            "sentence1:",
            example["sentence1"],
            "sentence2:",
            example["sentence2"],
        ])

        targets = id2label[example["label"]]


        return {"inputs": inputs, "targets": targets}



    processed_datasets = dataset.map(
        preprocess_function,
        remove_columns=dataset["train"].column_names,
        load_from_cache_file=False,
        desc="Running preprocessor on dataset",
    ).map(tokenizer_function,
        batched=True,
        load_from_cache_file=False,
        remove_columns=["inputs", "targets"],
        desc="Running tokenizer on dataset",
    )


    train_dataset = processed_datasets["train"].shuffle()
    eval_test = processed_datasets["validation"].train_test_split(0.5)
    eval_dataset = eval_test["train"]
    test_dataset = eval_test["test"]

    return train_dataset, eval_dataset, test_dataset