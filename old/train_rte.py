from datasets import load_dataset
from peft import PromptTuningConfig, get_peft_model, PeftType, TaskType, PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, GenerationConfig, default_data_collator
from safetensors import safe_open
import wandb
import torch
import numpy as np

device = "cuda"
source_len = 128
target_len = 3
num_virtual_tokens = 100
model_name_or_path = "fft_mrpc_rte"
tokenizer_name_or_path = "t5-base"
origin_prompt_save = "./origin"
lr = 0.3
batch_size = 32
num_epochs = 5
warmup_steps = 500
weight_decay = 1e-5

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, model_max_length=512, use_fast=True)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

peft_config = PromptTuningConfig(peft_type=PeftType.PROMPT_TUNING, task_type=TaskType.SEQ_2_SEQ_LM, num_virtual_tokens=num_virtual_tokens)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
model.resize_token_embeddings(len(tokenizer))

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# print(model.prompt_encoder.default.embedding.weight)

origin_emb_weights = safe_open(f"{origin_prompt_save}/adapter_model.safetensors", framework="pt", device="cpu").get_slice("prompt_embeddings")[:, :]
# print(origin_emb_weights)

model.prompt_encoder.default.embedding.weight = torch.nn.Parameter(origin_emb_weights)
# print(model.prompt_encoder.default.embedding.weight)

from preprocess import get_rte
train_dataset, eval_dataset, test_dataset = get_rte(tokenizer=tokenizer, source_len=source_len, target_len=target_len)

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    
    preds[preds==-100] = tokenizer.pad_token_id
    labels[labels==-100] = tokenizer.pad_token_id

    # print(preds, labels)
    preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # print(preds, labels)
    
    correct = 0
    total = 0
    for pred, true in zip(preds, labels):
        if pred.strip() == true.strip():
            correct += 1
        total += 1
    accuracy = correct / total
    return {"accuracy": accuracy}


training_args = Seq2SeqTrainingArguments(
    "out",
    per_device_train_batch_size=batch_size,
    learning_rate=lr,
    num_train_epochs=num_epochs,
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="no",
    predict_with_generate=True,
    generation_config=GenerationConfig(max_new_tokens=target_len),
    weight_decay=weight_decay,
    warmup_steps=warmup_steps
)

trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,
)

model.print_trainable_parameters()
trainer.train()

trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")

if wandb.run is not None:
    wandb.finish()

model.save_pretrained("./rte")

model.print_trainable_parameters()