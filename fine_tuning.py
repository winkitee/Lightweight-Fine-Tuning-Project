# Lightweight Fine-Tuning Project

# PEFT technique: LoRA
# Model: gpt2
# Evaluation approach: Accuracy
# Fine-tuning dataset: imdb 


# Setup

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current device: {device}")

if device == torch.device("cuda"):
    print(torch.cuda.get_device_name(0))

model_name = "gpt2"
num_labels = 2 # Positive, Negative

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token


# Loading and Evaluating a Foundation Model

pretrained_model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
).to(device)

if pretrained_model.config.pad_token_id is None:
    pretrained_model.config.pad_token_id = tokenizer.eos_token_id

dataset = load_dataset("imdb")
train_dataset = dataset["train"].shuffle(seed=42).select(range(2000))
test_dataset = dataset["test"].shuffle(seed=42).select(range(1000))

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

train_tokenized_dataset = train_dataset.map(tokenize_function, batched=True)
test_tokenized_dataset = test_dataset.map(tokenize_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

metric = evaluate.load('accuracy')

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return metric.compute(predictions=predictions, references=labels)

eval_args = TrainingArguments(
    output_dir="./pretrained-gpt2-eval",
    per_device_eval_batch_size=4,
)

eval_trainer = Trainer(
    model=pretrained_model,
    args=eval_args,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

pretrained_eval_result = eval_trainer.evaluate(eval_dataset=test_tokenized_dataset)
print(f"Pretrained Model Accuracy: {pretrained_eval_result['eval_accuracy']}")


# Performing Parameter-Efficient Fine-Tuning

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    AutoPeftModelForSequenceClassification,
)

config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_CLS,
)

lora_model = get_peft_model(pretrained_model, config).to(device)
lora_model.config.pad_token_id = tokenizer.eos_token_id
lora_model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir="lora-gpt2-imdb",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=2,
    learning_rate=1e-4,
    eval_strategy="epoch",
    logging_steps=500,
    save_strategy="epoch",
)

trainer_lora = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=train_tokenized_dataset,
    eval_dataset=test_tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer_lora.train()

peft_save_directory = "peft-lora-gpt2"
trainer_lora.model.save_pretrained(peft_save_directory)
print(f"PEFT LoRA '{peft_save_directory}' model saved.")


# Performing Inference with a PEFT Model

inference_model = AutoPeftModelForSequenceClassification.from_pretrained(peft_save_directory).to(device)
inference_model.config.pad_token_id = tokenizer.eos_token_id

inference_eval_args = TrainingArguments(
    output_dir="./peft-lora-gpt2-eval",
    per_device_eval_batch_size=4,
)

inference_trainer = Trainer(
    model=inference_model,
    args=inference_eval_args,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

lora_eval_result = inference_trainer.evaluate(eval_dataset=test_tokenized_dataset)
print(f"Pretrained Model Accuracy: {pretrained_eval_result['eval_accuracy']}")
print(f"PEFT LoRA Model Accuracy: {lora_eval_result['eval_accuracy']}")
print(f"Accuracy Improvement: {lora_eval_result['eval_accuracy'] - pretrained_eval_result['eval_accuracy']}")
