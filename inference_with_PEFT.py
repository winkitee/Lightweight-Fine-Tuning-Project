import torch
from transformers import GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset
from peft import AutoPeftModelForSequenceClassification
import evaluate

# Set device (use GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Load accuracy metric
metric = evaluate.load('accuracy')

def compute_metrics(eval_pred):
    """Compute accuracy from evaluation predictions."""
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Load and preprocess dataset
dataset = load_dataset("imdb")
test_dataset = dataset["test"].shuffle().select(range(100))  # Select a subset of 100 samples

def tokenize_function(examples):
    """Tokenize input text with padding and truncation."""
    return tokenizer(examples["text"], padding="max_length", truncation=True)

test_tokenized_dataset = test_dataset.map(tokenize_function, batched=True)

# Load pre-trained PEFT LoRA model
inference_model = AutoPeftModelForSequenceClassification.from_pretrained("peft-lora-gpt2").to(device)
inference_model.config.pad_token_id = tokenizer.eos_token_id  # Set padding token ID

# Define evaluation arguments
inference_eval_args = TrainingArguments(
    output_dir="./peft-lora-gpt2-eval",
    per_device_eval_batch_size=4,
)

# Initialize Trainer for evaluation
inference_trainer = Trainer(
    model=inference_model,
    args=inference_eval_args,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Run evaluation
lora_eval_result = inference_trainer.evaluate(eval_dataset=test_tokenized_dataset)
print(f"PEFT LoRA Model Accuracy: {lora_eval_result['eval_accuracy']}")