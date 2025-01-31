# Lightweight Fine-Tuning Project

This project fine-tunes the GPT-2 model using Parameter-Efficient Fine-Tuning (PEFT) with LoRA on the IMDB dataset for sentiment classification.

## Install Required Packages

Run the following command to install all dependencies:
```
pip install torch transformers datasets evaluate peft
```

## Running the Fine-Tuning Script

Run the script to perform fine-tuning using LoRA:

```
python fine_tuning.py
```

## Results

The script will print the accuracy of both the pre-trained and fine-tuned models.

The fine-tuned model will be saved to `peft-lora-gpt2`.

```
Pretrained Model Accuracy: 0.489
PEFT LoRA Model Accuracy: 0.895
Accuracy Improvement: 0.406
```