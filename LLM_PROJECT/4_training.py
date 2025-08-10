# 4_training.py
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch
from model_setup import load_model

MODEL_DIR = "fine_tuned_model"

def main():
    train_ds = load_dataset("csv", data_files={"train": "train.csv"})["train"]
    test_ds = load_dataset("csv", data_files={"test": "test.csv"})["test"]

    model, tokenizer = load_model()

    train_ds = train_ds.map(lambda batch: tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128), batched=True)
    test_ds = test_ds.map(lambda batch: tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128), batched=True)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        eval_strategy="epoch",
        learning_rate=2e-4,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        fp16=True,
        save_strategy="epoch",
        logging_dir="./logs"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=data_collator,
    )

    trainer.train()
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

if __name__ == "__main__":
    main()
