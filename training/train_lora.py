"""
LoRA Fine-Tuning Pipeline for Qwen2.5-0.5B on Apple Silicon M4.

Custom multi-domain specialization:
  1. Medical & Pharmacological Reasoning
  2. Film Direction & Screenwriting Theory
  3. Advanced Systems & Concurrency Code
  4. Academic Research Methodology
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType


def load_conversational_dataset(jsonl_path: Path, tokenizer) -> Dataset:
    """Load JSONL dataset and format into tokenized input_ids using model's chat template."""
    with open(jsonl_path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    formatted_texts = []
    for row in rows:
        messages = row["messages"]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        formatted_texts.append(text)

    raw_dataset = Dataset.from_dict({"text": formatted_texts})

    def tokenize_fn(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=1024,
            padding=False,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_dataset = raw_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    return tokenized_dataset


def main():
    base_model_dir = Path("/Users/ivinbaiju/Stark/Markv0.0/models/checkpoints/qwen2.5-0.5b")
    output_merged_dir = Path("/Users/ivinbaiju/Stark/Markv0.0/models/checkpoints/qwen2.5-0.5b-expert")
    adapter_output_dir = Path("/Users/ivinbaiju/Stark/Markv0.0/training/adapters")
    dataset_file = Path("/Users/ivinbaiju/Stark/Markv0.0/training/dataset.jsonl")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"=== Starting LoRA Fine-Tuning on {device.upper()} (Apple M4) ===")

    # 1. Load Tokenizer
    print(f"Loading tokenizer from {base_model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(str(base_model_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Dataset
    print(f"Loading and formatting dataset from {dataset_file}...")
    train_dataset = load_conversational_dataset(dataset_file, tokenizer)
    print(f"Dataset ready: {len(train_dataset)} high-standard multi-domain samples.")

    # 3. Load Base Model in FP16/BF16
    print(f"Loading base model onto {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        str(base_model_dir),
        dtype=torch.float32 if device == "cpu" else torch.float16,
    ).to(device)

    # 4. Attach LoRA Configuration
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    print("LoRA Model Parameter Summary:")
    model.print_trainable_parameters()

    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir=str(adapter_output_dir),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        num_train_epochs=5,
        logging_steps=1,
        save_strategy="no",
        fp16=True if device == "mps" else False,
        use_cpu=True if device == "cpu" else False,
        report_to="none",
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # 6. Execute Training
    print("Beginning LoRA Training Epochs...")
    train_result = trainer.train()
    print(f"Training Complete! Final Loss: {train_result.training_loss:.4f}")

    # 7. Merge Adapters into Standalone Base Model
    print("Merging trained LoRA adapters back into base weights for zero-overhead inference...")
    merged_model = model.merge_and_unload()

    output_merged_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(output_merged_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_merged_dir))

    print(f"✓ Standalone fine-tuned model saved to: {output_merged_dir}")
    print("Ready to run in Mark v0.0 Custom Inference Engine!")


if __name__ == "__main__":
    main()
