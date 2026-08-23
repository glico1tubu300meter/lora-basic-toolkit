"""
Minimal, reusable QLoRA fine-tuning script.

Unlike the earlier NARUTO experiment (15-002-lora-finetune-qlora), this
script contains no franchise-specific content — all training data is
supplied externally as a JSONL file, so it can be reused for any dataset
without touching trademarked material.

Data format (JSONL, one example per line):
    {"prompt": "<user message>", "response": "<assistant message>"}

Usage:
    conda activate lora
    python train_lora.py --data ../data/sample_data.jsonl --output_dir ../out/lora_adapter

    # override hyperparameters as needed
    python train_lora.py --data my_data.jsonl --output_dir my_out \
        --model Qwen/Qwen2.5-7B-Instruct --r 16 --alpha 32 --epochs 3 --lr 2e-4
"""

import argparse
import json
import os
import sys

import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def parse_args():
    p = argparse.ArgumentParser(description="Minimal QLoRA fine-tuning on an external JSONL dataset.")
    p.add_argument("--data", required=True, help="Path to a JSONL file with {'prompt','response'} per line.")
    p.add_argument("--output_dir", required=True, help="Where to save the trained LoRA adapter.")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Base model name or path.")
    p.add_argument("--r", type=int, default=16, help="LoRA rank.")
    p.add_argument("--alpha", type=int, default=32, help="LoRA alpha (scaling = alpha/r).")
    p.add_argument("--dropout", type=float, default=0.05, help="LoRA dropout.")
    p.add_argument("--epochs", type=float, default=3.0, help="Number of training epochs.")
    p.add_argument("--lr", type=float, default=2e-4, help="Learning rate.")
    p.add_argument("--batch_size", type=int, default=2, help="Per-device train batch size.")
    p.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps.")
    p.add_argument("--max_seq_length", type=int, default=1024, help="Max tokenized sequence length.")
    p.add_argument(
        "--target_modules", nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        help="Which linear layers to attach LoRA adapters to.",
    )
    p.add_argument("--use_dora", action="store_true", help="Use DoRA instead of plain LoRA.")
    return p.parse_args()


def load_jsonl_dataset(path, tokenizer):
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            messages = [
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": row["response"]},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            examples.append({"text": text})
    if not examples:
        raise ValueError(f"No examples found in {path}")
    return Dataset.from_list(examples)


def main():
    args = parse_args()

    # Qwen2.5系の既定dtypeはbfloat16だが、Pascal世代(GTX1080Ti等)はbf16の勾配スケーリングに
    # 未対応のため、fp16に統一してAMP+GradScalerとの衝突を避ける。
    torch.set_default_dtype(torch.float16)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map={"": 0},
        dtype=torch.float16,
        disable_mmap=True,
    )

    lora_config = LoraConfig(
        r=args.r,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.target_modules,
        use_dora=args.use_dora,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_jsonl_dataset(args.data, tokenizer)
    print(f"Loaded {len(dataset)} training examples from {args.data}")

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        fp16=False,  # AMP+GradScalerがbf16由来テンソルと衝突するため無効化(LoRAアダプタはfp32のまま学習)
        bf16=False,
        logging_steps=1,
        save_strategy="epoch",
        report_to=[],
        dataset_text_field="text",
        max_length=args.max_seq_length,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()

    adapter_dir = os.path.join(args.output_dir, "lora_adapter")
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"\nSaved LoRA adapter to: {adapter_dir}")


if __name__ == "__main__":
    main()
