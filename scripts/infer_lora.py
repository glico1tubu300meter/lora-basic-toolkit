"""
Minimal LoRA inference script: compare base model vs LoRA-adapted output
for a given prompt.

Usage:
    conda activate lora
    python infer_lora.py --adapter ../out/lora_adapter "質問文をここに"
    python infer_lora.py --adapter ../out/lora_adapter --base_only "質問文をここに"
"""

import argparse
import sys

import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def parse_args():
    p = argparse.ArgumentParser(description="Compare base vs LoRA-adapted generation.")
    p.add_argument("prompt", help="Prompt to send to the model.")
    p.add_argument("--adapter", required=True, help="Path to the trained LoRA adapter directory.")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Base model name or path.")
    p.add_argument("--base_only", action="store_true", help="Skip the adapter, generate with the base model only.")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    return p.parse_args()


def generate(tokenizer, model, prompt, max_new_tokens, temperature):
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=False
    ).to(model.device)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=0.9,
        )
    generated = output[0][input_ids.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main():
    args = parse_args()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map={"": 0},
        dtype=torch.float16,
        disable_mmap=True,
    )

    print(f"[base] {generate(tokenizer, model, args.prompt, args.max_new_tokens, args.temperature)}\n")

    if not args.base_only:
        model = PeftModel.from_pretrained(model, args.adapter)
        print(f"[lora] {generate(tokenizer, model, args.prompt, args.max_new_tokens, args.temperature)}")


if __name__ == "__main__":
    main()
