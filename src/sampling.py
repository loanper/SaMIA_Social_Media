from model_loader import load_model
from tqdm import tqdm
from utils import load_jsonl, add_jsonl
import argparse
import torch
from huggingface_hub import login
import os
import json

YOUR_HUGGINGFACE_TOKEN = ""  # <-- Colle ton token HuggingFace ici
if YOUR_HUGGINGFACE_TOKEN:
    login(YOUR_HUGGINGFACE_TOKEN)

def generate_text(model, tokenizer, prompt: str, device, max_new_tokens: int) -> str:
    tokenizer.pad_token = tokenizer.eos_token
    encoding = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    input_ids = encoding.input_ids.to(device)
    attention_mask = encoding.attention_mask.to(device)

    with torch.inference_mode():
        gen_token = model.generate(
            input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            temperature=0.2,
            max_new_tokens=max_new_tokens,
            top_k=50,
            top_p=0.8,
        )
    gen_text = tokenizer.batch_decode(gen_token, skip_special_tokens=False)[0]
    return gen_text


def load_processed_source_inputs(path: str):
    if not os.path.exists(path):
        return set()
    processed = set()
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            source = row.get("source_input")
            if isinstance(source, str) and source:
                processed.add(source)
    return processed

def get_prefix(text: str, prefix_ratio: float) -> str:
    num_words = len(text.split())
    if num_words <= 1:
        return text.strip()
    num_prefix_words = int(round(num_words * prefix_ratio))
    num_prefix_words = max(1, min(num_words - 1, num_prefix_words))
    prefix = " ".join(text.split()[:num_prefix_words])
    return prefix


def infer_max_new_tokens(text_length: int, prefix_ratio: float, user_value: int = None) -> int:
    if user_value is not None:
        return user_value
    suffix_words = max(1, int(round(text_length * (1 - prefix_ratio))))
    inferred = int(round(suffix_words * 1.6))
    return min(64, max(8, inferred))


def extract_generated_continuation(full_generation: str, prefix: str) -> str:
    text = full_generation.strip()
    if text.startswith(prefix):
        return text[len(prefix):].strip()
    return text

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", default="gpt-j-6B", choices=["gpt-j-6B", "opt-6.7b", "pythia-6.9b", "Llama-2-7b", "Llama-3-8b", "Qwen2.5-7B", "Qwen2.5-3B", "gpt2"], type=str)
    parser.add_argument("--text_length", default=32, choices=[32, 64, 128, 256], type=int)
    parser.add_argument("--num_samples", default=10, type=int)
    parser.add_argument("--prefix_ratio", default=0.5, type=float)
    parser.add_argument("--max_new_tokens", default=None, type=int)
    parser.add_argument("--input_file", type=str, default=None, help="Path to input JSONL file")
    parser.add_argument("--output_file", type=str, default=None, help="Path to output JSONL file")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to output_file instead of overwriting it.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing output_file by skipping already generated source_input lines (implies --append).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on how many NEW records to generate in this run (use with --resume for chunking).",
    )
    
    args = parser.parse_args()
    
    model_name = args.model_name
    text_length = args.text_length
    num_samples = args.num_samples
    prefix_ratio = args.prefix_ratio
    max_new_tokens = infer_max_new_tokens(text_length, prefix_ratio, args.max_new_tokens)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model, tokenizer = load_model(model_name, device)
    model.eval()

    if args.input_file:
        input_path = args.input_file
    else:
        input_path = f"./wikimia/{text_length}.jsonl"

    if args.output_file:
        output_path = args.output_file
    else:
        output_path = f"./sample/{model_name}/{text_length}.jsonl"

    print(f"Reading from {input_path}")
    print(f"Writing to {output_path}")
    print(f"Using max_new_tokens={max_new_tokens}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    processed = set()
    if args.resume:
        args.append = True
        processed = load_processed_source_inputs(output_path)
        print(f"Resume enabled: {len(processed)} already processed records found")

    if os.path.exists(output_path) and not args.append:
        open(output_path, "w", encoding="utf-8").close()
        print("Output file exists: overwriting (use --append to keep existing lines)")

    lines = load_jsonl(input_path)
    created = 0
    # tqdm total reflects the cap when using --limit.
    tqdm_total = args.limit if args.limit is not None else len(lines)
    for line in tqdm(lines, total=tqdm_total):
        new_line = {}
        full_input = line["input"]

        if args.resume and full_input in processed:
            continue

        prefix = get_prefix(full_input, prefix_ratio=prefix_ratio)
        new_line["input"] = prefix
        new_line["source_input"] = full_input
        new_line["prefix_ratio"] = prefix_ratio
        new_line["text_length"] = text_length
        for i in range(num_samples):
            full_generation = generate_text(model, tokenizer, prefix, device, max_new_tokens=max_new_tokens)
            new_line[f"output_{i}"] = full_generation
            new_line[f"continuation_{i}"] = extract_generated_continuation(full_generation, prefix)

        add_jsonl(new_line, output_path)
        if args.resume:
            processed.add(full_input)

        created += 1
        if args.limit is not None and created >= args.limit:
            print(f"Reached --limit={args.limit}; stopping")
            break

