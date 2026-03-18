import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GPTJForCausalLM
from transformers import OPTForCausalLM
from transformers import GPTNeoXForCausalLM

def load_model(name, gpu):
    if name == "gpt-j-6B":
        model = GPTJForCausalLM.from_pretrained(
            "EleutherAI/gpt-j-6B",
            revision="float16",
            torch_dtype=torch.float16,
            cache_dir="../cache/gpt-j-6B",
        ).to(gpu)
        tokenizer = AutoTokenizer.from_pretrained(
            "EleutherAI/gpt-j-6B",
            cache_dir="../cache/gpt-j-6B",
        )

    elif name == "Llama-2-7b":
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            torch_dtype=torch.float16,
            cache_dir="../cache/Llama-2-7b-hf",
        ).to(gpu)
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            cache_dir="../cache/Llama-2-7b-hf",
        )
    
    elif name == "opt-6.7b":
        model = OPTForCausalLM.from_pretrained(
            "facebook/opt-6.7b",
            torch_dtype=torch.float16,
            cache_dir="../cache/opt-6.7b",
        ).to(gpu)
        tokenizer = AutoTokenizer.from_pretrained(
            "facebook/opt-6.7b",
            cache_dir="../cache/opt-6.7b",
        )

    elif name == "pythia-6.9b":
        model = GPTNeoXForCausalLM.from_pretrained(
            "EleutherAI/pythia-6.9b-v0",
            torch_dtype=torch.float16,
            revision="step143000",
            cache_dir="../cache/pythia-6.9b-v0/step143000",
        ).to(gpu)
        tokenizer = AutoTokenizer.from_pretrained(
            "EleutherAI/pythia-6.9b-v0",
            revision="step143000",
            cache_dir="../cache/pythia-6.9b-v0/step143000",
        )

    elif name == "gpt2":
        use_cuda = isinstance(gpu, str) and gpu.startswith("cuda")
        model = AutoModelForCausalLM.from_pretrained(
            "gpt2",
            torch_dtype=torch.float16 if use_cuda else None,
            cache_dir="../cache/gpt2",
        ).to(gpu)
        tokenizer = AutoTokenizer.from_pretrained(
            "gpt2",
            cache_dir="../cache/gpt2",
        )

    elif name == "Llama-3-8b":
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3-8B",
            torch_dtype=torch.float16,
            cache_dir="../cache/Llama-3-8B",
        ).to(gpu)
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Meta-Llama-3-8B",
            cache_dir="../cache/Llama-3-8B",
        )

    elif name == "Qwen2.5-7B":
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-7B",
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir="../cache/Qwen2.5-7B",
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-7B",
            cache_dir="../cache/Qwen2.5-7B",
        )

    elif name == "Qwen2.5-3B":
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-3B",
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir="../cache/Qwen2.5-3B",
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-3B",
            cache_dir="../cache/Qwen2.5-3B",
        )
    
    return model, tokenizer