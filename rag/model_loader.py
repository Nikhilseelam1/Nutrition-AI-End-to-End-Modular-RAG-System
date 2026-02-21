# rag/model_loader.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.cuda.empty_cache()

def load_llm_model():
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    print(f"[INFO] Using model_id: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    torch.cuda.empty_cache()

    llm_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    ).to("cuda")

    print("Model loaded on:", next(llm_model.parameters()).device)

    return tokenizer, llm_model
