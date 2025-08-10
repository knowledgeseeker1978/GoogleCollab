# 3_model_setup.py
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import torch

MODEL_NAME = "gpt2"  # can also use "EleutherAI/pythia-1b" or quantized 7B

def load_model():
    print("🔍 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    print("🔍 Loading model with 8-bit precision...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        load_in_8bit=True,
        device_map="auto",
    )

    print("⚙️ Applying LoRA configuration...")
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["c_attn"]
    )
    model = get_peft_model(model, peft_config)

    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = load_model()
    model.save_pretrained("base_model")
    tokenizer.save_pretrained("base_model")
