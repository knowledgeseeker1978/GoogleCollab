# 7_inference.py
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "fine_tuned_model"

def predict(text_input):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    inputs = tokenizer(text_input, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=5)
    prediction_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return prediction_text

if __name__ == "__main__":
    example = "Date: 2025-08-10, Open: 150.76, Close: 152.30, Volume: 75000000, RSI: 58"
    print("Prediction:", predict(example))
