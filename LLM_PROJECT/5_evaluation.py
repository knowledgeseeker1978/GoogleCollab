# 5_evaluation.py
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate(model_path="fine_tuned_model", test_file="test.csv"):
    df = pd.read_csv(test_file)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    preds, true_vals = [], []
    for _, row in df.iterrows():
        inputs = tokenizer(row["text"], return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=5)
        prediction_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        try:
            predicted_price = float(prediction_text.split()[-1])
            preds.append(predicted_price)
            true_vals.append(row["label"])
        except:
            pass

    # Evaluation metrics
    mae = mean_absolute_error(true_vals, preds)
    rmse = mean_squared_error(true_vals, preds, squared=False)
    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")

if __name__ == "__main__":
    evaluate()
