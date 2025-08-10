# 6_backtesting.py
import pandas as pd
import numpy as np

def backtest(predictions_df):
    predictions_df["signal"] = np.where(predictions_df["predicted"] > predictions_df["actual"].shift(1), 1, -1)
    predictions_df["returns"] = predictions_df["signal"] * predictions_df["actual"].pct_change()
    cumulative_return = (1 + predictions_df["returns"]).cumprod() - 1
    sharpe_ratio = predictions_df["returns"].mean() / predictions_df["returns"].std() * np.sqrt(252)
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    return cumulative_return

if __name__ == "__main__":
    df = pd.read_csv("predictions.csv")
    backtest(df)
