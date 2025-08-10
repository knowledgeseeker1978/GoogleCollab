# 1_environment_setup.py
# Verify GPU and install dependencies

import os
import subprocess

def setup_environment():
    print("🔍 Checking GPU availability...")
    os.system("nvidia-smi")

    print("📦 Installing required Python libraries...")
    packages = [
        "transformers[torch]",
        "datasets",
        "peft",
        "accelerate",
        "bitsandbytes",
        "yfinance",
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "wandb"
    ]
    subprocess.check_call(["pip", "install"] + packages)

if __name__ == "__main__":
    setup_environment()
