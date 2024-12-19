import os
import pandas as pd


RESULT_PATH = os.path.join("logs", "LongForecasting")
OUTPUT_PATH = "extract_result.csv"


def extract_model_results(text: str):
    # 提取n_heads和d_model
    for line in text.split("\n"):
        if "n_heads" in line:
            n_heads = int(line.split("n_heads=")[1].split(",")[0])
        if "d_model" in line:
            d_model = int(line.split("d_model=")[1].split(",")[0])

    # 提取test mse
    mse = float(text.split("mse:")[1].split(",")[0])

    return {"n_heads": n_heads, "d_model": d_model, "mse": mse}


if __name__ == "__main__":
    files = os.listdir(RESULT_PATH)
    results = []
    for file in files:
        with open(os.path.join(RESULT_PATH, file), "r", encoding="utf-8") as f:
            text = f.read()
            results.append(extract_model_results(text))
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_PATH, index=False)
