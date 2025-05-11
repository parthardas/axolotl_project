import pandas as pd
import json

df = pd.read_csv("./data/train.csv", header=None, names=["label", "text"])

with open("./data/sentiment.jsonl", "w") as f:
    for _, row in df.iterrows():
        json.dump({
            "instruction": "Classify the sentiment of this comment",
            "input": row["text"],
            "output": "positive" if row["label"] == 1 else "negative"
        }, f)
        f.write("\n")
