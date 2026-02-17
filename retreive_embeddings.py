import numpy as np
import json

embeddings = np.load("embeddings.npy")
print(embeddings)

with open("texts.json", "r", encoding="utf-8") as f:
    texts = json.load(f)
    # print(texts)    