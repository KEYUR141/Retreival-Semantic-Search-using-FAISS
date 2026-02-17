import numpy as np
import faiss
import json
from sentence_transformers import SentenceTransformer

embeddings = np.load("embeddings.npy")
with open("texts.json", "r", encoding = "utf-8") as f:
    texts = json.load(f)


dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
query = "What is Succession About and studio who created it?"
query_embeddings = model.encode([query]).astype('float32')

k=5
distances, indexes = index.search(query_embeddings, k)
print("Distances:", distances)
print("Indexes:", indexes)

for idex in indexes[0]:
    print(f"Text {idex+1}:", texts[idex])