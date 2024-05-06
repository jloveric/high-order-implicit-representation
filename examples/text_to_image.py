# This is my first effort at making an interpolated dictionary
# This will be an image generator based on parquet data https://huggingface.co/datasets/yuvalkirstain/pickapic_v2/tree/main/data

import pandas as pd
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

# Our sentences we like to encode
sentences = [
    "This framework generates embeddings for each input sentence",
    "Sentences are passed as a list of string.",
    "The quick brown fox jumps over the lazy dog.",
]

# Sentences are encoded by calling model.encode()
sentence_embeddings = model.encode(sentences)

# Print the embeddings
for sentence, embedding in zip(sentences, sentence_embeddings):
    print("Sentence:", sentence)
    print("Embedding:", len(embedding))
    print("")

ans = pd.read_parquet("train-00000-of-00645-b66ac786bf6fb553.parquet")
print('ans', ans)