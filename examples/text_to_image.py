# This is my first effort at making an interpolated dictionary
# This will be an image generator based on parquet data https://huggingface.co/datasets/yuvalkirstain/pickapic_v2/tree/main/data

import pandas as pd
from sentence_transformers import SentenceTransformer
from PIL import Image
import numpy as np
import io




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
data = ans.iloc[0]

print('ans', data)
print(ans['caption'][0])
jpg_0 = ans['jpg_0'][0]
jpg_1 = ans['jpg_1'][0]

img = Image.open(io.BytesIO(jpg_0))
arr = np.asarray(img)
print('arr', arr)

"""
from torchdata.datapipes.iter import FileLister
import torcharrow.dtypes as dt
DTYPE = dt.Struct([dt.Field("Values", dt.int32)])
ource_dp = FileLister(".", masks="df*.parquet")
parquet_df_dp = source_dp.load_parquet_as_df(dtype=DTYPE)
list(parquet_df_dp)[0]
"""