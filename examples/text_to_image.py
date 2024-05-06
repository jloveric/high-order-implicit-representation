# This is my first effort at making an interpolated dictionary
# This will be an image generator based on parquet data https://huggingface.co/datasets/yuvalkirstain/pickapic_v2/tree/main/data

import pandas as pd

ans = pd.read_parquet("train-00000-of-00645-b66ac786bf6fb553.parquet")
print('ans', ans)