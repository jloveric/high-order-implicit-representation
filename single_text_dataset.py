import torch
from torch.utils.data import Dataset
from typing import List


class SingleTextDataset(Dataset):
    def __init__(self, filenames: List[str], features: int = 10, targets: int = 10):
        feature_list, target_list = dataset_from_file(
            filenames[0], features=features, targets=targets)
        self.inputs = feature_list
        self.output = target_list
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.output)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return (self.input[idx]-128+0.5)/128.0, (self.output[idx]-128+0.5)/128.0


def ascii_to_float(ascii_tensor: torch.Tensor):
    return (ascii_tensor-128+0.5)/128


def float_to_ascii(float_tensor: torch.Tensor):
    return ((float_tensor+1.0)*128-0.5).int()


def generate_dataset(text_in: str, features: int, targets: int):

    udata = text_in.decode("utf-8")
    text = udata.encode("ascii", "ignore")

    final = len(text)-(targets+features)
    feature_list = []
    target_list = []
    for i in range(final):
        nf = [ord(val) for val in text[i:(i+features)]]
        feature_list.append(nf)
        nt = [ord(val) for val in text[(i+features):(i+features+targets)]]
        target_list.append(nt)

    return feature_list, target_list


def dataset_from_file(filename: str, features: int, targets: int):
    with open(filename, "r") as f:
        return generate_dataset(text_in=f.read(), features=features, targets=targets)
